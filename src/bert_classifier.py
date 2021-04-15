'''
based on tutorial [here](https://github.com/yk/huggingface-nlp-demo)
'''

import pytorch_lightning as pl
import torch
import transformers
from torch import nn

from bias_detection.plotting_metrics import plot_confusion_matrix, plot_prf1_per_class, plot_label_analysis


class BiasClassifier(pl.LightningModule):
    def __init__(self, bert_model, epochs, learning_rate, momentum, class_names,
                 class_frequencies=None, freeze_bert=True, weight_classes='none',
                 num_ff_layers=1, hidden_size=40, dropout=.1):
        super().__init__()
        self.model_type = bert_model
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.drop = nn.Dropout(p=dropout)  # try removing this?
        bert_outsize = 768
        ## print the model structure
        if num_ff_layers == 0:
            self.out = nn.Linear(bert_outsize, len(class_names))
        else:
            # start with layer that has input bert_outsize and output the desired hidden size
            modules = [nn.Linear(bert_outsize, hidden_size)]
            modules.append(nn.Dropout(p=dropout))
            for i in range(num_ff_layers):
                modules.append(nn.Linear(hidden_size, hidden_size))
                modules.append(nn.Dropout(p=dropout))
            # end with a layer that connects the hidden size layers to the desired number of outputs
            modules.append(nn.Linear(hidden_size, len(class_names)))
            self.out = nn.Sequential(*modules)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.class_names = class_names
        self.weight_classes = weight_classes

        if freeze_bert:
            # freeze bert layer - only train classifier layer
            for param in self.bert.parameters():
                param.requires_grad = False

        if weight_classes == 'freq':
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_frequencies).float())
        elif weight_classes == 'inv_freq':
            weight = [1 / freq for freq in class_frequencies]
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).float())
        elif weight_classes == 'incremental':
            weight = [10] * (len(self.class_names) - 1) + [1.0]
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).float())
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler_output = self.bert(input_ids, attention_mask)
        dropped = self.drop(pooler_output)
        logits = self.out(dropped)
        return logits

    def step(self, batch, batch_idx):
        labels = batch['label']
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = self.loss(logits, labels)
        predictions = (logits.argmax(-1)).long()
        return {'loss': loss,
                'predictions': predictions,
                'labels': labels,
                'annotations': batch['annotated_label'],
                }

    def training_step(self, batch, batch_idx):
        outputs = self.step(batch, batch_idx)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.step(batch, batch_idx)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.step(batch, batch_idx)
        return outputs

    def epoch_end(self, outputs, data_set):
        predictions = torch.cat([o['predictions'] for o in outputs], 0)
        labels = torch.cat([o['labels'] for o in outputs], 0)
        annotations = [annotation for o in outputs for annotation in o['annotations']]
        prf1 = self.calculate_prf1(labels, predictions)
        for key, value in prf1.items():
            if key.endswith('none'):
                continue
            self.log(f'{key}_{data_set}', value)
        figure = plot_prf1_per_class(prf1, class_names=self.class_names)
        self.logger.experiment.add_figure(f"epoch={self.current_epoch}_prf1-per-class-{data_set}", figure)
        loss = sum([o['loss'] for o in outputs])
        self.log(f'loss_{data_set}', loss)

        figure = plot_confusion_matrix(labels, predictions, class_names=self.class_names)
        self.logger.experiment.add_figure(f'epoch={self.current_epoch}_{data_set}_cm', figure)
        analysis_fig = plot_label_analysis(predictions, annotations, self.class_names)
        self.logger.experiment.add_figure(f'epoch={self.current_epoch}_analysis_{data_set}', analysis_fig)

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, 'train')
        if (self.current_epoch == 0):
            self.make_computational_graph()

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

    def calculate_prf1(self, y_true, y_pred):
        results_pl = {}
        for average_type in ['weighted', 'micro', 'macro', 'none']:
            precision = pl.metrics.functional.classification.precision(y_pred, y_true,
                                                                       num_classes=len(self.class_names),
                                                                       class_reduction=average_type)
            recall = pl.metrics.functional.classification.recall(y_pred, y_true, num_classes=len(self.class_names),
                                                                 class_reduction=average_type)
            f1 = pl.metrics.functional.f1(y_pred, y_true, num_classes=len(self.class_names), average=average_type)
            results_pl[f'p-{average_type}'] = precision
            results_pl[f'r-{average_type}'] = recall
            results_pl[f'f1-{average_type}'] = f1
        acc = pl.metrics.functional.classification.accuracy(y_pred, y_true)
        results_pl['acc'] = acc

        return results_pl

    def make_computational_graph(self):
        """
        following https://learnopencv.com/tensorboard-with-pytorch-lightning/
        :return:
        """
        device = self.device
        sample_input = {'input_ids': torch.tensor([[101, 2307, 3105, 999, 102, 0, 0, 0]], device=device),
                        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], device=device),
                        'label': torch.tensor([0], device=device),
                        # 'annotated_label': ['0'],
                        }
        self.logger.experiment.add_graph(self,
                                         (sample_input['input_ids'], sample_input['attention_mask']),
                                         )

    @classmethod
    def from_hparams(cls, hparams_file):
        if type(hparams_file) is str:
            with open(hparams_file, 'r') as hp:
                hparams = yaml.safe_load(hp)
        elif type(hparams_file) is dict:
            hparams = hparams_file
        else:
            raise TypeError(f"Type of `hparams_file` must be stirng or dictionary. Received {type(hparams_file)}")
        return cls(bert_model=hparams['bert_model'],
                   epochs=hparams['epochs'],
                   learning_rate=hparams['learning_rate'],
                   momentum=hparams['momentum'],
                   class_names=hparams['class_names'],
                   class_frequencies=hparams['class_frequencies'],
                   freeze_bert=hparams['freeze_bert'],
                   weight_classes=hparams['weight_classes'],
                   num_ff_layers=hparams['num_ff_layers'],
                   hidden_size=hparams['hidden_size'],
                   dropout=hparams['dropout'],
                   )
