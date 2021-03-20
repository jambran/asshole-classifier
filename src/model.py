import pytorch_lightning as pl
import torch
import sklearn
from transformers import BertForSequenceClassification

from plotting_metrics import plot_confusion_matrix


def get_weight_tensor_from_class_frequencies(frequencies):
    return torch.tensor([1 / frequency for frequency in frequencies])


class AssholeClassifier(pl.LightningModule):
    def __init__(self, learning_rate, possible_labels, class_frequencies=None,
                 use_title=True, use_text=True):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                   num_labels=len(possible_labels),
                                                                   output_attentions=True,
                                                                   output_hidden_states=False)
        self.learning_rate = learning_rate
        self.possible_labels = possible_labels
        self.use_title = use_title
        self.use_text = use_text
        if not (use_title or use_text):
            raise ValueError(f'At least one of `use_title` and `use_text` must be true. '
                             f'Received use_title={use_title} and use_text={use_text}.')
        if class_frequencies is None:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=get_weight_tensor_from_class_frequencies(class_frequencies))

        # metrics for evaluation
        self.acc = pl.metrics.Accuracy()
        self.p = pl.metrics.Precision()
        self.r = pl.metrics.Recall()
        self.f1 = pl.metrics.F1()

        self.save_hyperparameters()

    def forward(self, text_ids, attention_mask):
        logits = self.model(text_ids,
                            attention_mask=attention_mask)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                )

    def step(self, batch, batch_idx):
        ids = torch.LongTensor()
        attention_mask = torch.LongTensor()
        if self.use_title:
            ids = torch.cat((ids, batch['title_ids'].squeeze()), dim=1)
            attention_mask = torch.cat((attention_mask, batch['title_attention_mask'].squeeze()), dim=1)
        if self.use_text:
            ids = torch.cat((ids, batch['text_ids'].squeeze()), dim=1)
            attention_mask = torch.cat((attention_mask, batch['text_attention_mask'].squeeze()), dim=1)
        output = self.forward(ids,
                              attention_mask)
        logits = output['logits']
        loss = self.loss(logits, batch['is_asshole'])
        predictions = (logits.argmax(-1)).float()
        return {'loss': loss,
                'predictions': predictions,
                'labels': batch['is_asshole'],
                }

    def epoch_end(self, outputs, data_name):
        predictions = torch.cat([o['predictions'] for o in outputs], 0)
        labels = torch.cat([o['labels'] for o in outputs], 0)
        prf1 = self.calculate_prf1(predictions, labels)
        for key, value in prf1.items():
            self.log(f"{data_name}_{key}", value)
        loss = sum([o['loss'] for o in outputs])
        self.log(f'{data_name}_loss', loss)

        figure = plot_confusion_matrix(labels.cpu(),
                                       predictions.cpu(),
                                       class_names=self.possible_labels)
        self.logger.experiment.add_figure(f'{data_name}_cm_epoch={self.current_epoch}', figure)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, 'val')

    def calculate_prf1(self, predictions, labels):
        return {'p': self.p(predictions, labels),
                'r': self.r(predictions, labels),
                'f1': self.f1(predictions, labels),
                'acc': self.acc(predictions, labels),
                }
