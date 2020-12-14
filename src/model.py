import pytorch_lightning as pl
import torch
import sklearn
from transformers import BertForSequenceClassification

from plotting_metrics import plot_confusion_matrix


class AssholeClassifier(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.learning_rate = learning_rate


    def forward(self, title_ids, text_ids):
        self.model(title_ids)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                )
    def step(self, batch, batch_idx):
        logits = self.forward(batch['title_ids'],
                              batch['text_ids'])
        loss = self.loss(logits, batch['label'])
        predictions = (logits.argmax(-1)).float()
        return {'loss': loss,
                'predictions': predictions,
                'labels': batch['labels'],
                }

    def epoch_end(self, outputs):
        predictions = torch.cat([o['predictions'] for o in outputs], 0)
        labels = torch.cat([o['labels'] for o in outputs], 0)
        prf1 = self.calculate_prf1(labels, predictions)
        for key, value in prf1.items():
            self.log(f"train_{key}", value)
        loss = sum([o['loss'] for o in outputs])
        self.log('train_loss', loss)

        figure = plot_confusion_matrix(labels, predictions, class_names = self.test_ds.get_labels())
        self.logger.experiment.add_figure(f'train_cm_epoch={self.current_epoch}', figure)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def train_epoch_end(self, outputs):
        self.epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs)

    def calculate_prf1(self, y_true, y_pred):
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        return {'p': precision,
                'r': recall,
                'f1': f1,
                'acc': acc,
                }
