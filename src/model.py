import pytorch_lightning as pl
import torch
import sklearn

class AssholeClassifier(pl.LightningModule):
    def __init(self):
        super().__init__()


    def forward(self):
        pass

    def prepare_data(self):
        # set self.train_ds, self.val_ds, self.test_ds
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds,
                                           batch_size=self.batch_size,
                                           drop_last=False,
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds,
                                           batch_size=self.batch_size,
                                           drop_last=False,
                                           )
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.learning_rate,
                                )
    def step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
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

    def train_step(self, batch, batch_idx):
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
