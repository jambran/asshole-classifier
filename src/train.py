'''
script to train the model on the processed data
'''
import os

import torch
import pytorch_lightning as pl
from model import AssholeClassifier
from reddit_data_module import RedditDataModule


if __name__ == '__main__':
    torch.manual_seed(2)
    train_file = os.path.join('..', 'data', 'raw')
    data_module = RedditDataModule(data_dir=train_file)
    model = AssholeClassifier(learning_rate=1E-3,
                              possible_labels=[0, 1])
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger('logs', name='experiment', version=0),
    )
    trainer.fit(model, data_module)
    print('all done')