'''
script to train the model on the processed data
'''
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import AssholeClassifier
from reddit_data_module import RedditDataModule

if __name__ == '__main__':
    exp_name = 'debug'
    torch.manual_seed(2)
    train_dir = os.path.join('..', 'data', 'raw')
    data_module = RedditDataModule(data_dir=train_dir,
                                   data_filename='5000.csv')
    model = AssholeClassifier(learning_rate=1E-3,
                              possible_labels=[0, 1])
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=0,  # (1 if torch.cuda.is_available() else 0),
        max_epochs=10,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger('logs', name='10-epochs', version=0),
        callbacks=[ModelCheckpoint(period=1,
                                   save_last=True),
                   ]
    )
    trainer.fit(model, data_module)
    print('all done')
