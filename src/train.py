'''
script to train the model on the processed data
'''
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import AssholeClassifier
from reddit_data_module import RedditDataModule
from class_enum import ClassEnum

if __name__ == '__main__':
    exp_name = 'debug'
    torch.manual_seed(2)
    train_dir = os.path.join('..', 'data', 'raw')
    data_module = RedditDataModule(data_dir=train_dir,
                                   data_filename='top-850.csv',
                                   title_max_length=16,
                                   text_max_length=256,
                                   )
    possible_labels = [label.name for label in ClassEnum]
    data_module.prepare_data()
    data_module.setup() # need to run this to access data_module.train
    model = AssholeClassifier(learning_rate=.03,
                              possible_labels=possible_labels,
                              class_frequencies=data_module.train.dataset.get_class_frequencies(),
                              use_title=True,
                              use_text=True,
                              )
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=0,  # (1 if torch.cuda.is_available() else 0),
        max_epochs=1,
        fast_dev_run=exp_name == 'debug',
        logger=pl.loggers.TensorBoardLogger('logs', name=exp_name, version=0),
        callbacks=[ModelCheckpoint(save_top_k=-1,
                                   save_last=True),
                   ]
    )
    trainer.fit(model, data_module)
    print('all done')
