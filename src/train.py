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
    # for lr in [.03, .01, .003, .001]:

    # could try text
    # could try title + tldr
    # could try title + text
    lr = .01
    exp_name = f'text-lr={lr}'
    print(f"starting exp {exp_name}")
    torch.manual_seed(2)
    train_dir = os.path.join('..', 'data', 'raw')
    data_module = RedditDataModule(data_dir=train_dir,
                                   data_filename='10000.csv')
    model = AssholeClassifier(learning_rate=lr,
                              possible_labels=[0, 1])
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=0,  # (1 if torch.cuda.is_available() else 0),
        max_epochs=10,
        fast_dev_run=exp_name == 'debug',
        logger=pl.loggers.TensorBoardLogger('logs', name=exp_name, version=0),
        callbacks=[ModelCheckpoint(save_top_k=-1,
                                   save_last=True),
                   ]
    )
    trainer.fit(model, data_module)
    print('all done')
