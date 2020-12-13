import os

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer
import pandas as pd


class RedditDataSet(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        text = row['text']
        title = row['title']
        label = row['annotation']
        text_ids = self.transform(text)
        title_ids = self.transform(title)
        return {'title': title,
                'title_ids': title_ids,
                'text': text,
                'text_ids': text_ids,
                'label': label,
                'id': item,
                }


class RedditDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only
        # from a single GPU in distributed settings.
        self.transform = BertTokenizer('bert-base-uncased')

    def setup(self, stage=None):
        # There are also data operations you might want to perform on every GPU.
        data_file = os.path.join(self.data_dir, 'debug.csv')
        dataset = RedditDataSet(data_file, transform=self.transform)
        self.train, self.val, self.test = random_split(dataset, [.8, .1, .1])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)
