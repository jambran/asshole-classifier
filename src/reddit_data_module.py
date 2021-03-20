import csv
import os
from collections import Counter

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from transformers import BertTokenizer

from class_enum import ClassEnum


class RedditDataSet(Dataset):
    def __init__(self, csv_file, transform, title_max_length=32, text_max_length = 256):
        self.csv_file = csv_file
        self.transform = transform
        self.title_max_length = title_max_length
        self.text_max_length = text_max_length
        self.id_to_instance = {}
        for i, instance in enumerate(self):
            self.id_to_instance[i] = instance

    def __iter__(self):
        with open(self.csv_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='|')
            headers = next(reader)
            for instance in reader:
                d = {headers[i]: instance[i] for i in range(len(headers))}
                # need to convert the strings to ints: '0' -> 0
                d['is_asshole'] = ClassEnum[d['is_asshole']].value
                yield d

    def __len__(self):
        return len(self.id_to_instance)

    def __getitem__(self, item):
        attributes_dict = self.id_to_instance[item]
        title = self.transform(attributes_dict['title'],
                               max_length=self.title_max_length,
                               truncation=True,
                               padding='max_length',
                               return_tensors='pt',
                               )
        text = self.transform(attributes_dict['text'],
                              max_length=self.text_max_length,
                              truncation=True,
                              padding='max_length',
                              return_tensors='pt',
                              )
        to_add = {'title_ids': title['input_ids'],
                  'title_attention_mask': title['attention_mask'],
                  'text_ids': text['input_ids'],
                  'text_attention_mask': text['attention_mask'],
                  }
        result = {**attributes_dict, **to_add}
        return result

    def get_class_frequencies(self):
        c = Counter(item['is_asshole'] for item in self)
        frequencies = [c[i] for i in range(len(c))]
        return frequencies


class RedditDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 data_filename: str,
                 batch_size: int = 32,
                 title_max_length: int = 32,
                 text_max_length: int = 256,
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.data_filename = data_filename
        self.batch_size = batch_size
        self.title_max_length = title_max_length
        self.text_max_length = text_max_length

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only
        # from a single GPU in distributed settings.
        self.transform = BertTokenizer.from_pretrained('bert-base-uncased')

    def setup(self, stage=None):
        # There are also data operations you might want to perform on every GPU.
        data_file = os.path.join(self.data_dir, self.data_filename)
        dataset = RedditDataSet(data_file,
                                transform=self.transform,
                                title_max_length=self.title_max_length,
                                text_max_length=self.text_max_length)
        total_instances = len(dataset)
        num_train, num_val = int(total_instances * .8), int(total_instances * .1)
        num_test = total_instances - num_train - num_val
        self.train, self.val, self.test = random_split(dataset, [num_train, num_val, num_test])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
