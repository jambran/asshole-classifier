import csv
import os

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from transformers import BertTokenizer


class RedditDataSet(Dataset):
    def __init__(self, csv_file, transform):
        self.csv_file = csv_file
        self.transform = transform
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
                d['is_asshole'] = int(d['is_asshole'])
                yield d

    def __len__(self):
        return len(self.id_to_instance)

    def __getitem__(self, item):
        attributes_dict = self.id_to_instance[item]
        title = self.transform(#attributes_dict['title'],
                               attributes_dict['text'],
                               max_length=400,
                               truncation=True,
                               padding='max_length',
                               return_tensors='pt',
                               )
        # text = self.transform(attributes_dict['text'])  # consider adding this feature later
        to_add = {'title_ids': title['input_ids'],
                  'attention_mask': title['attention_mask']}
        result = {**attributes_dict, **to_add}
        return result


class RedditDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 data_filename: str,
                 batch_size: int = 32,
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.data_filename = data_filename
        self.batch_size = batch_size

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only
        # from a single GPU in distributed settings.
        self.transform = BertTokenizer.from_pretrained('bert-base-uncased', max_length=32)

    def setup(self, stage=None):
        # There are also data operations you might want to perform on every GPU.
        data_file = os.path.join(self.data_dir, self.data_filename)
        dataset = RedditDataSet(data_file, transform=self.transform)
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
