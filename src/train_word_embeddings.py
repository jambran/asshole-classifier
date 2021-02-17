from pathlib import Path

import gensim
from nltk.tokenize import word_tokenize

from reddit_data_module import RedditDataSet
import scipy
import logging



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        handlers=[logging.FileHandler('word_embeds.log'),
                                  logging.StreamHandler()],
                        )
    logging.info('prepping sentences')
    train_file = Path('../data/raw/10000.csv')
    data_set = RedditDataSet(csv_file=str(train_file),
                             transform=None)

    sentences = []
    for instance in data_set:
        title_tokens = word_tokenize(instance['title'])
        text_tokens = word_tokenize(instance['text'])
        sentences.append(title_tokens)
        sentences.append(text_tokens)

    sent_lens = [len(word_tokenize(instance['text'])) for instance in data_set]
    logging.info(f'mean sentence length = {scipy.mean(sent_lens)}')
    logging.info(f'median sentence length = {scipy.median(sent_lens)}')


    logging.info('begin model training')
    ft_model = gensim.models.fasttext.FastText(sentences=sentences)

    logging.info('saving the model')
    model_save_path = Path('../models/word_embeddings/ft.model')
    ft_model.save(str(model_save_path))
    logging.info('model saved')
