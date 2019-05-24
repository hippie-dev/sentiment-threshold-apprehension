import os

import pandas as pd
import re

from bs4 import BeautifulSoup
from nltk import TweetTokenizer

from keras.k_utils import create_glove_embeddings, word_embedding_lookup
from utils import pad_data_for_cnn


def read_tweet_csv_file(opt, remove_quotes=True):
    """
    Remove " from the files as it causes issues
    """
    csv_path = opt.real_dataset_path
    if remove_quotes:
        with open(csv_path, encoding="utf8") as fin:
            new_text = re.sub('\"',  '', fin.read())
        with open(csv_path, "w", encoding="utf8") as f:
            f.write(new_text)

    header_names = {'uuid', 'tweet_id', 'user_name', 'screen_name', 'tweet', 'date_time', 'retweet_count', 'fav_count',
                    'link'}
    csv_file = pd.read_csv(csv_path,
                           encoding='ISO-8859-1',
                           header=None,
                           names=header_names,
                           sep=';',
                           low_memory=False)
    csv_file.columns = csv_file.iloc[0]
    csv_file = csv_file.reindex(csv_file.index.drop(0))
    return csv_file


def tweet_cleaner(raw_data):
    soup = BeautifulSoup(raw_data, 'lxml')
    # HTML decoding
    soup_data = soup.get_text()
    # Remove @ Mentions and http links
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    pat3 = r'http?://[A-Za-z0-9./]+'
    pat4 = r"(?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})"
    url_pat = r"[A-Za-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$"

    combined_pat = r'|'.join((pat1, pat3, pat4, url_pat))
    stripped = re.sub(combined_pat, '', soup_data)
    # Remove non letter characters
    letters_only = re.sub("[^a-zA-Z0-9_.\"\'/$-]", " ", stripped)
    lower_case = letters_only.lower()

    tw = TweetTokenizer()
    tokenized_sent = " ".join([item.strip() for item in tw.tokenize(lower_case)])
    # print(" ".join([item.strip() for item in tw.tokenize(lower_case)]))
    return tokenized_sent
    # Remove tokenize as of now
    # words = tok.tokenize(lower_case)
    # return (" ".join(words)).strip()
    # return lower_case


def get_data(opt):
    print('Reading real data')
    r_data = read_tweet_csv_file(opt, remove_quotes=True)
    print('Data shape {}'.format(r_data.shape))
    r_data.drop(['tweet_id', 'user_name', 'screen_name', 'retweet_count', 'fav_count',
                 'link'], axis=1, inplace=True)
    # Drop NaN
    r_data = r_data[pd.notnull(r_data['tweet'])]
    r_data['tweet'] = r_data['tweet'].map(lambda x: tweet_cleaner(x))
    # Remove tweets whose length is less than 3 or 5
    mask = r_data['tweet'].astype('str').str.len() > 3
    r_data = r_data.loc[mask]
    print('Saving to CSV file')
    csv_save_path = os.path.join(opt.cleaned_tweet_save_path)
    r_data.to_csv(csv_save_path, index=None, header=True)
    print('Saved')

    return r_data


def prepare_real_data(real_data, opt):
    X = real_data['tweet']

    transfer_word_to_ix = {
        'END': 0,
        'UNK': 1,
    }
    transfer_ix_to_word = {
        0: 'END',
        1: 'UNK',
    }

    count = 2

    for sent in X:
        words = sent.split()
        for word in words:
            if word.strip() in transfer_word_to_ix:
                continue
            else:
                transfer_word_to_ix[word] = count
                transfer_ix_to_word[count] = word
                count += 1

    x_real_test = []
    for sent in X:
        words = sent.split()
        encoded_words = []
        for word in words:
            if word in transfer_word_to_ix:
                encoded_words.append(transfer_word_to_ix[word])
            else:
                encoded_words.append(transfer_word_to_ix['UNK'])
        x_real_test.append(encoded_words)

    x_real_test_encoded = pad_data_for_cnn(x_real_test, opt)
    real_embedding_matrix = create_glove_embeddings(word_index=transfer_word_to_ix,
                                                    total_words=len(transfer_ix_to_word),
                                                    embed_size=opt.embed_dim,
                                                    max_seq_len=opt.sent_len,
                                                    padded_data=x_real_test_encoded,
                                                    glove_path=opt.glove_path,
                                                    save_path=opt.embedding_save_path)

    real_embedded_test_input = word_embedding_lookup(x=x_real_test_encoded,
                                                     max_seq_length=opt.sent_len,
                                                     num_words=len(transfer_ix_to_word),
                                                     embedding_dim=opt.embed_dim,
                                                     weights=real_embedding_matrix)
    real_embedded_test_input = real_embedded_test_input.reshape(real_embedded_test_input.shape[0],
                                                                real_embedded_test_input.shape[1],
                                                                real_embedded_test_input.shape[2], 1)
    return real_embedded_test_input, x_real_test_encoded

