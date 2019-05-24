import os
import numpy as np
import _pickle as cPickle
import mmap

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding
from tqdm import tqdm
from timeit import default_timer as timer


def pad_data_for_cnn(raw_data_arr, opt):
    data_lengths = [len(d) for d in raw_data_arr]
    if opt.max_len != None:
        new_data = []
        new_data_lengths = []

        for data_len, data in zip(data_lengths, raw_data_arr):
            if data_len < opt.max_len:
                new_data.append(data)
                new_data_lengths.append(data_len)

        data_lengths = new_data_lengths
        raw_data_arr = new_data

    pad = opt.filter_shape - 1
    padded_data = []
    for data in raw_data_arr:
        temp = []
        for i in range(pad):
            temp.append(0)
        temp = temp + data
        while len(temp) < opt.max_len + 2 * pad:
            temp.append(0)
        padded_data.append(temp)
    padded_data = np.array(padded_data, dtype='int32')
    return padded_data


def load_data(data_set_path):
    dataset_path = os.path.join(data_set_path, 'yelp.p')
    dataset = cPickle.load(open(dataset_path, 'rb'))
    train_x, val_x, test_x = dataset[0], dataset[1], dataset[2]
    train_y, val_y, test_y = dataset[3], dataset[4], dataset[5]
    word_to_ix, ix_to_word = dataset[6], dataset[7]

    train_y = np.array(train_y, dtype='float32')
    val_y = np.array(val_y, dtype='float32')
    test_y = np.array(test_y, dtype='float32')

    return train_x, val_x, test_x, train_y, val_y, test_y, word_to_ix, ix_to_word


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def create_glove_embeddings(word_index, total_words, embed_size, max_seq_len, padded_data, glove_path, save_path):
    print(f'Loading Glove Word Embeddings from {glove_path}')
    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in tqdm(f, total=get_num_lines(glove_path)):
            values = line.split()
            # Insert word, value as key & value pair respectively in embeddigns_index
            word = values[0]
            co_efficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = co_efficients
    print(f'Retrieved {len(embeddings_index)} from Glove word embeddings {glove_path}')

    print('Creating word embedding matrix \n')
    embedding_matrix = np.zeros((total_words, embed_size))
    for word, i in word_index.items():
        if i > total_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # print('Creating word embeddings representation for sequence')
    # sent_emb_start = timer()
    # temp = np.zeros((padded_data.shape[0], max_seq_len, embed_size))
    # for i in tqdm(range(padded_data.shape[0])):
    #     for j in range(max_seq_len):
    #         temp[i][j] = embedding_matrix[padded_data[i][j]]
    # sent_emb_end = timer()
    # print(f'Final shape of word embeddings is {temp.shape} and took around {sent_emb_end - sent_emb_start: .2f}')
    # print('Dumping to a pickle file \n')
    # # cPickle.dump(temp, open(save_path, "wb"))
    # print('Completed !!!')
    # return temp
    return embedding_matrix


def word_embedding_lookup(x, max_seq_length, num_words, embedding_dim, weights):
    input_layer = Input(shape=(max_seq_length, ), name='text_input')
    embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        input_length=max_seq_length,
        weights=[weights], trainable=False,
        name="word_embedding"
    )
    embedding_text = embedding_layer(input_layer)
    embedding_model = Model(inputs=input_layer, outputs=embedding_text)
    sequence_embedding = embedding_model.predict(x)
    return sequence_embedding

