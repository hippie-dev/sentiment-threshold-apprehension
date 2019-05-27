import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from nltk import TweetTokenizer
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split

from k_utils import pad_data_for_cnn, create_glove_embeddings
from seq_generator import My_Generator
from timeit import default_timer as timer


class TransferModel:
    def __init__(self, dcnn_weights_path, label_op_loss, dense_count, opt, reconstruction_loss=None):
        self.dcnn_weights_path = dcnn_weights_path
        self.reconstruction_loss_function = reconstruction_loss
        self.label_op_loss_function = label_op_loss
        self.number_of_dense_layers_to_add = dense_count
        self.opt = opt

    def build_model(self):
        # f = h5py.File(self.dcnn_weights_path)
        m = load_model(self.dcnn_weights_path)
        print(m.summary())
        for idx, layer in enumerate(m.layers):
            print(f'Setting {layer.name} to trainable = True')
            layer.trainable = True
            # if idx >= len(m.layers) - 3:
            #     layer.trainable = True
            # else:
            #     layer.trainable = False

        print('Adding few more layers...')
        ll = m.layers[8].output
        ll = Dropout(0.3)(ll)
        for i in range(self.number_of_dense_layers_to_add):
            ll = Dense(300, name=f'label_dense_{i + 2}')(ll)

        transfer_train_predictions = Dense(1, activation="sigmoid", name='new_label_op')(ll)
        transfer_model = Model(inputs=m.input,
                               outputs=[m.get_layer('reconstruction_output').output, transfer_train_predictions])

        transfer_model.compile(optimizer=self.opt.optimizer,
                               loss={'reconstruction_output': 'mean_squared_error',
                                     'new_label_op': 'binary_crossentropy'},
                               loss_weights={'reconstruction_output': 0.4,
                                             'new_label_op': 1.},
                               metrics={'new_label_op': 'accuracy'})
        print(transfer_model.summary())
        tf.keras.utils.plot_model(
            transfer_model,
            to_file='transfer_model.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB'
        )

        return transfer_model

    def prepare_data(self):
        print('Annotated data pre processing')
        og_data = pd.read_csv(self.opt.dataset_path)
        print(f'Total Count of data is {og_data.count()}')
        print(f'Columns in the original dataset are {og_data.columns}')

        """
        Processing text data
        """
        og_data.drop(columns=['id', 'annotation'], axis=1, inplace=True)
        og_data['attack_type'] = og_data['type'].apply(lambda x: eval(x)[0].lower())
        og_data.drop(columns=['type'], axis=1, inplace=True)

        print(f"Label Value count is {og_data['attack_type'].value_counts()}")

        print('Processing text ...')
        og_data["text"] = og_data['text'].str.replace("\n", " ")
        og_data["text"] = og_data["text"].map(lambda x: self.tweet_cleaner(x))
        print('Text processing completed \n')

        """
        Processing the labels
        """
        attack_map = {
            'vulnerability': 1,
            'ransomware': 1,
            'ddos': 1,
            'leak': 1,
            '0day': 1,
            'botnet': 1,
            'all': 1,
            'general': 0
        }

        og_data = og_data.replace({"attack_type": attack_map})
        print(f'Final label value counts are {og_data["attack_type"].value_counts()}')

        X = og_data['text']
        y = og_data['attack_type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.opt.total_transfer_train_x = len(y_train)
        self.opt.n_transfer_validation = len(y_test)

        """
        Recording all the words to sequences
        """
        print('Converting sentence to sequences and padding the data')
        convert_to_seq_start = timer()
        transfer_word_to_ix = {
            'END': 0,
            'UNK': 1,
        }
        transfer_ix_to_word = {
            0: 'END',
            1: 'UNK',
        }

        count = 2

        for sent in X_train:
            words = sent.split()
            for word in words:
                if word.strip() in transfer_word_to_ix:
                    continue
                else:
                    transfer_word_to_ix[word] = count
                    transfer_ix_to_word[count] = word
                    count += 1

        self.opt.transfer_n_words = len(transfer_word_to_ix)
        """
        Converting sentences to sequences for train data
        """
        x_transfer_train = []
        x_transfer_test = []
        for sent in X_train:
            encoded_words = []
            words = sent.split()
            for word in words:
                if word in transfer_word_to_ix:
                    encoded_words.append(transfer_word_to_ix[word])
                else:
                    encoded_words.append(transfer_word_to_ix['UNK'])
            x_transfer_train.append(encoded_words)

        """
        Converting sentences to sequences for test data
        """
        for sent in X_test:
            words = sent.split()
            encoded_words = []
            for word in words:
                if word in transfer_word_to_ix:
                    encoded_words.append(transfer_word_to_ix[word])
                else:
                    encoded_words.append(transfer_word_to_ix['UNK'])
            x_transfer_test.append(encoded_words)

        """
        Padding the data
        """
        x_transfer_train_encoded = pad_data_for_cnn(x_transfer_train, self.opt)
        x_transfer_test_encoded = pad_data_for_cnn(x_transfer_test, self.opt)

        print(f'Sequence conversion and padding completed in {timer() - convert_to_seq_start: .2f}')

        """
        Create word embedding and sequence generator
        """
        transfer_embedding_matrix = create_glove_embeddings(word_index=transfer_word_to_ix,
                                                            total_words=self.opt.transfer_n_words,
                                                            embed_size=self.opt.embed_dim,
                                                            max_seq_len=self.opt.sent_len,
                                                            padded_data=x_transfer_train_encoded,
                                                            glove_path=self.opt.glove_path,
                                                            save_path=self.opt.embedding_save_path)

        my_training_batch_generator = My_Generator(x=x_transfer_train_encoded, y=np.array(list(y_train)), batch_size=32,
                                                   word_emb_matrix=transfer_embedding_matrix,
                                                   max_seq_len=self.opt.sent_len, embed_size=self.opt.embed_dim)

        my_validation_batch_generator = My_Generator(x=x_transfer_test_encoded, y=np.array(list(y_test)), batch_size=32,
                                                     word_emb_matrix=transfer_embedding_matrix,
                                                     max_seq_len=self.opt.sent_len, embed_size=self.opt.embed_dim)

        return my_training_batch_generator, my_validation_batch_generator, \
               x_transfer_train_encoded, x_transfer_test_encoded, y_train, np.array(list(y_test)), transfer_embedding_matrix, X

    @staticmethod
    def tweet_cleaner(raw_data):
        soup = BeautifulSoup(raw_data, 'html.parser')
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
        #     tokenized_arr = [x.strip() for x in tw.tokenize(lower_case)]
        tokenized_sent = " ".join([item.strip() for item in tw.tokenize(lower_case)])
        # print(" ".join([item.strip() for item in tw.tokenize(lower_case)]))
        return tokenized_sent


