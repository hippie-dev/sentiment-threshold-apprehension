import os
import numpy as np
from math import floor
from tensorflow.python.keras import backend as K


class kerasOptions(object):
    """

    """
    def __init__(self):
        # Global config
        self.n_words = None
        self.total_train_x = None
        self.total_validation_x = None
        self.embed_dim = 100  # Either Glove size or embed size that you want ot have. Typically 100 / 300 / 500
        self.filter_sizes = [300, 300, 500]
        self.strides = [2, 2, 2]
        self.filter_shape = 5
        self.max_len = 305
        # self.max_len = 305
        self.sent_len = self.max_len + 2 * (self.filter_shape - 1)
        self.sent_len_2 = np.int32(floor((self.sent_len - self.filter_shape) / self.strides[0]) + 1)
        self.sent_len_3 = np.int32(floor((self.sent_len_2 - self.filter_shape) / self.strides[1]) + 1)
        self.sent_len_4 = np.int32(floor((self.sent_len_3 - self.filter_shape) / self.strides[2]) + 1)

        self.dropout_ratio = 0.5
        self.batch_size = 32
        self.padding = 'valid'
        self.activation = 'relu'

        self.glove_path = os.path.join('dataset', 'glove.6B.100d.txt')
        self.embedding_save_path = 'word_emb.pkl'
        self.tbpath = 'graph'

        # Pure CNN config
        self.use_cnn_model = False
        self.cnn_filter_shapes = [3, 4, 5]
        self.cnn_epoch = 1
        self.cnn_model_save_name = os.path.join('models', 'baseline_cnn_10.h5')
        self.restore_cnn_model = True

        # DCNN
        self.train_dcnn_model = True
        self.test_dcnn_model = True
        self.use_cnn_emb_weights = False
        self.dcnn_op_dim = 1
        self.optimizer = 'rmsprop'
        self.reconstruction_loss_weight = K.variable(0.8)
        self.label_loss_weight = K.variable(1.0)
        self.dcnn_epoch = 1
        self.dcnn_model_save_name = os.path.join('models', f'dcnn_model_{self.dcnn_epoch}.h5')

        # Transfer
        self.train_transfer_model = True
        self.dataset_path = os.path.join('dataset', 'tweets_annotated.csv')
        self.total_transfer_train_x = None
        self.transfer_n_words = None
        self.transfer_learn_epochs = 3
        self.transfer_reconstruction_loss_weight = K.variable(1.0)
        self.transfer_label_loss_weight = K.variable(0.2)
        self.n_transfer_validation = 0
        self.transfer_model_save_name = os.path.join('models', f'transfer_model_{self.transfer_learn_epochs}.h5')

        #Real data
        self.test_real_data = False
        self.real_dataset_path = os.path.join('dataset', 'ancestry.csv')
        self.cleaned_tweet_save_path = os.path.join('dataset', 'cleaned.csv')

        # Baseline
        self.train_baseline = False
        self.baseline_embed_dim = 100
        self.baseline_sent_len = 313
        self.baseline_drop_out_ratio = 0.4
        self.baseline_epochs = 10
        self.baseline_batchsize = 32
        self.baseline_cnn_save_name = os.path.join('models', f'baseline_cnn_{self.baseline_epochs}.h5')

