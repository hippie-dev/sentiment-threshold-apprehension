from math import floor

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv1D, Lambda, Conv2DTranspose, Flatten, Dense, Dropout, Conv2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import softmax
import tensorflow as tf
import numpy as np


class CNN_DCNN:
    def __init__(self, max_sent_len=None, embedding_dimension=None,
                 filter_sizes=[300, 300, 500], filter_shapes=[5, 5, 5], strides=[2, 2, 2],
                 dcnn_op_dim=1, padding='valid',
                 activation='relu', optimizer='rmsprop', name='cnn_dcnn', alpha=K.variable(1.0), beta=K.variable(1.0),
                 n_words=0):
        self.max_sent_len = max_sent_len
        self.embedding_dimension = embedding_dimension
        self.filter_sizes = filter_sizes
        self.filter_shapes = filter_shapes
        self.strides = strides
        self.dcnn_op_dim = dcnn_op_dim
        self.padding = padding
        self.activation = activation
        self.optimizer = optimizer
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.n_words = n_words

        self.sent_len_2 = np.int32(floor((self.max_sent_len - self.filter_shapes[0]) / self.strides[0]) + 1)
        self.sent_len_3 = np.int32(floor((self.sent_len_2 - self.filter_shapes[1]) / self.strides[1]) + 1)

    def build_model(self):
        print('----------------------------------- Inside build model -----------------------------------')

        if len(self.filter_sizes) != len(self.filter_shapes):
            raise Exception("Please define filter shape and filter sizes of same length")

        # wnorm_input = Input(shape=(self.embedding_dimension, 1), dtype='float32', name=f'{self.name}_word_embedding')
        #
        # raw_batch_x = Input(shape=(313, 1), dtype='float32', name=f'{self.name}_raw_batch')

        # raw_batch_x_s = K.squeeze(raw_batch_x, axis=-1)
        #
        # wnorm_input_s = K.squeeze(wnorm_input, axis=-1)
        # seq_input_1 = Input(shape=(self.max_sent_len, self.embedding_dimension, 1),
        #                     dtype='float32', name=f'{self.name}_embedded_input_1')

        #
        # x_org_input = Input(shape=(self.max_sent_len),
        #                     dtype='float32', name=f'{self.name}_x_org')

        seq_input = Input(shape=(self.max_sent_len, self.embedding_dimension, 1),
                          dtype='float32', name=f'{self.name}_embedded_input')

        print(f'Seq input shape is {seq_input.shape}')

        cnn_1 = Conv2D(filters=self.filter_sizes[0], kernel_size=[self.filter_shapes[0], self.embedding_dimension], strides=[self.strides[0], 1],
                           padding=self.padding, activation=self.activation, name=f'{self.name}_h1_3')(seq_input)

        print(f'CNN_1 shape is {cnn_1.shape}')

        cnn_2 = Conv2D(filters=self.filter_sizes[1], kernel_size=[self.filter_shapes[1], 1], strides=[self.strides[1], 1],
                       padding=self.padding, activation=self.activation, name=f'{self.name}_h2_3')(cnn_1)

        print(f'CNN_2 shape is {cnn_2.shape}')
        print(f'Sent len 3 is {self.sent_len_3}')

        cnn_3 = Conv2D(filters=self.filter_sizes[2], kernel_size=[self.sent_len_3, 1], padding=self.padding,
                       activation=self.activation, name=f'{self.name}_h3_3')(cnn_2)

        print(f'CNN_3 shape is {cnn_3.shape}')

        H = Lambda(lambda w: K.squeeze(w, axis=2))(cnn_3)
        mid = Flatten()(H)
        mid = Dense(300, name='label_dense_1')(mid)
        label_op = Dense(1, name='label_op', activation='sigmoid')(mid)

        dcnn_3 = Conv2DTranspose(filters=self.filter_sizes[1], kernel_size=[self.sent_len_3, 1],
                                 padding=self.padding, activation=self.activation, name=f'{self.name}_h2_t_3')(cnn_3)

        dcnn_2 = Conv2DTranspose(filters=self.filter_sizes[0], kernel_size=[self.filter_shapes[1], 1], strides=[self.strides[1], 1],
                                 padding=self.padding, activation=self.activation, name=f'{self.name}_h2_t_2')(dcnn_3)

        reconstruction_output = Conv2DTranspose(filters=1, kernel_size=[self.filter_shapes[0], self.embedding_dimension],
                                                strides=[self.strides[0], 1],
                                 padding=self.padding, activation=self.activation, name='reconstruction_output')(dcnn_2)

        print(f'Reconstruction op shape is {reconstruction_output.shape}')

        model = Model(inputs=seq_input, outputs=[reconstruction_output, label_op])
        model.compile(optimizer=self.optimizer,
                      loss={'reconstruction_output': 'mse', 'label_op': 'binary_crossentropy'},
                      loss_weights={'reconstruction_output': 0.4, 'label_op': 1.},
                      metrics={'label_op': 'accuracy'})

        return model

    @staticmethod
    def reconstruction_loss(loss_weight, wnorm, raw_input, n_words):
        def loss(y_true, y_pred):
            y_true = K.squeeze(y_true, axis=-1)
            # Squeeze y_pred; i.e remove last layer which is of dim 1
            y_pred_squeezed = K.squeeze(y_pred, axis=-1)

            """
            Reconstructing x_org
            """
            # reversed_wnorm = Lambda(lambda x: )
            # reversed_wnorm = dict(map(reversed, wnorm.items()))
            # x_org = Lambda(lambda x: [tf.reshape(tf.where(tf.equal(wnorm, word)), [-1])[0] for sent in x for word in sent])(y_true)
            # x_org =
            # x_org = [reversed_wnorm.get(word) for sent in y_true for word in sent]
            x_org = raw_input
            print(raw_input.shape)
            x_temp = Lambda(lambda x: tf.cast(tf.reshape(x, [-1, ]), dtype=tf.int32))(x_org)

            K.print_tensor(K.shape(y_pred_squeezed), message='y_pred_squeezed are ')
            print(f'Inside decoder....After reshape of x_norm is {y_pred_squeezed.shape}')

            # Calc prob logits
            print(type(y_pred_squeezed))
            print(type(wnorm))
            print(f'wnorm shape is {wnorm.shape}')
            prob_logits = K.batch_dot(y_pred_squeezed, wnorm, axes=[2, 1])
            prob = Lambda(lambda x: tf.nn.log_softmax(x * 100, axis=-1, name='prob_lambda'))(prob_logits)
            print(f'Prob shape is {prob.shape}')
            prob = Lambda(lambda x: tf.reshape(x, [-1, n_words]))(prob)
            # prob = K.reshape(prob, [-1, wnorm.shape[0]])
            print(f'Prob reshaped is {prob.shape}')

            """
            Get prob of all the words
            """
            idx = Lambda(lambda x: tf.range(K.shape(x)[0], K.shape(x)[1]))(y_pred_squeezed)
            all_idx = K.transpose(K.stack([idx, x_temp]))
            all_prob = Lambda(lambda prob_idx_list: tf.gather_nd(prob_idx_list[0], prob_idx_list[1]))([prob, all_idx])

            K.print_tensor(K.shape(all_prob), message='all_prob shape is: ')
            recons_loss = Lambda(lambda x: -tf.reduce_mean(x))(all_prob)

            # K.print_tensor(loss, message='Loss is: ')
            # weighted_recons_loss = loss_weight * recons_loss

            return recons_loss
        return loss

    # def build_model_old(self):
    #     print('----------------------------------- Inside build model -----------------------------------')
    #     if len(self.filter_sizes) != len(self.filter_shapes):
    #         raise Exception("Please define filter shape and filter sizes of same length")
    #
    #     seq_input = Input(shape=(self.max_sent_len, self.embedding_dimension),
    #                       dtype='float32', name=f'{self.name}_embedded_input')
    #
    #     cnn_1 = self.add_cnn_layer(seq_input, self.filter_sizes[0], [self.filter_shapes[0]],
    #                                [self.strides[0]], padding=self.padding,
    #                                activation=self.activation, name=f'{self.name}_h1_3')
    #     cnn_2 = self.add_cnn_layer(cnn_1, self.filter_sizes[1], [self.filter_shapes[1]],
    #                                [self.strides[1]], padding=self.padding,
    #                                activation=self.activation, name=f'{self.name}_h2_3')
    #     cnn_3 = self.add_cnn_layer(cnn_2, self.filter_sizes[2], [self.filter_shapes[2]],
    #                                [self.strides[2]], padding=self.padding,
    #                                activation=self.activation, name=f'{self.name}_h3_3')
    #
    #     mid = Flatten()(cnn_3)
    #     mid = Dense(300, name='label_dense_1')(mid)
    #     label_op = Dense(1, name='label_op', activation='sigmoid')(mid)
    #
    #     dcnn_3 = self.add_dcnn_layer(cnn_3, self.filter_sizes[1], kernel_size=[self.filter_shapes[2], 1],
    #                                  stride=[self.strides[2], 1], name=f'{self.name}_h2_t_3')
    #     dcnn_2 = self.add_dcnn_layer(dcnn_3, self.filter_sizes[0], kernel_size=[self.filter_shapes[1], 1],
    #                                  stride=[self.strides[1], 1], name=f'{self.name}_h2_t_2')
    #     reconstruction_output = self.add_dcnn_layer(dcnn_2, self.dcnn_op_dim,
    #                                                 kernel_size=[self.filter_shapes[0], 1],
    #                                                 stride=[self.strides[0], 1], name='reconstruction_output')
    #
    #     model = Model(inputs=seq_input, outputs=[reconstruction_output, label_op])
    #     model.compile(optimizer=self.optimizer,
    #                   # loss={'reconstruction_output': self.reconstruction_loss,
    #                   #       'label_op': 'binary_crossentropy'},
    #                   loss={'label_op': 'binary_crossentropy'},
    #                   # loss_weights={'reconstruction_output': 0.8, 'label_op': 1.},
    #                   metrics={'label_op': 'accuracy'})
    #     return model

    def add_cnn_layer(self, x_input, filter_size, kernel_size, stride, padding='valid', activation='relu', name=None):
        cnn_layer = Conv1D(filters=filter_size, kernel_size=kernel_size, strides=stride,
                           padding=padding, activation=activation, name=name)(x_input)
        return cnn_layer

    def add_dcnn_layer(self, x_input, filter_size, kernel_size, stride, padding='valid',
                       activation='relu', name=None):
        x_2d = Lambda(lambda x: K.expand_dims(x, axis=2))(x_input)
        x_2d_t = Conv2DTranspose(filters=filter_size, kernel_size=kernel_size, strides=stride,
                                 padding=padding, activation=activation)(x_2d)
        x_s = Lambda(lambda w: K.squeeze(w, axis=2), name=name)(x_2d_t)

        return x_s

    # @staticmethod
    # def reconstruction_loss(y_true, y_pred):
    #     y_true = K.l2_normalize(y_true, axis=-1)
    #     y_pred = K.l2_normalize(y_pred, axis=-1)
    #     loss = K.mean(1 - K.sum((y_true * y_pred), axis=-1))
    #
    #     return loss

    # @staticmethod
    # def reconstruction_loss(loss_weight):
    #     def loss(y_true, y_pred):
    #         y_true = K.l2_normalize(y_true, axis=-1)
    #         y_pred = K.l2_normalize(y_pred, axis=-1)
    #
    #         l = K.mean(1 - K.sum((y_true * y_pred), axis=-1))
    #         return l * loss_weight
    #
    #     return loss

    # @staticmethod
    # def reconstruction_loss(loss_weight, w_norm):
    #     def loss(y_true, y_pred):
    #         y_true = K.l2_normalize(y_true, axis=-1)
    #         y_pred = K.l2_normalize(y_pred, axis=-1)
    #
    #         y_true = tf.keras.layers.Lambda((lambda x: tf.Print(x, [x], message='Y_true = ', first_n=-1, summarize=10000)))(y_true)
    #         y_pred = tf.keras.layers.Lambda((lambda x: tf.Print(x, [x], message='Y_pred = ', first_n=-1, summarize=10000)))(y_pred)
    #
    #         y_true_shape = K.print_tensor(K.shape(y_true), message='y true shape is ')
    #         y_pred_shape = K.print_tensor(K.shape(y_pred), message='y true shape is ')
    #
    #         y_pred_squeezed = K.print_tensor(Lambda(lambda x: K.squeeze(x, axis=2))(y_pred), message='y_pred_squeezed = ')
    #
    #         K.print_tensor(y_pred_squeezed, message='y_pred_squeezed are ')
    #         K.print_tensor(K.shape(y_pred_squeezed), message='y_pred_squeezed shape is  ')
    #
    #         # prob_logits = K.batch_dot(y_pred_squeezed, w_norm, [[2], [1]])
    #         # prob = softmax(prob_logits, axis=-1)
    #
    #         l = K.mean(1 - K.sum((y_true * y_pred), axis=-1))
    #         print(f'Loss is {l}')
    #         return l * loss_weight
    #
    #     return loss

    @staticmethod
    def label_op_loss(loss_weight):
        def loss(y_true, y_pred):
            return K.mean(tf.nn.weighted_cross_entropy_with_logits(
                y_true,
                y_pred,
                loss_weight,
                name=None
            ), axis=-1)
            # l = K.binary_crossentropy(y_true, y_pred)
        return loss
