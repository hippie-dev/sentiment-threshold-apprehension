import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.utils import Sequence


class My_Generator(Sequence):
    def __init__(self, x, y, batch_size, word_emb_matrix, max_seq_len, embed_size, use_cnn_model=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.word_emb_matrix = word_emb_matrix
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size
        self.use_cnn_model = use_cnn_model

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        temp = np.zeros((batch_x.shape[0], self.max_seq_len, self.embed_size))
        if not self.use_cnn_model:
            for i in range(batch_x.shape[0]):
                for j in range(self.max_seq_len):
                    temp[i][j] = self.word_emb_matrix[batch_x[i][j]]
        else:
            temp = self.get_embedded_input(self.word_emb_matrix, batch_x)


        # batch_x = temp
        batch_x_emb = temp.reshape(batch_x.shape[0], self.max_seq_len, self.embed_size, 1)

        # word_emb_expanded = self.word_emb_matrix.reshape(self.word_emb_matrix.shape[0], self.word_emb_matrix.shape[1], 1)

        return batch_x_emb, [batch_x_emb, batch_y]

    @staticmethod
    def get_embedded_input(model, encoded_text):
        """
        Get embedding layer output from a CNN model as the input for CNN_DCNN model
        """
        embedding_layer_model = Model(inputs=model.input, outputs=model.get_layer('word_embedding').output)
        return embedding_layer_model.predict(encoded_text)
