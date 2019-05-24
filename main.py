import os

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model

from keras.CustomCallBacks import WeightCallback
from keras.cnn_dcnn_model import CNN_DCNN
from keras.cnn_model import CNN
from keras.k_utils import load_data, create_glove_embeddings, word_embedding_lookup
from keras.keras_config import kerasOptions

from tqdm import tqdm
import pandas as pd
from keras.k_utils import pad_data_for_cnn
from timeit import default_timer as timer

from keras.real_data import get_data, prepare_real_data
from keras.seq_generator import My_Generator
import h5py
from time import time
from keras.transfer_model import TransferModel


def train_cnn_model(emb_layer, x_train, y_train, x_val, y_val, opt):
    model = CNN(
        embedding_layer=emb_layer,
        num_words=opt.n_words,
        embedding_dim=opt.embed_dim,
        filter_sizes=opt.cnn_filter_shapes,
        feature_maps=opt.filter_sizes,
        max_seq_length=opt.sent_len,
        dropout_rate=opt.dropout_ratio,
        hidden_units=200,
        nb_classes=2
    ).build_model()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(),
        metrics=['accuracy']
    )

    #     y_train = y_train.reshape(-1, 1)
    #     model = build_model(emb_layer, opt)
    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(
        x_train, y_train,
        epochs=opt.cnn_epoch,
        batch_size=opt.batch_size,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping]
    )

    with open("CNN_train_history.txt", "w") as f:
        print(history.history, file=f)
    return model


def train_baseline_cnn(emb_layer, x_train, y_train, x_val, y_val, opt):
    model = CNN(
        embedding_layer=emb_layer,
        num_words=opt.transfer_n_words,
        embedding_dim=opt.baseline_embed_dim,
        filter_sizes=opt.cnn_filter_shapes,
        feature_maps=opt.filter_sizes,
        max_seq_length=opt.baseline_sent_len,
        dropout_rate=opt.baseline_drop_out_ratio,
        hidden_units=200,
        nb_classes=2
    ).build_model()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(),
        metrics=['accuracy']
    )

    #     y_train = y_train.reshape(-1, 1)
    #     model = build_model(emb_layer, opt)
    print(model.summary())
    tb_call_back = TensorBoard(log_dir=f'{opt.tbpath}/baseline_cnn_{time()}', histogram_freq=1,
                               write_graph=True, write_images=True)

    checkpoint = ModelCheckpoint("baseline_cnn.h5", monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(
        x_train, y_train,
        epochs=opt.baseline_epochs,
        batch_size=opt.baseline_batchsize,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, tb_call_back, checkpoint]
    )

    with open("CNN_train_baseline_history.txt", "w") as f:
        print(history.history, file=f)
    return model


def train_cnn_dcnn_model(train_generator, validation_generator, wnorm, opt):
    model = CNN_DCNN(max_sent_len=opt.sent_len,
                     embedding_dimension=opt.embed_dim,
                     filter_sizes=opt.filter_sizes,
                     filter_shapes=[opt.filter_shape, opt.filter_shape, opt.sent_len_3],
                     strides=opt.strides,
                     dcnn_op_dim=opt.dcnn_op_dim,
                     padding=opt.padding,
                     activation=opt.activation,
                     optimizer=opt.optimizer,
                     name='dcnn',
                     alpha=opt.reconstruction_loss_weight,
                     beta=opt.label_loss_weight,
                     n_words=wnorm.shape[0]).build_model()

    model.summary()
    tb_call_back = TensorBoard(log_dir=f'{opt.tbpath}/cnn_dcnn_{time()}', histogram_freq=1,
                               write_graph=True, write_images=True)

    checkpoint = ModelCheckpoint(os.path.join('models', f"dcnn_cp_{opt.dcnn_epoch}.h5"), monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    callbacks = [early_stopping,
                 checkpoint,
                 tb_call_back
                 # WeightCallback(opt.reconstruction_loss_weight, opt.label_loss_weight, opt.dcnn_epoch),
                 ]

    hist = model.fit_generator(generator=train_generator,
                               steps_per_epoch=(opt.total_train_x // opt.batch_size),
                               epochs=opt.dcnn_epoch,
                               verbose=1,
                               validation_data=validation_generator,
                               validation_steps=(opt.total_validation_x // opt.batch_size),
                               callbacks=callbacks,
                               use_multiprocessing=False,
                               workers=8,
                               max_queue_size=32)

    # hist = model.fit(x_train, [x_train, y_train], epochs=opt.dcnn_epoch,
    #                  batch_size=opt.batch_size, callbacks=[early_stopping])
    with open("CNN_DCNN_train_history.txt", "w") as f:
        print(hist.history, file=f)

    return model


def get_embedded_input(model, encoded_text):
    """
    Get embedding layer output from a CNN model as the input for CNN_DCNN model
    """
    embedding_layer_model = Model(inputs=model.input, outputs=model.get_layer('word_embedding').output)
    return embedding_layer_model.predict(encoded_text)


def model_predict(model, embedded_ip, raw_text, label_layer_name):
    total_items = embedded_ip.shape[0]
    label_model = Model(inputs=model.input, outputs=model.get_layer(label_layer_name).output)
    y_prob = label_model.predict(embedded_ip).reshape(-1)
    y_pred = y_prob > 0.5
    return y_prob, y_pred


def main():
    opt = kerasOptions()

    if opt.train_dcnn_model or opt.test_dcnn_model:
        print('Loading the dataset ...')
        dataset_load_start = timer()
        train_x, val_x, test_x, train_y, val_y, test_y, word_to_ix, ix_to_word = load_data('../dataset')

        opt.n_words = len(ix_to_word)
        dataset_load_end = timer()

        # train_x = train_x[:5000]
        # train_y = train_y[:5000]
        #
        # test_x = test_x[:5000]
        # test_y = test_y[:5000]
        #
        # val_x = val_x[:5000]
        # val_y = val_y[:5000]

        opt.total_train_x = len(train_y)
        opt.total_validation_x = len(val_y)

        print(f'Total train x is {opt.total_train_x}')
        print(f'Total Val x is {opt.total_validation_x}')
        print(f'Dataset loading completed in {dataset_load_end - dataset_load_start: .2f} \n')
        print(f'Total Number of words is {opt.n_words} \n')

        with tqdm(total=100, desc="Padding and Encoding data ", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            train_x_encoded = pad_data_for_cnn(train_x, opt)
            pbar.update(50)
            test_x_encoded = pad_data_for_cnn(test_x, opt)
            pbar.update(25)
            val_x_encoded = pad_data_for_cnn(val_x, opt)
            pbar.update(25)

    """
    Pure CNN model training
    """
    if opt.train_dcnn_model or opt.test_dcnn_model:
        if opt.use_cnn_model:
            if opt.restore_cnn_model:
                cnn_model = load_model(opt.cnn_model_save_name)
                print('Restoring CNN model')
            else:
                """
                For pure CNN in order to obtain word embeddings
                Label encoding is required since its a multi class classification model
                """
                le = LabelEncoder()
                y_c_train = le.fit_transform(train_y)
                y_c_train = tf.keras.utils.to_categorical(y_c_train, 2)

                y_c_val = le.fit_transform(val_y)
                y_c_val = tf.keras.utils.to_categorical(y_c_val, 2)

                cnn_start = timer()
                print(f'Running pure CNN model for extracting word embeddings for {opt.cnn_epoch} epochs.')
                cnn_model = train_cnn_model(None, train_x_encoded, y_c_train,
                                            val_x_encoded, y_c_val, opt)
                cnn_model.save(opt.cnn_model_save_name)
                cnn_end = timer()
                print(f'Pure CNN model execution completed in {cnn_end - cnn_start: .2f} '
                      f'and is saved in {opt.cnn_model_save_name}')

            my_training_batch_generator = My_Generator(x=train_x_encoded, y=train_y, batch_size=32,
                                                       word_emb_matrix=cnn_model,
                                                       max_seq_len=opt.sent_len, embed_size=opt.embed_dim,
                                                       use_cnn_model=True)
            my_validation_batch_generator = My_Generator(x=val_x_encoded, y=val_y, batch_size=32,
                                                         word_emb_matrix=cnn_model,
                                                         max_seq_len=opt.sent_len, embed_size=opt.embed_dim,
                                                         use_cnn_model=True)
        else:
            """
            Loading Glove word embeddings
            """
            print('Loading Glove embeddings \n')
            embedding_matrix = create_glove_embeddings(word_index=word_to_ix,
                                                       total_words=opt.n_words,
                                                       embed_size=opt.embed_dim,
                                                       max_seq_len=opt.sent_len,
                                                       padded_data=train_x_encoded,
                                                       glove_path=opt.glove_path,
                                                       save_path=opt.embedding_save_path)

            print(f'Emb matrix shape is {embedding_matrix.shape}')
            print(f'Train x encoded shape is {train_x_encoded.shape}')
            print(f'Val x encoded shape is {val_x_encoded.shape}')
            print(f'Sent len passed is {opt.sent_len}')
            my_training_batch_generator = My_Generator(x=train_x_encoded, y=train_y, batch_size=32,
                                                       word_emb_matrix=embedding_matrix,
                                                       max_seq_len=opt.sent_len, embed_size=opt.embed_dim)
            my_validation_batch_generator = My_Generator(x=val_x_encoded, y=val_y, batch_size=32,
                                                         word_emb_matrix=embedding_matrix,
                                                         max_seq_len=opt.sent_len, embed_size=opt.embed_dim)


    """
    DCNN Model training using the above word embeddings
    """

    if opt.train_dcnn_model:
        clear_session()
        print(f'Starting training of CNN - DCNN model for {opt.dcnn_epoch} epochs.')
        dcnn_start = timer()
        model_cnn_dcnn = train_cnn_dcnn_model(my_training_batch_generator, my_validation_batch_generator,
                                              embedding_matrix, opt)
        # model_cnn_dcnn = train_cnn_dcnn_model(embedded_train_input,
        # train_y[:5000], embedded_val_input, val_y[:5000], opt)
        model_cnn_dcnn.save(opt.dcnn_model_save_name)
        dcnn_end = timer()
        print(f'DCNN model execution completed in {dcnn_end - dcnn_start: .2f} and is saved in {opt.cnn_model_save_name}')

    """
    Testing the model
    """
    if opt.test_dcnn_model:
        clear_session()
        print('Testing the model \n')
        print('Getting word embeddings for test data ')
        embedded_test_ip_start_time = timer()
        if opt.use_cnn_emb_weights:
            model_cnn = load_model(opt.cnn_model_save_name)
            embedded_test_input = get_embedded_input(model_cnn, test_x_encoded)
        else:
            embedded_test_input = word_embedding_lookup(x=test_x_encoded,
                                                        max_seq_length=opt.sent_len,
                                                        num_words=opt.n_words,
                                                        embedding_dim=opt.embed_dim,
                                                        weights=embedding_matrix)
        embedded_test_input = embedded_test_input.reshape(embedded_test_input.shape[0], embedded_test_input.shape[1],
                                                          embedded_test_input.shape[2], 1)
        embedded_test_ip_end_time = timer()
        print(f'Embedded test input from model shape is {embedded_test_input.shape}')
        print(f'Word Embeddings Extraction completed in {embedded_test_ip_end_time - embedded_test_ip_start_time: .2f}')

        # model_cnn_dcnn = load_model(opt.dcnn_model_save_name,
        #                             custom_objects={'reconstruction_loss': CNN_DCNN.reconstruction_loss})
        model_cnn_dcnn = load_model(opt.dcnn_model_save_name)
        y_prob, y_pred = model_predict(model_cnn_dcnn, embedded_test_input, test_x_encoded, 'label_op')
        predictions = pd.DataFrame({'test_y': test_y, 'prob': y_prob, 'pred': y_pred})
        predictions = predictions[['test_y', 'prob', 'pred']]
        print('Predictions are:')
        print(predictions)

        total_correct = 0
        print('Calculating total accuracy: ')
        for idx, item in enumerate(y_pred):
            if item:
                if test_y[idx] == 1:
                    total_correct += 1
            else:
                if test_y[idx] == 0:
                    total_correct += 1

        print(f'Got total correct of {total_correct} out of {len(test_y)}')
        print(f'Total accuracy on test data is {total_correct * 100 / len(test_y)}')
        print('Training completed and can be taken to next stage for transfer learning !!!')

    if opt.train_baseline:
        clear_session()
        """
        This step is essential in order to load the data
        """

        dcnn_weights = opt.dcnn_model_save_name
        transfer_m_obj = TransferModel(dcnn_weights_path=dcnn_weights,
                                       label_op_loss=CNN_DCNN.label_op_loss,
                                       dense_count=2,
                                       opt=opt)
        transfer_training_batch, validation_batch_gen, x_transfer_train_encoded, x_transfer_test_encoded, y_train, y_test, transfer_embedding_matrix, og_data = transfer_m_obj.prepare_data()

        le = LabelEncoder()
        y_c_train = le.fit_transform(y_train)
        y_c_train = tf.keras.utils.to_categorical(y_c_train, 2)

        y_c_val = le.fit_transform(y_test)
        y_c_val = tf.keras.utils.to_categorical(y_c_val, 2)

        cnn_start = timer()
        print(f'Running pure CNN model for extracting word embeddings for {opt.cnn_epoch} epochs.')
        cnn_model = train_baseline_cnn(None, x_transfer_train_encoded, y_c_train,
                                    x_transfer_test_encoded, y_c_val, opt)
        cnn_model.save(opt.baseline_cnn_save_name)

    """
    Transfer Learn
    """
    if opt.train_transfer_model:
        clear_session()
        dcnn_weights = opt.dcnn_model_save_name

        transfer_m_obj = TransferModel(dcnn_weights_path=dcnn_weights,
                                       label_op_loss=CNN_DCNN.label_op_loss,
                                       dense_count=2,
                                       opt=opt)
        transfer_model = transfer_m_obj.build_model()
        transfer_training_batch, validation_batch_gen, x_transfer_train_encoded, x_transfer_test_encoded, y_train, y_test, transfer_embedding_matrix, og_data = transfer_m_obj.prepare_data()

        print('Training Transfer Learn Model')
        tb_call_back = TensorBoard(log_dir=f'{opt.tbpath}/transfer_{time()}', histogram_freq=1,
                                   write_graph=True, write_images=True)
        # tranfer_weight_callback = WeightCallback(opt.transfer_reconstruction_loss_weight,
        #                                          opt.transfer_label_loss_weight, opt.transfer_learn_epochs)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        callbacks = [tb_call_back]
        transfer_dcnn_start = timer()
        transfer_hist = transfer_model.fit_generator(generator=transfer_training_batch,
                                                     steps_per_epoch=(opt.total_transfer_train_x // opt.batch_size),
                                                     validation_data=validation_batch_gen,
                                                     validation_steps=(opt.n_transfer_validation // opt.batch_size),
                                                     epochs=opt.transfer_learn_epochs,
                                                     callbacks=callbacks,
                                                     verbose=1,
                                                     use_multiprocessing=False,
                                                     workers=8,
                                                     max_queue_size=32)
        transfer_model.save(opt.transfer_model_save_name)

        transfer_dcnn_end = timer()
        print(f'Transfer DCNN model execution completed in {transfer_dcnn_start - transfer_dcnn_end: .2f}')

        # transfer_model = load_model(opt.transfer_model_save_name)

        transfer_embedded_test_input = word_embedding_lookup(x=x_transfer_test_encoded,
                                                             max_seq_length=opt.sent_len,
                                                             num_words=opt.transfer_n_words,
                                                             embedding_dim=opt.embed_dim,
                                                             weights=transfer_embedding_matrix)

        transfer_embedded_test_input = transfer_embedded_test_input.reshape(transfer_embedded_test_input.shape[0],
                                                                            transfer_embedded_test_input.shape[1],
                                                                            transfer_embedded_test_input.shape[2], 1)
        y_prob, y_pred = model_predict(transfer_model, transfer_embedded_test_input, x_transfer_test_encoded,
                                       'new_label_op')
        predictions = pd.DataFrame({'test_y': y_test, 'prob': y_prob, 'pred': y_pred})
        predictions = predictions[['test_y', 'prob', 'pred']]
        print('Predictions are:')
        print(predictions)
        complete_real_predicted = pd.concat([og_data, predictions], axis=1)
        complete_real_predicted.to_csv('transfer_predicted', header=True)

        total_correct = 0
        incorrect_true = 0
        incorrect_false = 0
        print('Calculating total accuracy: ')
        for idx, item in enumerate(y_pred):
            if item:
                if list(y_test)[idx] == 1:
                    total_correct += 1
                else:
                    incorrect_true += 1
            else:
                if list(y_test)[idx] == 0:
                    total_correct += 1
                else:
                    incorrect_false += 1
        print(f'Got total correct of {total_correct} out of {len(y_test)}')

    if opt.test_real_data:
        real_data = get_data(opt)
        print(type(real_data))
        print(real_data.shape)
        transfer_model = load_model(opt.transfer_model_save_name)

        embedded_data_x, padded_data_x = prepare_real_data(real_data, opt)
        y_prob, y_pred = model_predict(transfer_model, embedded_data_x, padded_data_x, 'new_label_op')
        predictions = pd.DataFrame({'prob': y_prob, 'pred': y_pred})
        complete_real_predicted = pd.concat([real_data, predictions], axis=1)
        print(complete_real_predicted)
        complete_real_predicted.to_csv('real_predicted', header=True)
        print('Saved')


if __name__ == '__main__':
    main()
