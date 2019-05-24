from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
import logging


class WeightCallback(Callback):
    def __init__(self, reconstruction_loss_wt, label_op_loss_wt, total_epochs):
        self.alpha = reconstruction_loss_wt
        self.beta = label_op_loss_wt
        self.total_epochs = total_epochs

        self.max_wt = 1.0
        self.least_wt = 0.1

    def on_epoch_end(self, epoch, logs=None):
        if epoch != self.total_epochs - 1:
            print('\n-------------------------------CALL BACK RECEIVED ------------------------')
            print(f'\n---------------------------EPOCH IS {epoch} ------------------------------\n')

            step_increase = (self.max_wt - self.beta) / self.total_epochs
            step_decrease = (self.alpha - self.least_wt) / self.total_epochs

            step_increase = step_increase * (epoch + 1)
            step_decrease = step_decrease * (epoch + 1)

            new_alpha = K.get_value(self.alpha) - step_decrease
            new_alpha = K.variable(new_alpha)

            new_beta = K.get_value(self.beta) + step_increase
            new_beta = K.variable(new_beta)

            K.set_value(self.alpha, K.get_value(new_alpha))
            K.set_value(self.beta, K.get_value(new_beta))

            logging.info("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))

            print('Done setting')
