from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import clone_model
import numpy as np

import config_cv_optimizers


def generate_optimizers() -> dict:
    return {
        'Adam_1': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.9,
            beta_2=.999
        ),
        'Adam_2': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.8,
            beta_2=.999
        ),
        'Adam_3': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.7,
            beta_2=.999
        ),
        'Adam_4': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.9,
            beta_2=.9
        ),
        'Adam_5': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.8,
            beta_2=.9
        ),
        'Adam_6': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.7,
            beta_2=.9
        ),
        'Adam_7': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.9,
            beta_2=.8
        ),
        'Adam_8': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.8,
            beta_2=.8
        ),
        'Adam_9': Adam(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            beta_1=.7,
            beta_2=.8
        ),
        'SGD_1': SGD(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            momentum=0,
            nesterov=False
        ),
        'SGD_2': SGD(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            momentum=0.2,
            nesterov=False
        ),
        'SGD_3': SGD(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            momentum=0.4,
            nesterov=False
        ),
        'SGD_4': SGD(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            momentum=0.6,
            nesterov=False
        ),
        'SGD_5': SGD(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            momentum=0,
            nesterov=True
        ),
        'SGD_6': SGD(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            momentum=0.2,
            nesterov=True
        ),
        'SGD_7': SGD(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            momentum=0.4,
            nesterov=True
        ),
        'SGD_8': SGD(
            learning_rate=config_cv_optimizers.LR_SCAN_START,
            momentum=0.6,
            nesterov=True
        ),
    }


class ReinitializeWeights(Callback):
    def __init__(
        self,
        epochs_to_wait:int = 10
    ):
        super(ReinitializeWeights, self).__init__()
        self.epochs_to_wait = epochs_to_wait
        self.starting_loss = np.Inf
        self.wait = 0

    def on_epoch_end(
        self,
        epoch: int,
        logs=None
    ):
        current_loss = logs.get('loss')
        if epoch == 0:
            self.starting_loss = current_loss
            print(f'\nloss limit set to {self.starting_loss * .95:.2e}')
        if self.wait == self.epochs_to_wait and np.less(self.starting_loss * .95, current_loss):
            self.model.set_weights(
                clone_model(self.model).get_weights()
            )
            print(f"\nReinitialized weights from random. {'#' * 30}")
            self.wait = -1
        self.wait += 1
