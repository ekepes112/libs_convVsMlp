from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import clone_model
import numpy as np

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
