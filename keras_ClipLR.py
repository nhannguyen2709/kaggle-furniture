from keras import backend as K
from keras.callbacks import Callback

class ClipLR(Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose
        super(ClipLR, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        old_lr = K.get_value(self.model.optimizer.lr)
        if old_lr < 1e-7:
            new_lr = 0.00003
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                print('\nEpoch %05d: ClipLR clipping learning '
                        'rate to %s.' % (epoch + 1, new_lr))
