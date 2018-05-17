from keras import backend as K
from keras.callbacks import Callback

class ClipLR(Callback):
    def __init__(self, new_lr, verbose=0):
        self.verbose = verbose
        self.new_lr = new_lr
        super(ClipLR, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        old_lr = K.get_value(self.model.optimizer.lr)
        if old_lr < 1e-7:
            K.set_value(self.model.optimizer.lr, self.new_lr)
            if self.verbose > 0:
                print('\nEpoch %05d: ClipLR clipping learning '
                        'rate to %s.' % (epoch + 1, self.new_lr))
