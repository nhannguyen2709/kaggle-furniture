import numpy as np
import os
import pandas as pd

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.callbacks import ReduceLROnPlateau
from keras.backend import tensorflow_backend as K
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from keras_EMA import ExponentialMovingAverage
from data import FurnituresDatasetWithAugmentation, FurnituresDatasetNoAugmentation, get_image_paths_and_labels
from model_utils import build_xception, MultiGPUModel

input_shape = (448, 448)
batch_size = 8
num_workers = 4

x_from_train_images, y_from_train_images = get_image_paths_and_labels(
    data_dir='data/train/')
x_from_val_images, y_from_val_images = get_image_paths_and_labels(
    data_dir='data/validation/')
x_train, x_valid, y_train, y_valid = train_test_split(x_from_val_images, y_from_val_images, test_size=0.1)
x_train, y_train = np.concatenate((x_from_train_images, x_train)), np.concatenate((y_from_train_images, y_train))

print('Found {} images belonging to {} classes'.format(len(x_train), 128))
print('Found {} images belonging to {} classes'.format(len(x_valid), 128))
train_generator = FurnituresDatasetWithAugmentation(
    x_train, y_train,
    batch_size=batch_size, input_shape=input_shape)
valid_generator = FurnituresDatasetNoAugmentation(
    x_valid, y_valid,
    batch_size=batch_size, input_shape=input_shape)

filepath = 'checkpoint/xception/weights1.hdf5'
save_best = ExponentialMovingAverage(filepath=filepath,
                                        verbose=1,
                                        monitor='val_acc',
                                        save_best_only=True,
                                        mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_acc',
    factor=0.2,
    patience=2)
callbacks = [save_best, reduce_lr]

if os.path.exists(filepath):
    model = load_model(filepath)
else: 
    model = load_model('checkpoint/xception/trainval.fold1.best.hdf5')
    model.compile(optimizer=Adam(lr=1e-3, decay=1e-5), loss='categorical_crossentropy',
                  metrics=['acc'])
model.fit_generator(generator=train_generator,
                    epochs=20,
                    callbacks=callbacks,
                    validation_data=valid_generator,
                    workers=num_workers)
