import gc
import numpy as np
import os
import pandas as pd

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import shuffle

from data import FurnituresDatasetWithAugmentation, FurnituresDatasetNoAugmentation, get_image_paths_and_labels
from model_utils import build_xception


x_from_train_images, y_from_train_images = get_image_paths_and_labels(
    data_dir='data/train/')

input_shape = (299, 299)
batch_size = 32
num_workers = 4
n_splits = 5
n_repeats = 1
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=4)

fold = 0
for train_index, test_index in rskf.split(
        x_from_train_images, y_from_train_images):
    fold += 1
    
    x_train, x_valid = x_from_train_images[train_index], x_from_train_images[test_index]
    y_train, y_valid = y_from_train_images[train_index], y_from_train_images[test_index]
    print('\nFold {}'.format(fold))
    print('Found {} images belonging to {} classes'.format(len(x_train), 128))
    print('Found {} images belonging to {} classes'.format(len(x_valid), 128))
    train_generator = FurnituresDatasetWithAugmentation(
        x_train, y_train,
        batch_size=batch_size, input_shape=input_shape)
    valid_generator = FurnituresDatasetNoAugmentation(
        x_valid, y_valid,
        batch_size=batch_size, input_shape=input_shape)

    trainval_filepath = 'checkpoint/xception/trainval.fold{}.best.hdf5'.format(
        fold)
    save_best_trainval = ModelCheckpoint(filepath=trainval_filepath,
                                         verbose=1,
                                         monitor='val_acc',
                                         save_best_only=True,
                                         mode='max')
    callbacks = [save_best_trainval]
    print('Train the last Dense layer')
    model = build_xception()
    model.compile(optimizer=Adam(lr=1e-3, decay=1e-5), loss='categorical_crossentropy',
                  metrics=['acc'])
    model.fit_generator(generator=train_generator,
                        epochs=5,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=num_workers)

    K.clear_session()

    print("\nFine-tune block 13 and block 14's layers")
    model = load_model(trainval_filepath)
    for i in range(1, 22):
        model.layers[-i].trainable = True
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    print('Trainable params: {:,}'.format(trainable_count))
    model.compile(optimizer=Adam(lr=K.get_value(model.optimizer.lr) * 0.5, decay=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.fit_generator(generator=train_generator,
                        epochs=25,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=num_workers)

    K.clear_session()
