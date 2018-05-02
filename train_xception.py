import gc
import numpy as np
import os
import pandas as pd

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalMaxPooling2D
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import shuffle

from data import FurnituresDatasetWithAugmentation, FurnituresDatasetNoAugmentation
from keras_EMA import ExponentialMovingAverage
from utils import build_xception, get_image_paths_and_labels, MultiGPUModel


x_from_train_images, y_from_train_images = get_image_paths_and_labels(
    data_dir='data/train/')
x_from_val_images, y_from_val_images = get_image_paths_and_labels(
    data_dir='data/validation/')

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=4)
for val_index, minival_index in sss.split(
        x_from_val_images, y_from_val_images):
    x_from_minival_images, y_from_minival_images = x_from_val_images[
        minival_index], y_from_val_images[minival_index]
    x_from_val_images, y_from_val_images = x_from_val_images[
        val_index], y_from_val_images[val_index]

input_shape = (299, 299)
batch_size = 16
num_workers = 8
n_splits = 3
n_repeats = 1
num_gpus = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=4)

fold = 0
for train_index, test_index in rskf.split(
        x_from_train_images, y_from_train_images):
    fold += 1
    if fold == 1:
        pass
    else:
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
        save_best_trainval = ExponentialMovingAverage(filepath=trainval_filepath,
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

        print("\nFine-tune block 13 and block 14's layers")
        K.clear_session()
        model = load_model(trainval_filepath)
        for i in range(1, 19):
            model.layers[-i].trainable = True
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        print('Trainable params: {:,}'.format(trainable_count))
        model.compile(optimizer=Adam(lr=K.get_value(model.optimizer.lr) * 0.5, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['acc'])
        model.fit_generator(generator=train_generator,
                            epochs=5,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            workers=num_workers)

        K.clear_session()
        model = load_model(trainval_filepath)
        model.fit_generator(generator=train_generator,
                            epochs=5,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            workers=num_workers)

        K.clear_session()
        model = load_model(trainval_filepath)
        model.fit_generator(generator=train_generator,
                            epochs=5,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            workers=num_workers)

        print('\nFine-tune on the validation set')
        K.clear_session()
        print(
            'Found {} images belonging to {} classes'.format(
                len(x_from_val_images),
                128))
        print(
            'Found {} images belonging to {} classes'.format(
                len(x_from_minival_images),
                128))
        val_generator = FurnituresDatasetWithAugmentation(
            x_from_val_images, y_from_val_images, batch_size=batch_size, input_shape=input_shape)
        minival_generator = FurnituresDatasetNoAugmentation(
            x_from_minival_images, y_from_minival_images,
            batch_size=batch_size, input_shape=input_shape)
        valminival_filepath = 'checkpoint/xception/valminival.fold{}.best.hdf5'.format(
            fold)
        save_best_valminival = ExponentialMovingAverage(filepath=valminival_filepath,
                                                        verbose=1,
                                                        monitor='val_acc',
                                                        save_best_only=True,
                                                        mode='max')
        callbacks = [save_best_valminival]
        model = load_model(trainval_filepath)
        model.fit_generator(generator=val_generator,
                            epochs=5,
                            callbacks=callbacks,
                            validation_data=minival_generator,
                            workers=num_workers)

        K.clear_session()
        model = load_model(trainval_filepath)
        model.fit_generator(generator=val_generator,
                            epochs=5,
                            callbacks=callbacks,
                            validation_data=minival_generator,
                            workers=num_workers)
        
        K.clear_session()