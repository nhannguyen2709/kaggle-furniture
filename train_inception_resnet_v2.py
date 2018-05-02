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
from keras_CLR import CyclicLR
from utils import build_inception_resnet_v2, get_image_paths_and_labels, MultiGPUModel


x_from_train_images, y_from_train_images = get_image_paths_and_labels(
    data_dir='data/train/')
x_from_val_images, y_from_val_images = get_image_paths_and_labels(
    data_dir='data/validation/')

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=2)
for val_index, minival_index in sss.split(x_from_val_images, y_from_val_images):
    x_from_minival_images, y_from_minival_images = x_from_val_images[
        minival_index], y_from_val_images[minival_index]
    x_from_val_images, y_from_val_images = x_from_val_images[
        val_index], y_from_val_images[val_index]

# 5-fold cross-validation
input_shape = (299, 299)
batch_size = 8
epochs = 7
num_workers = 4
n_splits = 5
n_repeats = 1
num_gpus = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=2)

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

    trainval_lr_scheduler = CyclicLR(base_lr=1e-4, step_size=int(len(train_generator)*2), mode='exp_range')
    trainval_filepath = 'checkpoint/inception_resnet_v2/trainval.fold{}.best.hdf5'.format(
        fold)
    save_best_trainval = ExponentialMovingAverage(filepath=trainval_filepath,
        verbose=1,
        monitor='val_acc',
        save_best_only=True,
        mode='max')
    callbacks = [trainval_lr_scheduler, save_best_trainval]
    # # multi-gpu train
    # base_model = build_inception_resnet_v2()
    # # if os.path.exists(filepath):
    # #    base_model = load_model(filepath)
    # base_model.compile(optimizer=Adam(lr=1e-3),
    #                    loss='categorical_crossentropy',
    #                    metrics=['acc'])
    # parallel_model = MultiGPUModel(base_model, gpus=num_gpus)
    # parallel_model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'])
    # parallel_model.fit_generator(generator=train_generator,
    #                              epochs=epochs,
    #                              callbacks=callbacks,
    #                              validation_data=valid_generator,
    #                              workers=num_workers)   
    
    # del train_generator, valid_generator
    # gc.collect()
    
    # print('Found {} images belonging to {} classes'.format(len(x_from_val_images), 128))
    # print('Found {} images belonging to {} classes'.format(len(x_from_minival_images), 128))
    # val_generator = FurnituresDatasetWithAugmentation(
    #     x_from_val_images, y_from_val_images, batch_size=batch_size, input_shape=input_shape)
    # minival_generator = FurnituresDatasetNoAugmentation(
    #     x_from_minival_images, y_from_minival_images,
    #     batch_size=batch_size, input_shape=input_shape)
    # valminival_lr_scheduler = CyclicLR(base_lr=1e-4, max_lr=1e-3, step_size=int(len(val_generator)*2), mode='exp_range'
    # valminival_filepath = 'checkpoint/inception_resnet_v2/valminival.fold{}.best.hdf5'.format(
    #     fold)
    # save_best_valminival = ExponentialMovingAverage(filepath=valminival_filepath,
    #     verbose=1,
    #     monitor='val_acc',
    #     save_best_only=True,
    #     mode='max')
    # callbacks = [valminival_lr_scheduler, save_best_valminival]
    
    # base_model.compile(optimizer=Adam(lr=1e-4),
    #                    loss='categorical_crossentropy',
    #                    metrics=['acc'])
    # parallel_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['acc'])
    # parallel_model.fit_generator(generator=val_generator,
    #                              epochs=10,
    #                              callbacks=callbacks,
    #                              validation_data=minival_generator,
    #                              workers=num_workers)

    # single-gpu train
    if os.path.exists(trainval_filepath):
        model = load_model(trainval_filepath)
        for i in range(1, 31):
            model.layers[-i].trainable = True
        model.compile(optimizer=Adam(lr=K.get_value(model.optimizer.lr)*0.5),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        steps_per_epoch=100,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=num_workers)
    else:
        model = build_inception_resnet_v2()
        model.compile(optimizer=Adam(lr=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['acc'])            
        model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=num_workers)

    del train_generator, valid_generator
    gc.collect()
    print('Found {} images belonging to {} classes'.format(len(x_from_val_images), 128))
    print('Found {} images belonging to {} classes'.format(len(x_from_minival_images), 128))
    val_generator = FurnituresDatasetWithAugmentation(
        x_from_val_images, y_from_val_images, batch_size=batch_size, input_shape=input_shape)
    minival_generator = FurnituresDatasetNoAugmentation(
        x_from_minival_images, y_from_minival_images,
        batch_size=batch_size, input_shape=input_shape)
    valminival_lr_scheduler = CyclicLR(base_lr=1e-4, max_lr=1e-3, step_size=int(len(val_generator)*2), mode='exp_range')
    valminival_filepath = 'checkpoint/inception_resnet_v2/valminival.fold{}.best.hdf5'.format(
        fold)
    save_best_valminival = ExponentialMovingAverage(filepath=valminival_filepath,
        verbose=1,
        monitor='val_acc',
        save_best_only=True,
        mode='max')
    callbacks = [valminival_lr_scheduler, save_best_valminival]
    if os.path.exists(valminival_filepath):
        model = load_model(valminival_filepath)
    else:
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['acc'])
    model.fit_generator(generator=val_generator,
                        epochs=10,
                        callbacks=callbacks,
                        validation_data=minival_generator,
                        workers=num_workers)

    K.clear_session()
