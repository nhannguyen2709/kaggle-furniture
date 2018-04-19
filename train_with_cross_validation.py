import gc
import numpy as np
import os
import pandas as pd

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import shuffle

from data import FurnituresDataset
from utils import build_inception_resnet_v2, get_image_paths_and_labels
from keras_CLR import CyclicLR


x_from_train_images, y_from_train_images = get_image_paths_and_labels(
    data_dir='data/train/')
x_from_valid_images, y_from_valid_images = get_image_paths_and_labels(
    data_dir='data/validation/')
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True)
test_generator = test_datagen.flow_from_directory(
    'data/validation',
    batch_size=32,
    target_size=(299, 299),
    class_mode='categorical',
    shuffle=False)

# k-fold cross-validation on train images, evaluate on validation images
batch_size = 16
epochs = 1
num_workers = 10
fold = 0
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)

for train_index, test_index in rskf.split(
        x_from_train_images, y_from_train_images):
    fold += 1
    x_train, x_valid = x_from_train_images[train_index], x_from_train_images[test_index]
    y_train, y_valid = y_from_train_images[train_index], y_from_train_images[test_index]
    x_train, y_train = shuffle(x_train, y_train)
    print('Found {} images belonging to {} classes'.format(len(x_train), 128))
    print('Found {} images belonging to {} classes'.format(len(x_valid), 128))

    train_generator = FurnituresDataset(
        x_train, y_train, batch_size=batch_size)
    valid_generator = FurnituresDataset(
        x_valid, y_valid, batch_size=batch_size, shuffle=False)
    save_best = ModelCheckpoint(
        'checkpoint/inception_resnet_v2/fold{}.weights.best.hdf5'.format(fold),
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    clr_triangular = CyclicLR(
        mode='exp_range',
        max_lr=1e-3,
        step_size=len(x_train) //
        batch_size *
        2)
    callbacks = [clr_triangular, save_best]

    model = build_inception_resnet_v2()
    model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=num_workers)
    val_set_loss, val_set_acc = model.evaluate_generator(
        generator=test_generator, workers=num_workers)
    print('\nFold: {} -- val_loss: {} -- val_acc: {}'.format(fold,
                                                             val_set_loss, val_set_acc))
