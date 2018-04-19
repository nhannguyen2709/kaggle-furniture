import argparse
import cv2
import os
import numpy as np
from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import shuffle


class FurnituresDataset(Sequence):
    def __init__(
            self,
            x_set,
            y_set,
            batch_size,
            num_classes=128,
            shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = np.array([cv2.resize(cv2.imread(file_name), (299, 299))
                                for file_name in batch_x])
        augmented_data = self.datagen.flow(
            batch_imgs,
            to_categorical(
                np.array(batch_y),
                num_classes=self.num_classes),
            batch_size=self.batch_size).next()
        del batch_imgs
        return augmented_data


class FurnituresDatasetNoLabels(Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = np.array([cv2.resize(cv2.imread(file_name), (299, 299))
                               for file_name in batch_x])
        return batch_imgs / 255.
