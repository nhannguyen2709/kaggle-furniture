import argparse
import cv2
import gc
import os
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import shuffle


class FurnituresDatasetWithAugmentation(Sequence):
    def __init__(
            self,
            x_set,
            y_set,
            batch_size,
            input_shape,
            percent_cropped=0.1,
            num_classes=128,
            shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.percent_cropped = percent_cropped
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_train_begin()
        self.on_epoch_end()

    def on_train_begin(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)

    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_imgs = np.array([cv2.resize(cv2.imread(file_name), 
                                          self.input_shape, 
                                          interpolation=cv2.INTER_NEAREST)
                               for file_name in batch_x])
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True)
        augmented_data = datagen.flow(
            batch_imgs,
            to_categorical(
                np.array(batch_y),
                num_classes=self.num_classes),
            batch_size=self.batch_size).next()
        del batch_imgs, datagen
        gc.collect()

        return augmented_data


class FurnituresDatasetNoAugmentation(Sequence):
    def __init__(
            self,
            x_set,
            y_set,
            batch_size,
            input_shape,
            percent_cropped=0.1,
            num_classes=128):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.percent_cropped = percent_cropped
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_imgs = np.array([cv2.resize(cv2.imread(file_name), 
                                          self.input_shape, 
                                          interpolation=cv2.INTER_NEAREST)
                               for file_name in batch_x])

        return batch_imgs / 255., to_categorical(np.array(batch_y), num_classes=self.num_classes)


class FurnituresDatasetNoLabels(Sequence):
    def __init__(self, x_set, batch_size, input_shape, percent_cropped):
        self.x = x_set
        self.batch_size = batch_size
        self.percent_cropped = percent_cropped
        self.input_shape = input_shape

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_imgs = np.array([cv2.resize(cv2.imread(file_name), 
                                          self.input_shape, 
                                          interpolation=cv2.INTER_NEAREST)
                               for file_name in batch_x])

        return batch_imgs / 255.


# if __name__ == '__main__':
#     from utils import get_image_paths_and_labels
#     x_from_train_images, y_from_train_images = get_image_paths_and_labels(
#         data_dir='data/train/')
#     train_generator = FurnituresDatasetWithAugmentation(
#         x_from_train_images, y_from_train_images,
#         input_shape=(299, 299), batch_size=16)
#     for i in range(len(train_generator)):
#         x, y = train_generator.__getitem__(i)
#         print(x.shape, y.shape)
