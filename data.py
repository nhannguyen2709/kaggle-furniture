import argparse
import cv2
import gc
import os
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import Sequence, to_categorical
from keras.applications.imagenet_utils import _preprocess_numpy_input
from keras.preprocessing.image import ImageDataGenerator

import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.utils import shuffle


def get_image_paths_and_labels(data_dir):
    x = []
    y = []
    for folder in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, folder)
        for image_filename in sorted(os.listdir(class_path)):
            x.append(os.path.join(class_path, image_filename))
            y.append(int(folder) - 1)
    x = np.array(x)
    y = np.array(y)
    return x, y


class FurnituresDatasetWithAugmentation(Sequence):
    def __init__(
            self,
            x_set,
            y_set,
            batch_size,
            input_shape,
            num_classes=128,
            shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.input_shape = input_shape
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
        
    def _data_augmentation(self, images):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(20, 30), # rotate by +20 to +30 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 0.5)), # blur images with a sigma between 0 and 0.5
                            iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 1 and 3
                            iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 1 and 3
                        ]),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), # add gaussian noise to images
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-4, 0),
                                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                second=iaa.ContrastNormalization((0.5, 2.0))
                            )
                        ]),
                        iaa.ContrastNormalization((1.4, 1.6), per_channel=0.5) # improve or worsen the contrast      
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
        return seq.augment_images(images)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_imgs = [cv2.resize(cv2.imread(img_path), 
                                 (self.input_shape[0] + 20, self.input_shape[1] + 20), 
                                 interpolation=cv2.INTER_LINEAR) 
                      for img_path in batch_x]
        batch_imgs = self._data_augmentation(batch_imgs)
        batch_imgs = _preprocess_numpy_input(np.array(batch_imgs),
            data_format='channels_last', mode='torch')

        return batch_imgs, to_categorical(
            np.array(batch_y), num_classes=self.num_classes)


class FurnituresDatasetNoAugmentation(Sequence):
    def __init__(
            self,
            x_set,
            y_set,
            batch_size,
            input_shape,
            num_classes=128):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_imgs = [cv2.resize(cv2.imread(file_name),
                                 self.input_shape,
                                 interpolation=cv2.INTER_LINEAR)
                      for file_name in batch_x]
        batch_imgs = _preprocess_numpy_input(np.array(batch_imgs),
            data_format='channels_last', mode='torch')

        return batch_imgs, to_categorical(
            np.array(batch_y), num_classes=self.num_classes)
