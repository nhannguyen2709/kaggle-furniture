import argparse
import cv2
import gc
import os
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator

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


def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)

    return image


def randomShiftScaleRotate(image,
                           shift_limit=(-0.05, 0.05),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, _ = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + \
            np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))

    return image


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


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

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_imgs = []
        for img_path in batch_x:
            img = cv2.imread(img_path)
            img = cv2.resize(
                img,
                self.input_shape,
                interpolation=cv2.INTER_NEAREST)
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-50, 50),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))
            img = randomShiftScaleRotate(img,
                                         shift_limit=(-0.05, 0.05),
                                         scale_limit=(-0, 0),
                                         rotate_limit=(0, 30))
            img = randomHorizontalFlip(img)
            batch_imgs.append(img)

        return np.array(batch_imgs) / 255., to_categorical(
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
                                 interpolation=cv2.INTER_NEAREST)
                      for file_name in batch_x]

        return np.array(batch_imgs) / 255., to_categorical(
            np.array(batch_y), num_classes=self.num_classes)
