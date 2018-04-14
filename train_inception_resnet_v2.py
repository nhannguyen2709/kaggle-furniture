import argparse
import cv2
import os
import numpy as np

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

from utils import evaluate_model_no_data_augmentation
from keras_CLR import CyclicLR

train_dir = 'data/train'
valid_dir = "data/validation/"

# limit tensorflow's memory usage
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# K.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Script to train models')
parser.add_argument(
    '--epochs',
    default=100,
    type=int,
    metavar='N',
    help='number of total epochs')
parser.add_argument(
    '--batch-size',
    default=64,
    type=int,
    metavar='N',
    help='mini-batch size')
parser.add_argument(
    '--num-classes',
    default=128,
    type=int,
    metavar='N',
    help='number of classes')
parser.add_argument(
    '--train-lr',
    default=1e-3,
    type=float,
    metavar='LR',
    help='learning rate of train stage')
parser.add_argument(
    '--finetune-lr',
    default=1e-5,
    type=float,
    metavar='LR',
    help='learning rate of finetune stage')
parser.add_argument(
    '--checkpoint-path',
    default='checkpoint/inception_resnet_v2/weights.{epoch:02d}-{val_acc:.4f}.hdf5',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    type=str,
    metavar='PATH',
    help='path to latest checkpoint')
parser.add_argument(
    '--num-workers',
    default=4,
    type=int,
    metavar='N',
    help='maximum number of processes to spin up')
parser.add_argument(
    '--initial-epoch',
    default=20,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')


def train():
    global args
    args = parser.parse_args()
    print(args)

    model = InceptionResNetV2(include_top=False)                                                                                                                                                                                                                                                                                        
    x = GlobalMaxPooling2D(name='max_pool')(model.layers[-1].output)
    x = Dense(args.num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=model.layers[0].input, outputs=x)
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.summary()                                                                                                                                                                     
    if os.path.exists(
            'checkpoint/inception_resnet_v2/weights.best.hdf5'):
        model.load_weights(
            'checkpoint/inception_resnet_v2/weights.best.hdf5')

    img_size = (299, 299)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True)
    valid_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True)                                                                                                           

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=args.batch_size,
        target_size=img_size,
        class_mode='categorical',
        shuffle=True)
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        batch_size=args.batch_size,
        target_size=img_size,
        class_mode='categorical',
        shuffle=False)

    model.compile(optimizer=Adam(lr=args.train_lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc', verbose=1,
                                 mode='max', period=5)
    save_best = ModelCheckpoint(
        'checkpoint/inception_resnet_v2/weights.best.hdf5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, verbose=1)
    callbacks = [checkpoint, save_best, reduce_lr]

    model.fit_generator(generator=train_generator,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=args.num_workers,
                        initial_epoch=args.initial_epoch)


def train_with_finetune(finetuned_layers_names):
    global args
    args = parser.parse_args()

    model = InceptionResNetV2(include_top=False)
    x = GlobalMaxPooling2D(name='max_pool')(model.layers[-1].output)
    x = Dense(args.num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=model.layers[0].input, outputs=x)

    # Choose which layers to finetune
    finetuned_layers = [
        model.get_layer(
            name=layer_name) for layer_name in finetuned_layers_names]
    for layer in model.layers:
        if layer not in finetuned_layers:
            layer.trainable = False

    model.summary()
    # if os.path.exists(
    #         'checkpoint/inception_resnet_v2/weights.best.hdf5'):
    #     model.load_weights(
    #         'checkpoint/inception_resnet_v2/weights.best.hdf5')
    model.load_weights('checkpoint/inception_resnet_v2/weights.55-0.6029.hdf5')

    img_size = (299, 299)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
       	width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True)
    valid_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # width_shift_range=0.05,
        # height_shift_range=0.05,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=args.batch_size,
        target_size=img_size,
        class_mode='categorical',
        shuffle=True)
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        batch_size=args.batch_size,
        target_size=img_size,
        class_mode='categorical',
        shuffle=False)
    
    # Visualize data augmentation
    # x, _ = train_generator.next()
 
    # import cv2
    # for i in range(len(x)):
    #     img = x[i]
    #     cv2.imwrite('img'+str(i)+'.jpg', img)

    model.compile(optimizer=SGD(momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc', verbose=1,
                                 mode='max', period=5)
    save_best = ModelCheckpoint(
        'checkpoint/inception_resnet_v2/weights.best.hdf5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    clr_triangular = CyclicLR(mode='triangular', max_lr=1e-3, step_size=2993)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    #                               patience=5, verbose=1)
    callbacks = [clr_triangular, checkpoint, save_best]

    model.fit_generator(generator=train_generator,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=args.num_workers,
                        initial_epoch=args.initial_epoch)


if __name__ == '__main__':
    # Print out model's summary to get layer names
    # model = InceptionResNetV2(include_top=False)
    # x = GlobalMaxPooling2D(name='max_pool')(model.layers[-1].output)
    # x = Dense(128, activation='softmax', name='predictions')(x)
    # model = Model(inputs=model.layers[0].input, outputs=x)
    # print(len(model.layers))
    # model.summary()

    #train()
    #K.clear_session()
    # 'block8_7_conv', 'conv2d_191', 'conv2d_193', 'conv2d_194',
    # 'conv2d_192', 'conv2d_195', 'block8_8_conv', 'conv2d_197',

    train_with_finetune(finetuned_layers_names=[ 
                                                  
                                                'conv2d_198', 'conv2d_196', 'conv2d_199', 'block8_9_conv', 
                                                'conv2d_201', 'conv2d_202', 'conv2d_200', 'conv2d_203',  
                                                'block8_10_conv', 'conv_7b', 'predictions'])

    # evaluate_model_no_data_augmentation(valid_dir=valid_dir,
    #     input_shape=(299, 299), checkpoint_dir='checkpoint/',
    #     base_model=InceptionResNetV2(include_top=False), model_name='inception_resnet_v2',
    #     num_workers=args.num_workers)
