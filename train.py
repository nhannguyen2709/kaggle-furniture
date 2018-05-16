import argparse
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_ClipLR import ClipLR
from keras_EMA import ExponentialMovingAverage
from keras.models import load_model
from keras.optimizers import Adam
from keras import regularizers

from sklearn.utils.class_weight import compute_class_weight

from data import FurnituresDatasetWithAugmentation, FurnituresDatasetNoAugmentation, get_image_paths_and_labels
from model_utils import build_xception, build_densenet_201, build_inception_v3, build_inception_resnet_v2 

parser = argparse.ArgumentParser(
    description='Training')
parser.add_argument(
    '--batch-size',
    default=16,
    type=int,
    metavar='N',
    help='mini-batch size')
parser.add_argument(
    '--input-shape',
    nargs='+',
    type=int)
parser.add_argument(
    '--epochs',
    default=20,
    type=int,
    metavar='N',
    help='number of epochs when resuming')
parser.add_argument(
    '--resume',
    default='False',
    type=str,
    help='indicate whether to continue training')
parser.add_argument(
    '--model-name',
    type=str,
    help='model to be trained')
parser.add_argument(
    '--num-workers',
    default=4,
    type=int,
    metavar='N',
    help='maximum number of processes to spin up')


def train(batch_size, input_shape,
    x_train, y_train,
    x_valid, y_valid, 
    model_name, num_workers, 
    resume):
    print('Found {} images belonging to {} classes'.format(len(x_train), 128))
    print('Found {} images belonging to {} classes'.format(len(x_valid), 128))
    train_generator = FurnituresDatasetWithAugmentation(
        x_train, y_train,
        batch_size=batch_size, input_shape=input_shape)
    valid_generator = FurnituresDatasetNoAugmentation(
        x_valid, y_valid,
        batch_size=batch_size, input_shape=input_shape)
    class_weight = compute_class_weight('balanced'
                                               ,np.unique(y_train)
                                               ,y_train)

    filepath = 'checkpoint/{}/iter1.hdf5'.format(model_name)
    save_best = ExponentialMovingAverage(filepath=filepath,
                                verbose=1,
                                monitor='val_acc',
                                save_best_only=True,
                                mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
        factor=0.1,
        patience=2,
        verbose=1)
    clip_lr = ClipLR(verbose=1)
    callbacks = [save_best, reduce_lr, clip_lr]
    
    if resume == 'True':
        print('Resume training from the last checkpoint')
        model = load_model(filepath)
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        print('Trainable params: {:,}'.format(trainable_count))
        model.fit_generator(generator=train_generator,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        class_weight=class_weight,
                        workers=num_workers)
        K.clear_session()
    else:
        print('Train the last Dense layer')
        if os.path.exists(filepath):
            model = load_model(filepath)
        elif model_name == 'xception':
            model = build_xception()
            model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                            metrics=['acc'])
        elif model_name == 'inception_v3':
            model = build_inception_v3()
            model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                            metrics=['acc'])
        elif model_name == 'inception_resnet_v2':
            model = build_inception_resnet_v2()
            model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                            metrics=['acc'])
        elif model_name == 'densenet_201':
            model = build_densenet_201()
            model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                            metrics=['acc'])
        model.fit_generator(generator=train_generator,
                            epochs=1,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            class_weight=class_weight,
                            workers=num_workers)
        K.clear_session()

        print("\nFine-tune the network")
        model = load_model(filepath)
        for layer in model.layers:
            layer.trainable = True
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= regularizers.l2(0.0001)
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        print('Trainable params: {:,}'.format(trainable_count))
        model.compile(optimizer=Adam(lr=0.00003),
                        loss='categorical_crossentropy',
                        metrics=['acc'])
        model.fit_generator(generator=train_generator,
                            epochs=19,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            class_weight=class_weight,
                            workers=num_workers)
        K.clear_session()

        
if __name__ == '__main__':
    args = parser.parse_args()

    x_train, y_train = get_image_paths_and_labels(
        data_dir='data/train/')
    x_valid, y_valid = get_image_paths_and_labels(
        data_dir='data/validation/')

    train(args.batch_size, tuple(args.input_shape),
                        x_train, y_train, 
                        x_valid, y_valid, 
                        args.model_name, args.num_workers,
                        args.resume)
