import argparse
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from data import FurnituresDatasetWithAugmentation, FurnituresDatasetNoAugmentation, get_image_paths_and_labels
from model_utils import build_xception, build_densenet_201, build_inception_v3, build_inception_resnet_v2 

parser = argparse.ArgumentParser(
    description='Training')
parser.add_argument(
    '--batch-size',
    default=32,
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
parser.add_argument(
    '--num-layers-trained',
    type=int,
    metavar='N',
    help='number of layers to be trained in second stage')


def train_for_k_iterations(batch_size,
                        input_shape, x_from_train_images,
                        y_from_train_images, model_name,
                        num_workers, num_layers_trained,
                        n_iters=2, resume=args.resume):
    for iter in range(1, n_iters + 1):
        x_train, x_valid, y_train, y_valid = train_test_split(x_from_train_images, y_from_train_images, test_size=0.01)
        print('\nIteration {}'.format(iter))
        print('Found {} images belonging to {} classes'.format(len(x_train), 128))
        print('Found {} images belonging to {} classes'.format(len(x_valid), 128))
        train_generator = FurnituresDatasetWithAugmentation(
            x_train, y_train,
            batch_size=batch_size, input_shape=input_shape)
        valid_generator = FurnituresDatasetNoAugmentation(
            x_valid, y_valid,
            batch_size=batch_size, input_shape=input_shape)

        filepath = 'checkpoint/{}/iter{}.hdf5'.format(model_name,
                                                           iter)
        save_best = ModelCheckpoint(filepath=filepath,
                                    verbose=1,
                                    monitor='val_acc',
                                    save_best_only=True,
                                    mode='max')
        callbacks = [save_best]
        
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
                            workers=num_workers)
            K.clear_session()
        else:
            print('Train the last Dense layer')
            if os.path.exists(filepath):
                model = load_model(filepath)
            elif model_name == 'xception':
                model = build_xception()
                model.compile(optimizer=Adam(lr=1e-3, decay=1e-5), loss='categorical_crossentropy',
                              metrics=['acc'])
            elif model_name == 'inception_v3':
                model = build_inception_v3()
                model.compile(optimizer=Adam(lr=1e-3, decay=1e-5), loss='categorical_crossentropy',
                              metrics=['acc'])
            elif model_name == 'inception_resnet_v2':
                model = build_inception_resnet_v2()
                model.compile(optimizer=Adam(lr=1e-3, decay=1e-5), loss='categorical_crossentropy',
                              metrics=['acc'])
            elif model_name == 'densenet_201':
                model = build_densenet_201()
                model.compile(optimizer=Adam(lr=1e-3, decay=1e-5), loss='categorical_crossentropy',
                              metrics=['acc'])
            model.fit_generator(generator=train_generator,
                                epochs=5,
                                callbacks=callbacks,
                                validation_data=valid_generator,
                                workers=num_workers)
            K.clear_session()

            print("\nFine-tune previous blocks")
            model = load_model(filepath)
            for i in range(1, num_layers_trained):
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

        
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    x_from_train_images, y_from_train_images = get_image_paths_and_labels(
        data_dir='data/train/')
    train_for_k_iterations(args.batch_size, tuple(args.input_shape),
                        x_from_train_images, y_from_train_images, args.model_name, args.num_workers,
                        args.num_layers_trained)
