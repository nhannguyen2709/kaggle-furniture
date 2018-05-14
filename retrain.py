import argparse
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from data import FurnituresDatasetWithAugmentation, FurnituresDatasetNoAugmentation, get_image_paths_and_labels
from keras_EMA import ExponentialMovingAverage

parser = argparse.ArgumentParser(
    description='Retraining')
parser.add_argument(
    '--batch-size',
    default=8,
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
    help='model to be retrained')
parser.add_argument(
    '--num-workers',
    default=4,
    type=int,
    metavar='N',
    help='maximum number of processes to spin up')


def retrain_for_k_iterations(batch_size,
                        input_shape, merged_x,
                        merged_y, model_name,
                        num_workers,
                        resume, n_iters=2):
    for iter in range(1, n_iters + 1):
        x_train, x_valid, y_train, y_valid = train_test_split(merged_x, merged_y, test_size=0.005)
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
        save_best = ExponentialMovingAverage(filepath=filepath,
                                        verbose=1,
                                        monitor='val_acc',
                                        save_best_only=True,
                                        mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_acc',
            factor=0.2,
            patience=2)
        callbacks = [save_best, reduce_lr]

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
            model = load_model(filepath)
            model.compile(optimizer=Adam(lr=1e-3, decay=1e-5),
                        loss='categorical_crossentropy',
                        metrics=['acc'])
            model.fit_generator(generator=train_generator,
                        epochs=20,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        workers=num_workers)
            K.clear_session()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    x_from_train_images, y_from_train_images = get_image_paths_and_labels(
    data_dir='data/train/')
    x_from_val_images, y_from_val_images = get_image_paths_and_labels(
        data_dir='data/validation/')
    merged_x, merged_y = np.concatenate((x_from_train_images, x_from_val_images)), np.concatenate((y_from_train_images, y_from_val_images))

    retrain_for_k_iterations(args.batch_size, tuple(args.input_shape),
                        merged_x, merged_y, args.model_name, args.num_workers, arg.resume)
