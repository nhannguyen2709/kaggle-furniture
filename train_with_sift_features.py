import argparse
import gc
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd

from keras.applications.xception import Xception
from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Concatenate, Dense
from keras.losses import categorical_crossentropy
from keras.models import load_model, Model
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras_EMA import ExponentialMovingAverage

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import AugmentedDatasetWithSiftFeatures, DatasetWithSiftFeatures, get_image_paths_and_labels

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
parser.add_argument(
    '--scheme',
    default='trainval',
    type=str)


def train_with_sift_features(batch_size, input_shape,
                             x_train, y_train,
                             x_valid, y_valid,
                             sift_features_train,
                             sift_features_valid,
                             model_name, num_workers,
                             resume):
    print('Found {} images belonging to {} classes'.format(len(x_train), 128))
    print('Found {} images belonging to {} classes'.format(len(x_valid), 128))
    train_generator = AugmentedDatasetWithSiftFeatures(
        x_train, y_train, sift_features_train,
        batch_size=batch_size, input_shape=input_shape)
    valid_generator = DatasetWithSiftFeatures(
        x_valid, y_valid, sift_features_valid,
        batch_size=batch_size, input_shape=input_shape)
    class_weight = compute_class_weight(
        'balanced', np.unique(y_train), y_train)
    class_weight_dict = dict.fromkeys(np.unique(y_train))
    for key in class_weight_dict.keys():
        class_weight_dict.update({key: class_weight[key]})


    filepath = 'checkpoint/{}/sift_iter1.hdf5'.format(model_name)
    save_best = ModelCheckpoint(filepath=filepath,
                                verbose=1,
                                monitor='val_acc',
                                save_best_only=True,
                                mode='max')
    save_on_train_end = ModelCheckpoint(filepath=filepath,
                                        verbose=1,
                                        monitor='val_acc',
                                        period=args.epochs)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                                  factor=0.2,
                                  patience=2,
                                  verbose=1)
    callbacks = [save_best, save_on_train_end, reduce_lr]

    if resume == 'True':
        print('\nResume training from the last checkpoint')
        model = load_model(filepath)
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        print('Trainable params: {:,}'.format(trainable_count))
        model.fit_generator(generator=train_generator,
                            epochs=args.epochs,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            class_weight=class_weight_dict,
                            workers=num_workers)
    else:
        model = Xception(include_top=False, pooling='max')
        sift_features = Input(shape=(512, ))
        x = Concatenate()([model.layers[-1].output, sift_features])
        x = Dense(units=128, activation='linear', name='predictions', kernel_regularizer=regularizers.l2(0.0001))(x)
        model = Model([model.layers[0].input, sift_features], x)

        for layer in model.layers[:-1]:
            layer.trainable = False

        model.compile(optimizer=Adam(lr=0.001), loss='categorical_hinge',
                      metrics=['acc'])
        model.fit_generator(generator=train_generator,
                            epochs=5,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            class_weight=class_weight_dict,
                            workers=num_workers)
        K.clear_session()

        print("\nFine-tune the network")
        model = load_model(filepath)   
        for layer in model.layers:
            layer.trainable = True
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        print('Trainable params: {:,}'.format(trainable_count))
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='categorical_hinge',
                      metrics=['acc'])
        model.fit_generator(generator=train_generator,
                            epochs=30,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            class_weight=class_weight_dict,
                            workers=num_workers)
        K.clear_session()


if __name__ == '__main__':
    args = parser.parse_args()

    train_df = pd.read_csv('sift_train.csv')
    valid_df = pd.read_csv('sift_valid.csv')
    merged_df = pd.concat([train_df, valid_df], ignore_index=True)
    y = merged_df.pop('2')
    y -= 1
    train_idx, valid_idx, y_train, y_valid = train_test_split(merged_df.index, y, test_size=0.01)
    x_train, sift_features_train = merged_df.iloc[train_idx]['1'].values, merged_df.iloc[train_idx].ix[:, '3':].values
    x_valid, sift_features_valid = merged_df.iloc[valid_idx]['1'].values, merged_df.iloc[valid_idx].ix[:, '3':].values
    del train_df, valid_df, merged_df, train_idx, valid_idx
    gc.collect()
    ss = StandardScaler()
    ss.fit(sift_features_train)
    sift_features_train = ss.transform(sift_features_train)
    sift_features_valid = ss.transform(sift_features_valid)

    train_with_sift_features(args.batch_size, tuple(args.input_shape),
        x_train, y_train,
        x_valid, y_valid,
        sift_features_train,
        sift_features_valid,
        args.model_name, args.num_workers,
        args.resume)
