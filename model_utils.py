import numpy as np
import os

from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate, Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from sklearn.model_selection import RepeatedStratifiedKFold

from data import FurnituresDatasetWithAugmentation, FurnituresDatasetNoAugmentation
from keras_squeeze_excite_network.se_densenet import SEDenseNetImageNet264
from keras_squeeze_excite_network.se_inception_v3 import SEInceptionV3
from keras_squeeze_excite_network.se_inception_resnet_v2 import SEInceptionResNetV2


class MultiGPUModel(Model):
    def __init__(self, base_model, gpus):
        parallel_model = multi_gpu_model(base_model, gpus)
        self.__dict__.update(parallel_model.__dict__)
        self._base_model = base_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._base_model, attrname)

        return super(MultiGPUModel, self).__getattribute__(attrname)


def build_densenet_201():
    model = DenseNet201(include_top=False, pooling='avg')
    output = Dense(128, activation='softmax', name='predictions')(
        model.layers[-1].output)
    model = Model(inputs=model.layers[0].input, outputs=output)
    finetuned_layers_names = [
        'predictions']
    finetuned_layers = [model.get_layer(name=layer_name)
                        for layer_name in finetuned_layers_names]
    for layer in model.layers:
        if layer not in finetuned_layers:
            layer.trainable = False

    return model


def build_se_inception_v3():
    model = SEInceptionV3(include_top=False, weights='imagenet', pooling='avg')
    output = Dense(128, activation='softmax', name='predictions')(
        model.layers[-1].output)
    model = Model(inputs=model.layers[0].input, outputs=output)

    return model


def build_inception_v3():
    model = InceptionV3(include_top=False, pooling='avg')
    output = Dense(128, activation='softmax', name='predictions')(
        model.layers[-1].output)
    model = Model(inputs=model.layers[0].input, outputs=output)
    finetuned_layers_names = [
        'predictions']
    finetuned_layers = [model.get_layer(name=layer_name)
                        for layer_name in finetuned_layers_names]
    for layer in model.layers:
        if layer not in finetuned_layers:
            layer.trainable = False

    return model


def build_se_inception_resnet_v2():
    model = SEInceptionResNetV2(
        include_top=False, weights='imagenet', pooling='avg')
    output = Dense(128, activation='softmax', name='predictions')(
        model.layers[-1].output)
    model = Model(inputs=model.layers[0].input, outputs=output)

    return model


def build_inception_resnet_v2():
    model = InceptionResNetV2(include_top=False, pooling='avg')
    output = Dense(128, activation='softmax', name='predictions')(
        model.layers[-1].output)
    model = Model(inputs=model.layers[0].input, outputs=output)
    finetuned_layers_names = [
        'predictions']
    finetuned_layers = [model.get_layer(name=layer_name)
                        for layer_name in finetuned_layers_names]
    for layer in model.layers:
        if layer not in finetuned_layers:
            layer.trainable = False

    return model


def build_xception():
    model = Xception(include_top=False, pooling='avg')
    max_pool = GlobalMaxPooling2D()(model.layers[-2].output)
    global_pool = Concatenate()([max_pool, model.layers[-1].output])
    global_pool = Dropout(0.5)(global_pool)
    output = Dense(128, activation='softmax', name='predictions')(global_pool)
    model = Model(inputs=model.layers[0].input, outputs=output)
    finetuned_layers_names = ['predictions']
    finetuned_layers = [model.get_layer(name=layer_name)
                        for layer_name in finetuned_layers_names]
    for layer in model.layers:
        if layer not in finetuned_layers:
            layer.trainable = False

    return model


def train_with_kfold_cv(random_state, batch_size,
                        input_shape, x_from_train_images,
                        y_from_train_images, model_name,
                        num_workers, num_layers_trained,
                        n_splits=5, n_repeats=1):
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    fold = 0
    for train_index, test_index in rskf.split(
            x_from_train_images, y_from_train_images):
        fold += 1

        x_train, x_valid = x_from_train_images[train_index], x_from_train_images[test_index]
        y_train, y_valid = y_from_train_images[train_index], y_from_train_images[test_index]
        print('\nFold {}'.format(fold))
        print('Found {} images belonging to {} classes'.format(len(x_train), 128))
        print('Found {} images belonging to {} classes'.format(len(x_valid), 128))
        train_generator = FurnituresDatasetWithAugmentation(
            x_train, y_train,
            batch_size=batch_size, input_shape=input_shape)
        valid_generator = FurnituresDatasetNoAugmentation(
            x_valid, y_valid,
            batch_size=batch_size, input_shape=input_shape)

        filepath = 'checkpoint/{}/fold{}.best.hdf5'.format(model_name,
                                                           fold)
        save_best = ModelCheckpoint(filepath=filepath,
                                    verbose=1,
                                    monitor='val_acc',
                                    save_best_only=True,
                                    mode='max')
        callbacks = [save_best]

        print('Train the last Dense layer')
        if os.path.exists(filepath):
            model = load_model(filepath)
        elif model_name == 'xception':
            model = build_xception()
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
