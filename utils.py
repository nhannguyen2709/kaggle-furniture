import numpy as np
import os

from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.layers import Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

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
    output = Dense(128, activation='softmax', name='predictions')(model.layers[-1].output)
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
    output = Dense(128, activation='softmax', name='predictions')(model.layers[-1].output)
    model = Model(inputs=model.layers[0].input, outputs=output)

    return model


def build_inception_v3():
    model = InceptionV3(include_top=False, pooling='avg')
    output = Dense(128, activation='softmax', name='predictions')(model.layers[-1].output)
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
    model = SEInceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
    output = Dense(128, activation='softmax', name='predictions')(model.layers[-1].output)
    model = Model(inputs=model.layers[0].input, outputs=output)

    return model


def build_inception_resnet_v2():
    model = InceptionResNetV2(include_top=False, pooling='avg')
    output = Dense(128, activation='softmax', name='predictions')(model.layers[-1].output)
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
    output = Dense(128, activation='softmax', name='predictions')(model.layers[-1].output)
    model = Model(inputs=model.layers[0].input, outputs=output)
    finetuned_layers_names = ['predictions']
    finetuned_layers = [model.get_layer(name=layer_name) 
                        for layer_name in finetuned_layers_names]
    for layer in model.layers:
        if layer not in finetuned_layers:
            layer.trainable = False
    
    return model


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


def lr_schedule(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        return lr * 0.25
    else:
        return lr
