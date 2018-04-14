import os

from keras.layers import Dense, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

def evaluate_model_no_data_augmentation(valid_dir, input_shape, checkpoint_dir, base_model, model_name, num_workers):
    valid_datagen = ImageDataGenerator(
        rescale=1. / 255)
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        batch_size=64,
        target_size=input_shape,
        class_mode='categorical',
        shuffle=False)
    x = GlobalMaxPooling2D(name='max_pool')(base_model.layers[-1].output)
    x = Dense(128, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.layers[0].input, outputs=x)

    model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)

    weights_filenames = sorted(os.listdir(model_checkpoint_dir))
    for weights_filename in weights_filenames:
        model.load_weights(os.path.join(model_checkpoint_dir, weights_filename))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
        val_loss, val_acc = model.evaluate_generator(valid_generator, workers=num_workers)
        print(weights_filename, val_loss, val_acc)
