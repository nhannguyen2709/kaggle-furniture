import numpy as np
import pandas as pd
import os

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.layers import Dense, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from model_utils import build_xception

num_workers = 4
test_data_dir = 'data/test'
submit_dir = 'submission/xception'
checkpoint_dir = 'checkpoint/xception'
submit_filename = 'weights2.csv'

test_folders = sorted(os.listdir(test_data_dir))
test_dirs = [[os.path.join(test_data_dir, test_folder)
             for test_folder in test_folders][0]]
folds = sorted(os.listdir(checkpoint_dir))

test_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_pred = np.zeros((12703, 128))
pred_times = 0
for test_dir in test_dirs:
    print('\nData {}'.format(test_dir.split('/')[-1]))
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        batch_size=64,
        target_size=(299, 299),
        class_mode='categorical',
        shuffle=False)
    for fold in folds:
        pred_times += 1	    
        print('Model obtained from {}'.format(fold))
        model = build_xception()
        model.load_weights(os.path.join(checkpoint_dir, fold))
        fold_pred = model.predict_generator(
            generator=test_generator, workers=num_workers, verbose=1)
        K.clear_session()
        test_pred += fold_pred
    del test_generator

test_pred /= pred_times
np.save('submission/xception/avg_train_finetune_12_crops.npy', test_pred)
test_pred = np.argmax(test_pred, axis=1)
test_pred = test_pred + 1.
# recreate test generator to extract image filenames
test_generator = test_datagen.flow_from_directory(
    test_dir,
    batch_size=64,
    target_size=(299, 299),
    class_mode='categorical',
    shuffle=False)

my_submit = pd.concat([pd.Series(test_generator.filenames),
                       pd.Series(test_pred)], axis=1)
my_submit.columns = ['id', 'predicted']
my_submit['id'] = my_submit['id'].map(lambda x: int(
    x.split('/')[-1].split('.')[0]))
my_submit['predicted'].fillna(-1, inplace=True)
my_submit['predicted'] = my_submit['predicted'].astype(int)
my_submit['predicted'] = my_submit['predicted']

sample_submit = pd.read_csv('submission/sample_submission_randomlabel.csv')
missing_pictures_idx = list(set.difference(
    set(sample_submit['id']), set(my_submit['id'])))
missing_pictures_pred = sample_submit.loc[sample_submit['id'].isin(
    missing_pictures_idx)]

final_submit = my_submit.append(missing_pictures_pred)
final_submit.loc[final_submit.predicted == -1, 'predicted'] = 101
final_submit.sort_values('id', inplace=True)
final_submit.to_csv(os.path.join(submit_dir, submit_filename), index=False)
