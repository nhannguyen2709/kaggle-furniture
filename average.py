import numpy as np
import pandas as pd
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.preprocessing.image import ImageDataGenerator

test_pred1 = np.load('submission/xception/iter12.npy')
test_pred2 = np.load('submission/inception_v3/avg_train_finetune_12_crops.npy')
test_pred = (test_pred1 + test_pred2) / 2
test_pred = np.argmax(test_pred, axis=1)
test_pred = test_pred + 1.

test_datagen = ImageDataGenerator(
    rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    'data/test/test12703',
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

final_submit.to_csv('submission/ensemble/ensemble1.csv', index=False)
