import numpy as np
import pandas as pd
import os

from keras.applications.densenet import DenseNet201
from keras.layers import Dense, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

test_dir = 'data/test'
num_workers = 10
submit_dir = 'submission/densenet_201'
submit_filename = 'avg_2_folds.csv'

test_datagen = ImageDataGenerator(
    rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    batch_size=64,
    target_size=(299, 299),
    class_mode='categorical',
    shuffle=False)

folds = ['fold1', 'fold2']
predictions = []
for fold in folds:
    model = DenseNet201(include_top=False)
    x = GlobalMaxPooling2D(name='max_pool')(model.layers[-1].output)
    x = Dense(128, activation='softmax', name='predictions')(x)
    model = Model(inputs=model.layers[0].input, outputs=x)
    model.load_weights('checkpoint/densenet_201/{}.weights.best.hdf5'.format(fold))
    fold_pred = model.predict_generator(generator=test_generator, workers=num_workers, verbose=1) 
    predictions.append(fold_pred)

test_pred = np.mean(np.array(predictions), axis=0)
test_pred = np.argmax(test_pred, axis=1)
test_pred = test_pred + 1.

my_submit = pd.concat([pd.Series(test_generator.filenames),
                       pd.Series(test_pred)], axis=1)
my_submit.columns = ['id', 'predicted']
my_submit['id'] = my_submit['id'].map(lambda x: int(x.replace('test12695/', '').replace('.jpg', '')))
my_submit['predicted'].fillna(-1, inplace=True)
my_submit['predicted'] = my_submit['predicted'].astype(int)
my_submit['predicted'] = my_submit['predicted']

sample_submit = pd.read_csv('submission/sample_submission_randomlabel.csv')
missing_pictures_idx = list(set.difference(set(sample_submit['id']), set(my_submit['id'])))
missing_pictures_pred = sample_submit.loc[sample_submit['id'].isin(missing_pictures_idx)]

final_submit = my_submit.append(missing_pictures_pred)
final_submit.loc[final_submit.predicted==-1, 'predicted'] = 101
final_submit.sort_values('id', inplace = True)
final_submit.to_csv(os.path.join(submit_dir, submit_filename), index=False)