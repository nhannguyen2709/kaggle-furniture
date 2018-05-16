import argparse
import numpy as np
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.backend import tensorflow_backend as K
from keras.layers import Dense, GlobalMaxPooling2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(
    description='Training')
parser.add_argument(
    '--test-dir',
    default='data/test',
    type=str,
    metavar='PATH',
    help='path to test images')
parser.add_argument(
    '--submit-dir',
    type=str,
    help='path to saved submission file')
parser.add_argument(
    '--submit-fname',
    type=str)
parser.add_argument(
    '--batch-size',
    default=64,
    type=int,
    metavar='N',
    help='mini-batch size')
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


if __name__=='__main__':
    args = parser.parse_args()

    num_workers = args.num_workers
    test_data_dir = args.test_dir
    submit_dir = args.submit_dir
    submit_filename = args.submit_fname

    test_folders = sorted(os.listdir(test_data_dir))
    test_dirs = [os.path.join(test_data_dir, test_folder) for test_folder in test_folders] # ['data/test/test12703'] 

    model_paths = ['checkpoint/{}/iter1.hdf5'.format(args.model_name)] 

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    test_pred = np.zeros((12703, 128))
    pred_times = 0
    for path in model_paths:
        print('\nModel obtained from {}'.format(path))
        model = load_model(path)
        for test_dir in test_dirs:
            print('Data {}'.format(test_dir.split('/')[-1]))
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                batch_size=args.batch_size,
                target_size=(299, 299),
                class_mode='categorical',
                shuffle=False)

            pred = model.predict_generator(
                generator=test_generator, workers=num_workers, verbose=1)
            test_pred += pred
            pred_times += 1
            del test_generator
        K.clear_session()

    test_pred /= pred_times
    np.save(os.path.join(submit_dir, '{}.npy'.format(submit_filename)), test_pred)
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
    my_submit['predicted'].fillna(83, inplace=True)
    my_submit['predicted'] = my_submit['predicted'].astype(int)
    my_submit['predicted'] = my_submit['predicted']

    sample_submit = pd.read_csv('submission/sample_submission_randomlabel.csv')
    missing_pictures_idx = list(set.difference(
        set(sample_submit['id']), set(my_submit['id'])))
    missing_pictures_pred = sample_submit.loc[sample_submit['id'].isin(
        missing_pictures_idx)]
    missing_pictures_pred['predicted'] = 83 # substitute random labels with 83, least frequent class in train dataset

    final_submit = my_submit.append(missing_pictures_pred)
    final_submit.sort_values('id', inplace=True)
    final_submit.to_csv(os.path.join(submit_dir, '{}.csv'.format(submit_filename)), index=False)
