import argparse
from keras.models import load_model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from data import FurnituresDatasetNoAugmentation, get_image_paths_and_labels

parser = argparse.ArgumentParser(
    description='Validating')
parser.add_argument(
    '--batch-size',
    default=64,
    type=int,
    metavar='N',
    help='mini-batch size')
parser.add_argument(
    '--input-shape',
    nargs='+',
    type=int)
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


if __name__== '__main__':
    args = parser.parse_args()

    x_valid, y_valid = get_image_paths_and_labels('data/validation/')
    valid_generator = FurnituresDatasetNoAugmentation(
            x_valid, y_valid, 
            batch_size=args.batch_size, input_shape=tuple(args.input_shape))

    filepath = 'checkpoint/{}/iter{}.hdf5'.format(args.model_name, 1)
    model = load_model(filepath)
    val_loss, val_acc = model.evaluate_generator(valid_generator, workers=args.num_workers, verbose=1)
    print('\n {} - val_loss: {}- val_acc: {}'.format(filepath.split('/')[1], val_loss, val_acc))
