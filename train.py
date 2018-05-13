import argparse

from model_utils import train_with_kfold_cv
from data import get_image_paths_and_labels

parser = argparse.ArgumentParser(
    description='Training')
parser.add_argument(
    '--random-state',
    type=int,
    metavar='N',
    help='random state when splitting data into k folds')
parser.add_argument(
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size')
parser.add_argument(
    '--input-shape',
    default=(299, 299),
    type=tuple)
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
    '--num-layers-trained',
    type=int,
    metvar='N',
    help='number of layers to be trained in second stage')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    x_from_train_images, y_from_train_images = get_image_paths_and_labels(
        data_dir='data/train/')
    train_with_kfold_cv(args.random_state, args.batch_size, args.input_shape,
                        x_from_train_images, y_from_train_images, args.model_name, args.num_workers,
                        args.num_layers_trained)
