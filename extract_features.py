import pandas as pd
import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalMaxPooling2D
from keras.models import Model

from data import FurnituresDatasetNoLabels
from utils import evaluate_model_no_data_augmentation, get_image_paths_and_labels

# extract features
x_from_train_images, y_from_train_images = get_image_paths_and_labels(
    data_dir='data/train/')
x_from_valid_images, y_from_valid_images = get_image_paths_and_labels(
    data_dir='data/validation/')

model = InceptionResNetV2(include_top=False)
features = GlobalMaxPooling2D(name='max_pool')(model.layers[-1].output)
features_extractor = Model(inputs=model.layers[0].input, outputs=features)

train_data = FurnituresDatasetNoLabels(x_from_train_images, batch_size=16)
features = features_extractor.predict_generator(train_data, workers=4, verbose=1)
train_df = pd.concat([pd.Series(x_from_train_images),
                      pd.Series(y_from_train_images),
                      pd.DataFrame(features)],
                     axis=1)

del features

valid_data = FurnituresDatasetNoLabels(x_from_valid_images, batch_size=16)
features = features_extractor.predict_generator(valid_data, workers=4, verbose=1)
valid_df = pd.concat([pd.Series(x_from_valid_images),
                      pd.Series(y_from_valid_images),
                      pd.DataFrame(features)],
                     axis=1)

merged_df = train_df.append(valid_df)
merged_df.to_csv('extracted_features.csv', index=False)