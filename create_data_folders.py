import os
import shutil
from tqdm import tqdm

num_train_samples = len(os.listdir('train/'))
num_valid_samples = len(os.listdir('validation/'))
print('Train images: {}, validation images: {}'.format(
    num_train_samples, num_valid_samples))

train_images = sorted(os.listdir('train/'))
valid_images = sorted(os.listdir('validation/'))

classes = sorted(set([int(image.split('_')[1].split('.')[0])
                      for image in train_images]))
# for cls in classes:
#     os.makedirs(os.path.join('data/train', str(cls)))
#     os.makedirs(os.path.join('data/validation', str(cls)))


def copyFile(src, dest):
    try:
        shutil.copy(src, dest)
    except shutil.Error as e:
        print('Error: %s' % e)
    except IOError as e:
        print('Error: %s' % e.strerror)


# if not os.path.exists('data/train'):
#     os.makedirs('data/train')

# if not os.path.exists('data/validation'):
#     os.makedirs('data/validation')

for image in tqdm(valid_images):
    cls = image.split('_')[1].split('.')[0]
    old_path = os.path.join('validation/', image)
    new_path = os.path.join('data/train', cls)
    shutil.copy2(old_path, os.path.join(new_path, 'val_'+image))

# for image in tqdm(valid_images):
#     cls = image.split('_')[1].split('.')[0]
#     old_path = os.path.join('validation/', image)
#     new_path = os.path.join('data/validation', cls)
#     shutil.copy2(old_path, new_path)

# for image in tqdm(train_images):
#     cls = image.split('_')[1].split('.')[0]
#     old_path = os.path.join('train/', image)
#     new_path = os.path.join('data/train', cls)
#     copyFile(old_path, new_path)
