import cv2
from imgaug import augmenters as iaa
import os
import shutil
from tqdm import tqdm

def copyFile(src, dest):
    try:
        shutil.copy(src, dest)
    except shutil.Error as e:
        print('Error: %s' % e)
    except IOError as e:
        print('Error: %s' % e.strerror)

def create_train_and_val_folders():
    num_train_samples = len(os.listdir('train/'))
    num_valid_samples = len(os.listdir('validation/'))
    print('Train images: {}, validation images: {}'.format(
        num_train_samples, num_valid_samples))
    train_images = sorted(os.listdir('train/'))
    valid_images = sorted(os.listdir('validation/'))
    labels_idxs = sorted(set([int(image.split('_')[1].split('.')[0])
                        for image in train_images]))

    if not os.path.exists('data/train'):
        os.makedirs('data/train')
    if not os.path.exists('data/validation'):
        os.makedirs('data/validation')
    for label_idx in labels_idxs:
        shutil.rmtree(os.path.join('data/train', str(label_idx)))
        os.makedirs(os.path.join('data/train', str(label_idx)))
        os.makedirs(os.path.join('data/validation', str(label_idx)))

    for image in tqdm(train_images):
        label_idx = image.split('_')[1].split('.')[0]
        old_path = os.path.join('train/', image)
        new_path = os.path.join('data/train', label_idx)
        copyFile(old_path, new_path)
    for image in tqdm(valid_images):
        label_idx = image.split('_')[1].split('.')[0]
        old_path = os.path.join('validation/', image)
        new_path = os.path.join('data/validation', label_idx)
        copyFile(old_path, new_path)

def crop_and_save_imgs(percent_cropped=0.1):
    for img_folder in sorted(os.listdir('data/train')):
        img_folder_path = os.path.join('data/train', img_folder)
        print('\n {}'.format(img_folder_path))
        for img_file in tqdm(sorted(os.listdir(img_folder_path))):
            img_file_path = os.path.join(img_folder_path, img_file)
            img_arr = cv2.imread(img_file_path)
            img_arr = iaa.Crop(px=(int(percent_cropped*img_arr.shape[0]),
                                    int(percent_cropped*img_arr.shape[1]),
                                    int(percent_cropped*img_arr.shape[0]),
                                    int(percent_cropped*img_arr.shape[1])),
                                keep_size=False).augment_image(img_arr)
            cv2.imwrite(img_file_path, img_arr)

    for img_folder in sorted(os.listdir('data/validation')):
        img_folder_path = os.path.join('data/validation', img_folder)
        print('\n {}'.format(img_folder_path))
        for img_file in tqdm(sorted(os.listdir(img_folder_path))):
            img_file_path = os.path.join(img_folder_path, img_file)
            img_arr = cv2.imread(img_file_path)
            img_arr = iaa.Crop(px=(int(percent_cropped*img_arr.shape[0]),
                                    int(percent_cropped*img_arr.shape[1]),
                                    int(percent_cropped*img_arr.shape[0]),
                                    int(percent_cropped*img_arr.shape[1])),
                                keep_size=False).augment_image(img_arr)
            cv2.imwrite(img_file_path, img_arr)

if __name__=='__main__':
    crop_and_save_imgs(0.1)