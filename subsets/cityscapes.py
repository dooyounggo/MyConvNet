"""
Cityscapes dataset for image segmentation.
https://www.cityscapes-dataset.com/
"""

import os
import shutil
import numpy as np
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.io import imsave


CITY_COLORMAP = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                 [0, 80, 100], [0, 0, 230], [119, 11, 32]]

ID_TO_LABEL_MAP = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0,
                            7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0])
# ID_TO_LABEL_MAP = np.array([1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 4, 5, 6, 1, 1, 1, 7, 1,
#                             8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 1, 18, 19, 20, 0])
# ID_TO_LABEL_MAP = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 0])


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    image_folder_train = os.path.join(subset_dir, 'leftImg8bit', 'train')
    train_images = []
    for folder in os.listdir(image_folder_train):
        curr_dir = os.path.join(image_folder_train, folder)
        for file in os.listdir(curr_dir):
            train_images.append(os.path.join(curr_dir, file))

    image_folder_val = os.path.join(subset_dir, 'leftImg8bit', 'val')
    val_images = []
    for folder in os.listdir(image_folder_val):
        curr_dir = os.path.join(image_folder_val, folder)
        for file in os.listdir(curr_dir):
            val_images.append(os.path.join(curr_dir, file))

    image_folder_test = os.path.join(subset_dir, 'leftImg8bit', 'test')
    test_images = []
    for folder in os.listdir(image_folder_test):
        curr_dir = os.path.join(image_folder_test, folder)
        for file in os.listdir(curr_dir):
            test_images.append(os.path.join(curr_dir, file))

    label_folder_train = os.path.join(subset_dir, 'gtFine', 'train')
    train_labels = []
    for folder in os.listdir(label_folder_train):
        curr_dir = os.path.join(label_folder_train, folder)
        for file in os.listdir(curr_dir):
            if file.split('_')[-1].lower() == 'labelids.png':
                train_labels.append(os.path.join(curr_dir, file))

    label_folder_val = os.path.join(subset_dir, 'gtFine', 'val')
    val_labels = []
    for folder in os.listdir(label_folder_val):
        curr_dir = os.path.join(label_folder_val, folder)
        for file in os.listdir(curr_dir):
            if file.split('_')[-1].lower() == 'labelids.png':
                val_labels.append(os.path.join(curr_dir, file))

    label_folder_test = os.path.join(subset_dir, 'gtFine', 'test')
    test_labels = []
    for folder in os.listdir(label_folder_test):
        curr_dir = os.path.join(label_folder_test, folder)
        for file in os.listdir(curr_dir):
            if file.split('_')[-1].lower() == 'labelids.png':
                test_labels.append(os.path.join(curr_dir, file))

    for i, (image_dir, label_dir) in enumerate(zip(train_images, train_labels)):
        if i % 200 == 0:
            print('Saving training data: {:6d}/{}...'.format(i, len(train_images)))

        image = imread(image_dir)
        imsave(os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, 'jpg')), image)
        if not copy:
            os.remove(label_dir)

        label_ext = label_dir.split('.')[-1]
        mask = imread(label_dir).astype(np.uint8)
        label = ID_TO_LABEL_MAP[mask].astype(np.uint8)

        imsave(os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, label_ext)), label)
        if not copy:
            os.remove(label_dir)

    for i, (image_dir, label_dir) in enumerate(zip(val_images, val_labels)):
        if i % 200 == 0:
            print('Saving validation data: {:6d}/{}...'.format(i, len(val_images)))

        image = imread(image_dir)
        imsave(os.path.join(destination_dir, 'validation', '{:010d}.{}'.format(i, 'jpg')), image)
        if not copy:
            os.remove(label_dir)

        label_ext = label_dir.split('.')[-1]
        mask = imread(label_dir).astype(np.uint8)
        label = ID_TO_LABEL_MAP[mask].astype(np.uint8)

        imsave(os.path.join(destination_dir, 'validation', '{:010d}.{}'.format(i, label_ext)), label)
        if not copy:
            os.remove(label_dir)

    for i, (image_dir, label_dir) in enumerate(zip(test_images, test_labels)):
        if i % 200 == 0:
            print('Saving test data: {:6d}/{}...'.format(i, len(test_images)))

        image = imread(image_dir)
        imsave(os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i, 'jpg')), image)
        if not copy:
            os.remove(label_dir)

        label_ext = label_dir.split('.')[-1]
        mask = imread(label_dir).astype(np.uint8)
        label = ID_TO_LABEL_MAP[mask].astype(np.uint8)

        imsave(os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i, label_ext)), label)
        if not copy:
            os.remove(label_dir)

    print('\nDone')


if __name__ == '__main__':
    subset_dir = "D:/Dropbox/Project/Python/datasets/cityscapes"
    destination_dir = "D:/Dropbox/Project/Python/tfdatasets/cityscapes"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = (
        'backgrounds', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
        'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    )

    image_dirs, label_dirs = sf.read_subset_seg(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
