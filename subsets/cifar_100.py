"""
CIFAR-100 Dataset
https://www.cs.toronto.edu/%7Ekriz/cifar.html
"""

import os
import csv
import shutil
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.io import imsave


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    train_folders = os.listdir(os.path.join(subset_dir, 'train'))
    test_folders = os.listdir(os.path.join(subset_dir, 'test'))

    class_names = []
    i = 0
    i_class = 0
    for folder in train_folders:
        subfolders = os.listdir(os.path.join(subset_dir, 'train', folder))
        for subfolder in subfolders:
            class_names.append(subfolder)
            filenames = os.listdir(os.path.join(subset_dir, 'train', folder, subfolder))
            for fname in filenames:
                if i % 500 == 0:
                    print('Saving training data: {:6d}...'.format(i))

                img_dir = os.path.join(subset_dir, 'train', folder, subfolder, fname)
                image = imread(img_dir)
                imsave(os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, 'jpg')), image, quality=100)
                if not copy:
                    os.remove(img_dir)

                f = open(os.path.join(destination_dir, 'train', '{:010d}.csv'.format(i)),
                         'w', encoding='utf-8', newline='')
                wrt = csv.writer(f)
                wrt.writerow([str(i_class)])
                f.close()
                i += 1

            i_class += 1

    i = 0
    i_class = 0
    for folder in test_folders:
        subfolders = os.listdir(os.path.join(subset_dir, 'test', folder))
        for subfolder in subfolders:
            filenames = os.listdir(os.path.join(subset_dir, 'test', folder, subfolder))
            for fname in filenames:
                if i % 500 == 0:
                    print('Saving test data: {:6d}...'.format(i))

                img_dir = os.path.join(subset_dir, 'test', folder, subfolder, fname)
                image = imread(img_dir)
                imsave(os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i, 'jpg')), image, quality=100)
                if not copy:
                    os.remove(img_dir)

                f = open(os.path.join(destination_dir, 'test', '{:010d}.csv'.format(i)),
                         'w', encoding='utf-8', newline='')
                wrt = csv.writer(f)
                wrt.writerow([str(i_class)])
                f.close()
                i += 1

            i_class += 1

    print('\nDone')

    print('(')
    for i, name in enumerate(class_names):
        print("'{}',".format(name), end='')
        if (i + 1) % 5 == 0:
            print('')
        else:
            print(' ', end='')
    print(')')


if __name__ == '__main__':
    subset_dir = "D:/Dropbox/Project/Python/datasets/cifar-100"
    destination_dir = "D:/Dropbox/Project/Python/tfdatasets/cifar-100"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = (
        'beaver', 'dolphin', 'otter', 'seal', 'whale',
        'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
        'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
        'bottle', 'bowl', 'can', 'cup', 'plate',
        'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
        'clock', 'keyboard', 'lamp', 'telephone', 'television',
        'bed', 'chair', 'couch', 'table', 'wardrobe',
        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
        'bear', 'leopard', 'lion', 'tiger', 'wolf',
        'bridge', 'castle', 'house', 'road', 'skyscraper',
        'cloud', 'forest', 'mountain', 'plain', 'sea',
        'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
        'crab', 'lobster', 'snail', 'spider', 'worm',
        'baby', 'boy', 'girl', 'man', 'woman',
        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
        'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
        'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
        'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'
    )

    image_dirs, label_dirs = sf.read_subset_cls(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
