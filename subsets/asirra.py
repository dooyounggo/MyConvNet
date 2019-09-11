import os
import csv
import shutil
import numpy as np
import subsets.subset_functions as sf

"""
The Asirra dataset
https://www.kaggle.com/c/dogs-vs-cats
"""


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    filenames = os.listdir(subset_dir)
    set_size = len(filenames)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for i, fname in enumerate(filenames):
        if i % 200 == 0:
            print('Saving subset data: {:6d}/{}...'.format(i, set_size))

        img_dir = os.path.join(subset_dir, fname)
        class_name = fname.split('.')[0]
        ext = fname.split('.')[-1]
        if copy:
            shutil.copy2(img_dir, os.path.join(destination_dir, '{:010d}.{}'.format(i, ext)))
        else:
            shutil.move(img_dir, os.path.join(destination_dir, '{:010d}.{}'.format(i, ext)))
        label = 0 if class_name.lower() == 'cat' else 1
        f = open(os.path.join(destination_dir, '{:010d}.csv'.format(i)), 'w', encoding='utf-8', newline='')
        wrt = csv.writer(f)
        wrt.writerow([str(label)])
        f.close()

    print('\nDone')


if __name__ == '__main__':
    subset_dir = "D:/Dropbox/Project/Python/datasets/asirra/validation"
    destination_dir = "D:/Dropbox/Project/Python/tfdatasets/asirra/validation"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = ('cat', 'dog')

    image_dirs, label_dirs = sf.read_subset_cls(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
