import os
import csv
import shutil
import numpy as np
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.transform import resize

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

    filenames = os.listdir(subset_dir)
    image_dirs = []
    label_dirs = []
    for fname in filenames:
        ext = fname.split('.')[-1].lower()
        full_filename = os.path.join(subset_dir, fname)
        if ext == 'csv':
            label_dirs.append(full_filename)
        elif ext == 'jpg' or ext == 'jpeg':
            image_dirs.append(full_filename)

    set_size = len(image_dirs)
    if len(label_dirs) == 0:
        label_dirs = None
    else:
        assert len(image_dirs) == len(label_dirs),\
            'Number of examples mismatch: {} images vs. {} labels'.format(len(image_dirs), len(label_dirs))

    if sample_size is not None and sample_size < set_size:
        if shuffle:
            idx = np.random.choice(np.arange(set_size), size=sample_size, replace=False).astype(int)
            image_dirs = list(np.array(image_dirs)[idx])
            label_dirs = list(np.array(label_dirs)[idx])
        else:
            image_dirs = image_dirs[:sample_size]
            label_dirs = label_dirs[:sample_size]
    else:
        if shuffle:
            idx = np.arange(set_size)
            np.random.shuffle(idx)
            image_dirs = list(np.array(image_dirs)[idx])
            label_dirs = list(np.array(label_dirs)[idx])

    return image_dirs, label_dirs, class_names
