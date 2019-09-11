import os
import shutil
import numpy as np
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

"""
Oxford-IIIT Pet dataset for two-class segmentation
https://research.sualab.com/practice/2018/11/23/image-segmentation-deep-learning.html
"""


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    image_names = os.listdir(os.path.join(subset_dir, 'images'))
    label_names = os.listdir(os.path.join(subset_dir, 'masks'))

    labels = []
    class_names = []
    set_size = len(image_names)

    assert len(image_names) == len(label_names), 'Number of examples mismatch'

    for i in range(set_size):
        if i % 200 == 0:
            print('Saving subset data: {:6d}/{}...'.format(i, set_size))

        image_dir = os.path.join(subset_dir, 'images', image_names[i])
        image_ext = image_dir.split('.')[-1]
        if copy:
            shutil.copy2(image_dir, os.path.join(destination_dir, '{:010d}.{}'.format(i, image_ext)))
        else:
            shutil.move(image_dir, os.path.join(destination_dir, '{:010d}.{}'.format(i, image_ext)))

        label_dir = os.path.join(subset_dir, 'masks', label_names[i])
        label_ext = label_dir.split('.')[-1]
        label_name = label_names[i].split('.')[0]

        mask = imread(label_dir)
        label = np.zeros(mask.shape, dtype=np.uint8)

        bkgd_idx = np.where(mask == 2)
        label[bkgd_idx[0], bkgd_idx[1]] = 1  # Background
        fore_idx = np.where(mask == 1)  # Foreground
        if label_name[0].isupper():
            label[fore_idx[0], fore_idx[1]] = 2  # Cat
        else:
            label[fore_idx[0], fore_idx[1]] = 3  # Dog

        imsave(os.path.join(destination_dir, '{:010d}.{}'.format(i, label_ext)), label)
        if not copy:
            os.remove(label_dir)

    print('\nDone')


if __name__ == '__main__':
    subset_dir = "D:/Python/datasets/oxford-iiit-pet-dataset-sualab/train"
    destination_dir = "D:/Python/tfdatasets/oxford-iiit-pet-dataset-sualab/train"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = ('background', 'cat', 'dog')

    filenames = os.listdir(subset_dir)
    image_dirs = []
    label_dirs = []
    for fname in filenames:
        ext = fname.split('.')[-1].lower()
        full_filename = os.path.join(subset_dir, fname)
        if ext == 'png':
            label_dirs.append(full_filename)
        elif ext == 'jpg' or ext == 'jpeg':
            image_dirs.append(full_filename)

    set_size = len(image_dirs)
    if len(label_dirs) == 0:
        label_dirs = None
    else:
        assert len(image_dirs) == len(label_dirs), \
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
