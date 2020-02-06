"""
Flickr-Faces-HQ Dataset
https://github.com/NVlabs/ffhq-dataset
"""

import os
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.io import imsave


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    folders = os.listdir(os.path.join(subset_dir))

    if not os.path.exists(os.path.join(destination_dir, 'train')):
        os.makedirs(os.path.join(destination_dir, 'train'))

    i = 0
    for folder in folders:
        if os.path.isdir(os.path.join(subset_dir, folder)):
            filenames = os.listdir(os.path.join(subset_dir, folder))
            for fname in filenames:
                if i % 500 == 0:
                    print('Saving data: {:6d}...'.format(i))

                img_dir = os.path.join(subset_dir, folder, fname)
                image = imread(img_dir)
                imsave(os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, 'jpg')), image, quality=100)
                if not copy:
                    os.remove(img_dir)
                i += 1


if __name__ == '__main__':
    subset_dir = "/mnt/D/Dropbox/Project/Python/datasets/ffhq_thumbnails"
    destination_dir = "/mnt/D/Dropbox/Project/Python/tfdatasets/ffhq_thumbnails"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = ('fake', 'real')

    image_dirs, label_dirs = sf.read_subset_cls(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
