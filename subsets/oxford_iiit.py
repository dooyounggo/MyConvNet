"""
Oxford-IIIT Pet dataset for classification (pre-processed).
https://research.sualab.com/practice/2018/11/23/image-segmentation-deep-learning.html
"""

import os
import csv
import shutil
import numpy as np
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.transform import resize


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    filenames = os.listdir(subset_dir)
    labels = []
    class_names = []
    for filename in filenames:
        filename = filename.split('_')[:-1]
        filename = '_'.join(filename)
        labels.append(filename)
        if filename not in class_names:
            class_names.append(filename)
    class_names.sort()
    class_names = tuple(class_names)
    print(class_names)

    set_size = len(filenames)

    for i in range(set_size):
        if i % 200 == 0:
            print('Saving subset data: {:6d}/{}...'.format(i, set_size))

        img_dir = os.path.join(subset_dir, filenames[i])
        ext = img_dir.split('.')[-1]
        if copy:
            shutil.copy2(img_dir, os.path.join(destination_dir, '{:010d}.{}'.format(i, ext)))
        else:
            shutil.move(img_dir, os.path.join(destination_dir, '{:010d}.{}'.format(i, ext)))
        label = class_names.index(labels[i])
        f = open(os.path.join(destination_dir, '{:010d}.csv'.format(i)), 'w', encoding='utf-8', newline='')
        wrt = csv.writer(f)
        wrt.writerow([str(label)])
        f.close()

    print('\nDone')


if __name__ == '__main__':
    subset_dir = "D:/Python/datasets/oxford-iiit-pet-dataset-sualab/test/images"
    destination_dir = "D:/Python/tfdatasets/oxford-iiit-pet-dataset-sualab/test"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = ('Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon',
                   'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog',
                   'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua',
                   'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
                   'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian',
                   'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
                   'wheaten_terrier', 'yorkshire_terrier')

    image_dirs, label_dirs = sf.read_subset_cls(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
