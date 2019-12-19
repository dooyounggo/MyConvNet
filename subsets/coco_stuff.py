"""
COCO-stuff dataset for image segmentation.
https://github.com/nightrome/cocostuff
"""

import os
import shutil
import numpy as np
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.io import imsave


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    train_images = os.listdir(os.path.join(subset_dir, 'train2017'))
    train_labels = os.listdir(os.path.join(subset_dir, 'stuffthingmaps_trainval2017', 'train2017'))
    val_images = os.listdir(os.path.join(subset_dir, 'val2017'))
    val_labels = os.listdir(os.path.join(subset_dir, 'stuffthingmaps_trainval2017', 'val2017'))

    if not os.path.exists(os.path.join(destination_dir, 'train')):
        os.makedirs(os.path.join(destination_dir, 'train'))

    i = 0
    for image, label in zip(train_images, train_labels):
        if i % 200 == 0:
            print('Saving training data: {:6d}...'.format(i))

        image_dir = os.path.join(subset_dir, 'train2017', image)
        image_ext = image_dir.split('.')[-1]
        if copy:
            shutil.copy2(image_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, image_ext)))
        else:
            shutil.move(image_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, image_ext)))

        label_dir = os.path.join(subset_dir, 'stuffthingmaps_trainval2017', 'train2017', label)
        label_ext = label_dir.split('.')[-1]

        mask = imread(label_dir).astype(np.uint8)
        label = mask + 2  # unlabeled: 255 -> 1 (= backgrounds)

        imsave(os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, label_ext)), label)
        if not copy:
            os.remove(label_dir)

        i += 1

    if not os.path.exists(os.path.join(destination_dir, 'validation')):
        os.makedirs(os.path.join(destination_dir, 'validation'))

    i = 0
    for image, label in zip(val_images, val_labels):
        if i % 200 == 0:
            print('Saving validation data: {:6d}...'.format(i))

        image_dir = os.path.join(subset_dir, 'val2017', image)
        image_ext = image_dir.split('.')[-1]
        if copy:
            shutil.copy2(image_dir, os.path.join(destination_dir, 'validation', '{:010d}.{}'.format(i, image_ext)))
        else:
            shutil.move(image_dir, os.path.join(destination_dir, 'validation', '{:010d}.{}'.format(i, image_ext)))

        label_dir = os.path.join(subset_dir, 'stuffthingmaps_trainval2017', 'val2017', label)
        label_ext = label_dir.split('.')[-1]

        mask = imread(label_dir).astype(np.uint8)
        label = mask + 2  # unlabeled: 255 -> 1 (= backgrounds)

        imsave(os.path.join(destination_dir, 'validation', '{:010d}.{}'.format(i, label_ext)), label)
        if not copy:
            os.remove(label_dir)

        i += 1

    print('\nDone')


if __name__ == '__main__':
    subset_dir = "F:/Python/datasets/coco-stuff"
    destination_dir = "F:/Python/tfdatasets/coco-stuff"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = (
        'unlabeled', 'person', 'bicycle', 'car',
        'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'plate', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
        'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'mirror', 'dining table', 'window',
        'desk', 'toilet', 'door', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'blender',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'hair brush', 'banner', 'blanket',
        'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard',
        'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence',
        'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other',
        'grass', 'gravel', 'ground-other', 'hill', 'house',
        'leaves', 'light', 'mat', 'metal', 'mirror-stuff',
        'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
        'platform', 'playingfield', 'railing', 'railroad', 'river',
        'road', 'rock', 'roof', 'rug', 'salad',
        'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw',
        'structural-other', 'table', 'tent', 'textile-other', 'towel',
        'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other',
        'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
        'waterdrops', 'window-blind', 'window-other', 'wood'
    )

    image_dirs, label_dirs = sf.read_subset_seg(subset_dir, shuffle=False, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
