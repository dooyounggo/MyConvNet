import os
import shutil
import numpy as np
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.io import imsave

"""
Oxford-IIIT Pet dataset for two-class segmentation
https://research.sualab.com/practice/2018/11/23/image-segmentation-deep-learning.html
"""


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    train_images = os.listdir(os.path.join(subset_dir, 'train2017'))
    train_labels = os.listdir(os.path.join(subset_dir, 'stuffthingmaps_trainval2017', 'train2017'))
    val_images = os.listdir(os.path.join(subset_dir, 'val2017'))
    val_labels = os.listdir(os.path.join(subset_dir, 'stuffthingmaps_trainval2017', 'val2017'))

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

    image_dirs, label_dirs = sf.read_subset_seg(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names


def read_subset_old(subset_dir, sample_size=None, shuffle=True, image_size=(512, 512)):
    filenames = os.listdir(os.path.join(subset_dir, 'images'))
    class_names = ('background', 'cat', 'dog')

    num_classes = len(class_names)
    set_size = len(filenames)
    if sample_size is not None and sample_size < set_size:
        if shuffle:
            idx = np.random.choice(np.arange(set_size), size=sample_size, replace=False).astype(int)
        else:
            idx = np.arange(sample_size).astype(int)
        set_size = sample_size
    else:
        idx = np.arange(set_size).astype(int)
        if shuffle:
            np.random.shuffle(idx)

    X_set = np.empty((set_size, image_size[0], image_size[1], 3), dtype=np.float32)
    Y_set = np.empty((set_size, image_size[0], image_size[1], num_classes), dtype=np.bool)
    for i in range(set_size):
        if i % 200 == 0:
            print('Reading subset data: {:6d}/{}...'.format(i, set_size))

        img_name = filenames[idx[i]].split('.')[0]
        img_path = os.path.join(subset_dir, 'images', '{}.jpg'.format(img_name))
        mask_path = os.path.join(subset_dir, 'masks', '{}.png'.format(img_name))

        img = imread(img_path)
        if img.shape[-1] == 4:
            img = img[:, :, 0:3]
        if len(img.shape) == 2:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            img = np.tile(img, (1, 1, 3))
        img = sf.resize_fit_expand(img, (image_size[0], image_size[1]), mode='constant', anti_aliasing=True)

        mask = imread(mask_path)
        label = np.zeros(list(mask.shape[0:2]) + [num_classes], dtype=np.uint8)

        bkgd_idx = np.where(mask == 2)
        label[bkgd_idx[0], bkgd_idx[1], :] = [1, 0, 0]  # Background
        fore_idx = np.where(mask == 1)  # Foreground
        if img_name[0].isupper():
            label[fore_idx[0], fore_idx[1], :] = [0, 1, 0]  # Cat
        else:
            label[fore_idx[0], fore_idx[1], :] = [0, 0, 1]  # Dog

        label = sf.resize_fit_expand(label, (image_size[0], image_size[1]),
                                     order=0, mode='constant', anti_aliasing=False).astype(np.float32)
        X_set[i] = img
        Y_set[i] = label

    print('\nDone')

    print('Data stats:')
    print(X_set.shape)
    print(X_set.min(), X_set.max())
    print(class_names)

    return X_set, Y_set, class_names
