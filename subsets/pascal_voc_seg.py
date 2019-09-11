import os
import shutil
import numpy as np
import subsets.subset_functions as sf
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

"""
PASCAL VOC dataset for segmentation
http://host.robots.ox.ac.uk/pascal/VOC/
"""

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    label_names = os.listdir(os.path.join(subset_dir, 'SegmentationClass'))

    set_size = len(label_names)
    for i in range(set_size):
        if i % 200 == 0:
            print('Saving subset data: {:6d}/{}...'.format(i, set_size))

        filename = label_names[i].split('.')[0]

        image_dir = os.path.join(subset_dir, 'JPEGImages', filename + '.jpg')
        image_ext = image_dir.split('.')[-1]
        if copy:
            shutil.copy2(image_dir, os.path.join(destination_dir, '{:010d}.{}'.format(i, image_ext)))
        else:
            shutil.move(image_dir, os.path.join(destination_dir, '{:010d}.{}'.format(i, image_ext)))

        label_dir = os.path.join(subset_dir, 'SegmentationClass', label_names[i])
        label_ext = label_dir.split('.')[-1]

        mask = imread(label_dir)
        label = np.zeros(mask.shape[0:2], dtype=np.uint8)
        for n, color in enumerate(VOC_COLORMAP):
            idx = np.where((mask[:, :, 0] == color[0]) & (mask[:, :, 1] == color[1]) & (mask[:, :, 2] == color[2]))
            label[idx[0], idx[1]] = n + 1  # 0 for edge pixels, 1 for backgrounds

        imsave(os.path.join(destination_dir, '{:010d}.{}'.format(i, label_ext)), label)
        if not copy:
            os.remove(label_dir)

    print('\nDone')


if __name__ == '__main__':
    subset_dir = "D:/Dropbox/Project/Python/datasets/pascal-voc/train"  # FIXME
    destination_dir = "D:/Dropbox/Project/Python/tfdatasets/pascal-voc/train"  # FIXME
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'dining_table', 'dog', 'horse',
                   'motorbike', 'person', 'potted_plant', 'sheep', 'sofa', 'train', 'tv_monitor')

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
