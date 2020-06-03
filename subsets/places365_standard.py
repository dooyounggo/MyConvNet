"""
Places365-standard dataset
http://places2.csail.mit.edu/download.html
"""

import os
import csv
import tarfile
import shutil
import argparse
import numpy as np
import subsets.subset_functions as sf


def save_as_tfdata(subset_dir, destination_dir, copy=True, shuffle=True, val_only=False):
    train_dir = os.path.join(subset_dir, 'train_large_places365standard')
    val_dir = os.path.join(subset_dir, 'val_large')
    test_dir = os.path.join(subset_dir, 'test_large')

    if not val_only:
        if not os.path.isdir(train_dir) and os.path.isfile(train_dir + '.tar'):
            train_tar = tarfile.open(train_dir + '.tar', 'r:')
            print('Extracting train_large_places365standard.tar ...')
            train_tar.extractall(path=subset_dir)
            print('Extraction complete.')
            train_tar.close()

        class_names = []
        full_filenames = []
        labels = []
        print('\nChecking training data...')
        sf.recursive_search([train_dir], full_filenames)
        for fname in full_filenames:
            sub_dir = fname.replace(train_dir, '')
            cname = '_'.join(sub_dir.split(os.sep)[2:-1])
            if cname not in class_names:
                class_names.append(cname)
            labels.append(class_names.index(cname))
        num_examples = len(full_filenames)
        assert num_examples == 1803460, 'The entire training images must be provided as whole' \
                                        ' (only {:,} images exist).'.format(num_examples)
        idx = np.arange(num_examples)
        if shuffle:
            np.random.shuffle(idx)  # Shuffle the files in advance since there are too many images in the dataset

        if not os.path.exists(os.path.join(destination_dir, 'train')):
            os.makedirs(os.path.join(destination_dir, 'train'))
        full_filenames = list(np.array(full_filenames)[idx])
        labels = list(np.array(labels)[idx])
        for i, (img_dir, label) in enumerate(zip(full_filenames, labels)):
            if i % 10000 == 0:
                print('Saving training data: {:8,}/{:,}...'.format(i, num_examples))

            ext = img_dir.split('.')[-1]
            if copy:
                shutil.copy2(img_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, ext)))
            else:
                shutil.move(img_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, ext)))
            f = open(os.path.join(destination_dir, 'train', '{:010d}.csv'.format(i)),
                     'w', encoding='utf-8', newline='')
            wrt = csv.writer(f)
            wrt.writerow([str(label)])
            f.close()

        print('(')
        for i, name in enumerate(class_names):
            print("'{}',".format(name), end='')
            if (i + 1) % 4 == 0:
                print('')
            else:
                print(' ', end='')
        print(')')

        if not os.path.isdir(test_dir) and os.path.isfile(test_dir + '.tar'):
            test_tar = tarfile.open(test_dir + '.tar', 'r:')
            print('Extracting test_large.tar ...')
            test_tar.extractall(path=subset_dir)
            print('Extraction complete.')
            test_tar.close()
        if not os.path.exists(os.path.join(destination_dir, 'test')):
            os.makedirs(os.path.join(destination_dir, 'test'))
        test_fnames = os.listdir(test_dir)
        test_fnames.sort()
        num_examples = len(test_fnames)
        assert num_examples == 328500, 'The entire test images must be provided as whole' \
                                       ' (only {:,} images exist).'.format(num_examples)
        for i, fname in enumerate(zip(test_fnames)):
            if i % 1000 == 0:
                print('Saving test data: {:8,}/{}...'.format(i, num_examples))

            img_dir = os.path.join(test_dir, fname)
            ext = img_dir.split('.')[-1]
            if copy:
                shutil.copy2(img_dir, os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i, ext)))
            else:
                shutil.move(img_dir, os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i, ext)))

    if not os.path.isdir(val_dir) and os.path.isfile(val_dir + '.tar'):
        val_tar = tarfile.open(val_dir + '.tar', 'r:')
        print('Extracting val_large.tar ...')
        val_tar.extractall(path=subset_dir)
        print('Extraction complete.')
        val_tar.close()
    if not os.path.exists(os.path.join(destination_dir, 'validation')):
        os.makedirs(os.path.join(destination_dir, 'validation'))
    val_fnames = os.listdir(val_dir)
    val_fnames.sort()
    num_examples = len(val_fnames)
    assert num_examples == 36500, 'The entire validation images must be provided as whole' \
                                  ' (only {:,} images exist).'.format(num_examples)
    with open(os.path.join(os.path.dirname(__file__), 'places365_val.txt')) as f:
        val_infos = f.readlines()
    for i, (fname, info) in enumerate(zip(val_fnames, val_infos)):
        if i % 1000 == 0:
            print('Saving validation data: {:8,}/{}...'.format(i, num_examples))

        img_dir = os.path.join(val_dir, fname)
        ext = img_dir.split('.')[-1]
        class_idx = info.rstrip().split(' ')[-1]

        if copy:
            shutil.copy2(img_dir, os.path.join(destination_dir, 'validation', '{:010d}.{}'.format(i, ext)))
        else:
            shutil.move(img_dir, os.path.join(destination_dir, 'validation', '{:010d}.{}'.format(i, ext)))
        f = open(os.path.join(destination_dir, 'validation', '{:010d}.csv'.format(i)),
                 'w', encoding='utf-8', newline='')
        wrt = csv.writer(f)
        wrt.writerow([str(class_idx)])
        f.close()

    print('\nDone')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '--subset_dir', help='Path to original data', type=str,
                        default='./datasets/Places365')
    parser.add_argument('--dest', '--destination_dir', help='Path to processed data', type=str,
                        default='./tfdatasets/places365')
    parser.add_argument('--copy', help='Whether to copy images instead of moving them', type=str, default='True')
    parser.add_argument('--shuffle', help='Whether to shuffle training images', type=str, default='True')
    parser.add_argument('--val_only', help='Whether to process validation images only', type=str, default='False')

    args = parser.parse_args()
    subset_dir = args.data
    destination_dir = args.dest
    copy = args.copy
    if copy.lower() == 'false' or copy == '0':
        copy = False
    else:
        copy = True
    shuffle = True  # shuffle=True for large datasets
    val_only = args.val_only
    if val_only.lower() == 'true' or val_only == '1':
        val_only = True
    else:
        val_only = False

    print('\nPath to original data:  \"{}\"'.format(subset_dir))
    print('Path to processed data: \"{}\"'.format(destination_dir))
    print('copy = {}, shuffle = {}, val_only = {}'.format(copy, shuffle, val_only))

    answer = input('\nDo you want to proceed? (Y/N): ')
    if answer.lower() == 'y' or answer.lower() == 'yes':
        save_as_tfdata(subset_dir, destination_dir, copy=copy, shuffle=shuffle, val_only=val_only)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = ()

    # Shuffling is disabled in order to minimize random file access
    image_dirs, label_dirs = sf.read_subset_cls(subset_dir, shuffle=False, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
