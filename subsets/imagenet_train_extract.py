"""
A script for extracting training images from .tar files.
"""

import os
import tarfile


def train_extract(train_dir, remove_tar=True):
    files = os.listdir(train_dir)
    for i, f in enumerate(files):
        if i % 100 == 0:
            print('Extracting...\t{}/1,000'.format(i))

        fname = os.path.join(train_dir, f)

        tar = tarfile.open(fname, 'r:')
        tar.extractall(path=os.path.join(train_dir, fname.split('.')[0]))
        tar.close()

        if remove_tar:
            os.remove(fname)

    print('Extraction complete.')


if __name__ == '__main__':
    train_dir = './ILSVRC2012_img_train'
    remove_tar = True
    train_extract(train_dir, remove_tar)
