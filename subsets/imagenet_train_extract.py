"""
A script for extracting training images from .tar files.
"""

import os
import tarfile


def train_extract(train_dir, remove_tar=True):
    files = os.listdir(train_dir)
    i = 0
    tar_list = []
    for f in files:
        if f.split('.')[-1].lower() == 'tar':
            tar_list.append(f)

    num_files = len(tar_list)
    for f in tar_list:
        if i % 50 == 0:
            print('Extracting tar files... {:4d}/{:,}'.format(i, num_files))

        fname = os.path.join(train_dir, f)
        tar = tarfile.open(fname, 'r:')
        tar.extractall(path=os.path.join(train_dir, fname.split('.')[0]))
        tar.close()
        if remove_tar:
            os.remove(fname)
        i += 1
    print('Extraction complete.')


if __name__ == '__main__':
    train_dir = 'D:/ILSVRC2012_img_train'
    remove_tar = True
    train_extract(train_dir, remove_tar)
