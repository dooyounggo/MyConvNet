"""
Stanford Dogs dataset for image classification.
http://vision.stanford.edu/aditya86/ImageNetDogs/
"""

import os
import csv
import shutil
import subsets.subset_functions as sf
from scipy.io import loadmat


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    train_info = os.path.join(subset_dir, 'train_list.mat')
    test_info = os.path.join(subset_dir, 'test_list.mat')

    train_dict = loadmat(train_info)
    test_dict = loadmat(test_info)

    train_images = train_dict['file_list']
    train_labels = train_dict['labels']
    test_images = test_dict['file_list']
    test_labels = test_dict['labels']

    set_size = len(train_images) + len(test_images)

    class_names = []

    if not os.path.exists(os.path.join(destination_dir, 'train')):
        os.makedirs(os.path.join(destination_dir, 'train'))

    i = 0
    for i_train, (fname, label) in enumerate(zip(train_images, train_labels)):
        if i % 500 == 0:
            print('Saving subset data: {:6d}/{}...'.format(i, set_size))

        fname = fname[0][0]
        label = label[0] - 1

        img_dir = os.path.join(subset_dir, 'images', fname)
        ext = img_dir.split('.')[-1]
        name = '-'.join(fname.split('-')[1:]).split('/')[0]
        if name not in class_names:
            class_names.append(name)

        if copy:
            shutil.copy2(img_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i_train, ext)))
        else:
            shutil.move(img_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i_train, ext)))
        f = open(os.path.join(destination_dir, 'train', '{:010d}.csv'.format(i_train)),
                 'w', encoding='utf-8', newline='')
        wrt = csv.writer(f)
        wrt.writerow([str(label)])
        f.close()
        i += 1

    if not os.path.exists(os.path.join(destination_dir, 'test')):
        os.makedirs(os.path.join(destination_dir, 'test'))

    for i_test, (fname, label) in enumerate(zip(test_images, test_labels)):
        if i % 500 == 0:
            print('Saving subset data: {:6d}/{}...'.format(i, set_size))

        fname = fname[0][0]
        label = label[0] - 1

        img_dir = os.path.join(subset_dir, 'images', fname)
        ext = img_dir.split('.')[-1]

        if copy:
            shutil.copy2(img_dir, os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i_test, ext)))
        else:
            shutil.move(img_dir, os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i_test, ext)))
        f = open(os.path.join(destination_dir, 'test', '{:010d}.csv'.format(i_test)),
                 'w', encoding='utf-8', newline='')
        wrt = csv.writer(f)
        wrt.writerow([str(label)])
        f.close()
        i += 1

    print('\nDone')

    print('(')
    for i, name in enumerate(class_names):
        print("'{}',".format(name), end='')
        if (i + 1) % 4 == 0:
            print('')
        else:
            print(' ', end='')
    print(')')


if __name__ == '__main__':
    subset_dir = "D:/Dropbox/Project/Python/datasets/stanford-dogs"
    destination_dir = "D:/Dropbox/Project/Python/tfdatasets/stanford-dogs"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = (
        'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese',
        'Shih-Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier',
        'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle',
        'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound',
        'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound',
        'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound',
        'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
        'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier',
        'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier',
        'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier',
        'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont',
        'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer',
        'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier',
        'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever',
        'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer',
        'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter',
        'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel',
        'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz',
        'schipperke', 'groenendael', 'malinois', 'briard',
        'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog',
        'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler',
        'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog',
        'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer',
        'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane',
        'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky',
        'affenpinscher', 'basenji', 'pug', 'Leonberg',
        'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian',
        'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke',
        'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle',
        'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog'
    )

    image_dirs, label_dirs = sf.read_subset_cls(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
