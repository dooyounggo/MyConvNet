"""
Food-101 dataset for image classification.
https://www.vision.ee.ethz.ch/datasets_extra/food-101/
"""

import os
import csv
import shutil
import subsets.subset_functions as sf


def save_as_tfdata(subset_dir, destination_dir, copy=True):
    with open(os.path.join(subset_dir, 'meta', 'train.txt')) as f:
        train_filenames = f.readlines()
    with open(os.path.join(subset_dir, 'meta', 'test.txt')) as f:
        test_filenames = f.readlines()

    ext = 'jpg'

    if not os.path.exists(os.path.join(destination_dir, 'train')):
        os.makedirs(os.path.join(destination_dir, 'train'))
    if not os.path.exists(os.path.join(destination_dir, 'test')):
        os.makedirs(os.path.join(destination_dir, 'test'))

    set_size = len(train_filenames)
    class_names = []
    label = -1
    for i, fname in enumerate(train_filenames):
        if i % 200 == 0:
            print('Saving training data: {:6d}/{}...'.format(i, set_size))
        cname = fname.split('/')[0]
        if cname not in class_names:
            class_names.append(cname)
            label += 1

        img_dir = os.path.join(subset_dir, 'images', fname.rstrip() + '.' + ext)
        img_dest = os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i, ext))
        if not os.path.isfile(img_dest):
            if copy:
                shutil.copy2(img_dir, img_dest)
            else:
                shutil.move(img_dir, img_dest)
        label_dest = os.path.join(destination_dir, 'train', '{:010d}.csv'.format(i))
        if not os.path.isfile(label_dest):
            with open(label_dest, 'w', encoding='utf-8', newline='') as f:
                wrt = csv.writer(f)
                wrt.writerow([str(label)])

    set_size = len(test_filenames)
    for i, fname in enumerate(test_filenames):
        if i % 200 == 0:
            print('Saving testing data: {:6d}/{}...'.format(i, set_size))
        cname = fname.split('/')[0]
        if cname not in class_names:
            class_names.append(cname)
            label += 1

        img_dir = os.path.join(subset_dir, 'images', fname.rstrip() + '.' + ext)
        img_dest = os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i, ext))
        if not os.path.isfile(img_dest):
            if copy:
                shutil.copy2(img_dir, img_dest)
            else:
                shutil.move(img_dir, img_dest)
        label_dest = os.path.join(destination_dir, 'test', '{:010d}.csv'.format(i))
        if not os.path.isfile(label_dest):
            with open(label_dest, 'w', encoding='utf-8', newline='') as f:
                wrt = csv.writer(f)
                wrt.writerow([str(label)])

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
    subset_dir = "D:/Dropbox/Project/Python/datasets/food-101"
    destination_dir = "D:/Dropbox/Project/Python/tfdatasets/food-101"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = (
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio',
        'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
        'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad',
        'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche',
        'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
        'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros',
        'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee',
        'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',
        'dumplings', 'edamame', 'eggs_benedict', 'escargots',
        'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
        'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari',
        'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi',
        'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole',
        'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog',
        'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
        'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons',
        'miso_soup', 'mussels', 'nachos', 'omelette',
        'onion_rings', 'oysters', 'pad_thai', 'paella',
        'pancakes', 'panna_cotta', 'peking_duck', 'pho',
        'pizza', 'pork_chop', 'poutine', 'prime_rib',
        'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
        'risotto', 'samosa', 'sashimi', 'scallops',
        'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara',
        'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
        'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
        'waffles'
    )

    image_dirs, label_dirs = sf.read_subset_cls(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
