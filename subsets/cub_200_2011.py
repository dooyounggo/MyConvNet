"""
Caltech-UCSD Birds-200-2011 dataset for classification.
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
"""

import os
import csv
import shutil
import argparse
import numpy as np
import subsets.subset_functions as sf


def save_as_tfdata(subset_dir, destination_dir, copy=True, shuffle=False):
    class_info = os.path.join(subset_dir, 'classes.txt')
    label_info = os.path.join(subset_dir, 'image_class_labels.txt')
    split_info = os.path.join(subset_dir, 'train_test_split.txt')

    classes = []
    f = open(class_info, 'r')
    lines = f.readlines()
    for line in lines:
        classes.append(line.rstrip().split(' ')[-1])
    f.close()
    print(classes)

    label = []
    f = open(label_info, 'r')
    lines = f.readlines()
    for line in lines:
        label.append(int(line.rstrip().split(' ')[-1]) - 1)
    f.close()

    splits = []
    f = open(split_info, 'r')
    lines = f.readlines()
    num_train = 0
    num_test = 0
    for line in lines:
        is_training = int(line.rstrip().split(' ')[-1])
        splits.append(is_training)
        if is_training == 1:
            num_train += 1
        elif is_training == 0:
            num_test += 1
    f.close()

    if not os.path.exists(os.path.join(destination_dir, 'train')):
        os.makedirs(os.path.join(destination_dir, 'train'))
    if not os.path.exists(os.path.join(destination_dir, 'test')):
        os.makedirs(os.path.join(destination_dir, 'test'))

    i = 0
    i_train = 0
    i_test = 0
    idx_train = np.arange(num_train)
    idx_test = np.arange(num_test)
    if shuffle:
        np.random.shuffle(idx_train)
        # np.random.shuffle(idx_test)
    for folder in classes:
        filenames = os.listdir(os.path.join(subset_dir, 'images', folder))
        filenames.sort()
        for fname in filenames:
            if i % 500 == 0:
                print('Saving subset data: {:6d}/{}...'.format(i, num_train + num_test))

            img_dir = os.path.join(subset_dir, 'images', folder, fname)
            ext = img_dir.split('.')[-1]

            if splits[i] == 1:
                idx = int(idx_train[i_train])
                if copy:
                    shutil.copy2(img_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(idx, ext)))
                else:
                    shutil.move(img_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(idx, ext)))
                f = open(os.path.join(destination_dir, 'train', '{:010d}.csv'.format(idx)),
                         'w', encoding='utf-8', newline='')
                wrt = csv.writer(f)
                wrt.writerow([str(label[i])])
                f.close()
                i_train += 1
            else:
                idx = int(idx_test[i_test])
                if copy:
                    shutil.copy2(img_dir, os.path.join(destination_dir, 'test', '{:010d}.{}'.format(idx, ext)))
                else:
                    shutil.move(img_dir, os.path.join(destination_dir, 'test', '{:010d}.{}'.format(idx, ext)))
                f = open(os.path.join(destination_dir, 'test', '{:010d}.csv'.format(idx)),
                         'w', encoding='utf-8', newline='')
                wrt = csv.writer(f)
                wrt.writerow([str(label[i])])
                f.close()
                i_test += 1
            i += 1

    print('\nDone')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '--subset_dir', help='Path to original data', type=str,
                        default='./datasets/CUB_200_2011')
    parser.add_argument('--dest', '--destination_dir', help='Path to processed data', type=str,
                        default='./tfdatasets/cub_200_2011')
    parser.add_argument('--copy', help='Whether to copy images instead of moving them', type=str, default='True')
    parser.add_argument('--shuffle', help='Whether to shuffle training images', type=str, default='False')

    args = parser.parse_args()
    subset_dir = args.data
    destination_dir = args.dest
    copy = args.copy
    if copy.lower() == 'false' or copy == '0':
        copy = False
    else:
        copy = True
    shuffle = args.shuffle
    if shuffle.lower() == 'true' or shuffle == '1':
        shuffle = True
    else:
        shuffle = False

    print('\nPath to original data:  \"{}\"'.format(subset_dir))
    print('Path to processed data: \"{}\"'.format(destination_dir))
    print('copy = {}, shuffle = {}'.format(copy, shuffle))

    answer = input('\nDo you want to proceed? (Y/N): ')
    if answer.lower() == 'y' or answer.lower() == 'yes':
        save_as_tfdata(subset_dir, destination_dir, copy=copy, shuffle=shuffle)


def read_subset(subset_dir, shuffle=False, sample_size=None):
    class_names = (
        'Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani',
        'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet',
        'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird',
        'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting',
        'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat',
        'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant',
        'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper',
        'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo',
        'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker',
        'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher',
        'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird',
        'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch',
        'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe',
        'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak',
        'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull',
        'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull',
        'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird',
        'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger',
        'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco',
        'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher',
        'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake',
        'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark',
        'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk',
        'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole',
        'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican',
        'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit',
        'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven',
        'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike',
        'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow',
        'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow',
        'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow',
        'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow',
        'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow',
        'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow',
        'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager',
        'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern',
        'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee',
        'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo',
        'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo',
        'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler',
        'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler',
        'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler',
        'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler',
        'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler',
        'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler',
        'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush',
        'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker',
        'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker',
        'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren',
        'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat'
    )

    image_dirs, label_dirs = sf.read_subset_cls(subset_dir, shuffle=shuffle, sample_size=sample_size)

    return image_dirs, label_dirs, class_names
