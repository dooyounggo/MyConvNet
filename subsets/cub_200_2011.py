import os
import csv
import shutil
import numpy as np
import subsets.subset_functions as sf
from skimage.io import imread

"""
Caltech-UCSD Birds-200-2011 dataset for classification
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
"""


def save_as_tfdata(subset_dir, destination_dir, copy=True):
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

    split = []
    f = open(split_info, 'r')
    lines = f.readlines()
    for line in lines:
        split.append(int(line.rstrip().split(' ')[-1]))
    f.close()

    i = 0
    i_train = 0
    i_test = 0
    for folder in classes:
        filenames = os.listdir(os.path.join(subset_dir, 'images', folder))
        for fname in filenames:
            if i % 500 == 0:
                print('Saving subset data: {:6d}...'.format(i))

            img_dir = os.path.join(subset_dir, 'images', folder, fname)
            ext = img_dir.split('.')[-1]

            if split[i] == 1:
                if copy:
                    shutil.copy2(img_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i_train, ext)))
                else:
                    shutil.move(img_dir, os.path.join(destination_dir, 'train', '{:010d}.{}'.format(i_train, ext)))
                f = open(os.path.join(destination_dir, 'train', '{:010d}.csv'.format(i_train)),
                         'w', encoding='utf-8', newline='')
                wrt = csv.writer(f)
                wrt.writerow([str(label[i])])
                f.close()
                i_train += 1
            else:
                if copy:
                    shutil.copy2(img_dir, os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i_test, ext)))
                else:
                    shutil.move(img_dir, os.path.join(destination_dir, 'test', '{:010d}.{}'.format(i_test, ext)))
                f = open(os.path.join(destination_dir, 'test', '{:010d}.csv'.format(i_test)),
                         'w', encoding='utf-8', newline='')
                wrt = csv.writer(f)
                wrt.writerow([str(label[i])])
                f.close()
                i_test += 1
            i += 1

    print('\nDone')


if __name__ == '__main__':
    subset_dir = "D:/Dropbox/Project/Python/datasets/CUB_200_2011"
    destination_dir = "D:/Dropbox/Project/Python/tfdatasets/CUB_200_2011"
    save_as_tfdata(subset_dir, destination_dir, copy=True)


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
