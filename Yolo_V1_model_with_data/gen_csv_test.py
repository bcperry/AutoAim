import os
import random


def gen_data_csv(image_directory, label_directory, test_size = 10):
    '''
    file creates the needed .csv files to load data to be used in yolo v1
    Args:
        image_directory: (string) location of image files
        label_directory: (string) location of text files
        train_size: (int) percent of data to be held out for testing (default = 10)

    Returns:
        NONE
    '''
    image_dir = os.listdir(image_directory)
    label_dir = os.listdir(label_directory)
    data_dict = {};

    # read and match images to text labels
    for image in image_dir:
        for label in label_dir:
            if image[:-4] == label[:-4]:
                data_dict[image] = label

    # determine the dictionary size
    dict_size = len(data_dict)

    # create the test set
    test_data_size = round(test_size*dict_size/100)
    test_data_sample = random.sample(data_dict.items(), test_data_size)

    test_data = {}
    for entry in test_data_sample:
        test_data[entry[0]] = entry[1]

    # create the train set
    train_data = {}
    for key in data_dict.keys():
        if key in test_data.keys():
            continue
        else:
            train_data[key] = data_dict[key]

    # write the .csv files
    with open('train.csv', 'w') as f:
        for key in train_data.keys():
            f.write("%s,%s\n" % (key, train_data[key]))

    with open('test.csv', 'w') as f:
        for key in test_data.keys():
            f.write("%s,%s\n" % (key, test_data[key]))


IMG_DIR = "halo_data"
LABEL_DIR = "halo_data/labels"
gen_data_csv(IMG_DIR, LABEL_DIR, 10)
