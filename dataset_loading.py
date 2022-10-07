import os
from urllib.request import urlretrieve
import zipfile
import tarfile
import numpy as np
import cv2 as cv
import json
np.random.seed(0)

tiny_imagenet_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

#download and extract tiny-imagenet
os.mkdir('datasets')
urlretrieve(tiny_imagenet_url, '/datasets/tiny-imagenet.zip')
zip_file = zipfile.ZipFile('datasets/tiny-imagenet.zip','r')
zip_file.extractall('datasets/')

# choose classes for training and ood detection
name_file = open('datasets/tiny-imagenet-200/wnids.txt')
all_names = name_file.read().split('\n')
np.random.shuffle(all_names)
in_distribution_class_names = all_names[:10]
out_of_distribution_class_names = all_names[10:20]

# create training dataset directory
os.mkdir('custom_datasets')



source = 'datasets/tiny-imagenet-200/train/'    
destination = 'custom_datasets/tiny-imagenet-training/'

try:
    os.mkdir(destination)

except:
    pass

for class_name in in_distribution_class_names:
    try:
        os.mkdir(destination + class_name)
    
        for img_name in os.listdir(source + class_name + '/images'):
            if img_name[-4:] != 'JPEG':
                continue
            image = cv.imread(source + class_name + '/images/' + img_name)
            image = cv.resize(image, (32, 32))
            cv.imwrite(destination + class_name + '/' + img_name, image)

    except:
        pass

# create validation set datasets from tiny imagenet

def get_validation_set(class_names, destination):
    annotations_file = open('datasets/tiny-imagenet-200/val/val_annotations.txt', 'r')
    label_map = [line.split('\t')[:2] for line in annotations_file.read().split('\n')]
    label_map = label_map[:-1]
    label_map = np.array(label_map)


    filtered_label_map = label_map[np.where([(i[1] in class_names) for i in label_map])[0]]

    

    source = 'datasets/tiny-imagenet-200/val/images/'
    try:
        os.mkdir(destination)
    except:
        pass

    for image_file_name, label in filtered_label_map:
        image = cv.imread(source + image_file_name)
        resized_image = cv.resize(image, (32, 32))
        try:
            os.mkdir(destination + label)
        except:
            pass
        cv.imwrite(destination + label + '/' + image_file_name, resized_image)

get_validation_set(in_distribution_class_names, 'custom_datasets/tiny-imagenet-val-in-distribution')
get_validation_set(out_of_distribution_class_names, 'custom_datasets/tiny-imagenet-val-out-of-distribution')

#download and extract imagenet v2

urlretrieve('https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz','datasets/imagenet-v2.tar.gz')
tarfile.TarFile('datasets/imagenet-v2.tar.gz','r').extractall('datasets')


# create validation datasets from imagenet v2


metadata = json.load(open('imagenetv2-metadata.json'))
label_map = [[k['cid'], k['wnid']]for k in metadata]
label_dict = {}

for i, j in label_map:
    label_dict[j] = i



def get_validation_set_from_imagenet_v2(classes, destination):
    source = 'datasets/imagenetv2-matched-frequency-format-val/'
    
    try:
        os.mkdir(destination)
    except:
        pass

    for name in classes:
        class_dir = source + str(label_dict[name])

        try:
            os.mkdir(destination + name)
        except:
            pass

        for image_path in os.listdir(class_dir):
            image = cv.imread(class_dir + '/' + image_path)
            image = cv.resize(image, (32, 32))
            cv.imwrite(destination + name + '/' + image_path, image)


get_validation_set_from_imagenet_v2(in_distribution_class_names, 'custom_datasets/imagenet-v2-val-in-distribution')





