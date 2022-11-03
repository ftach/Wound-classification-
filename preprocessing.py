# -*- coding: utf-8 -*-
'''Module which contains all the sub programs used to get the images prepared before training.
'''

import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
import pickle 
import json
from os import listdir, makedirs
from os.path import isfile, join, isdir
from scipy.special import comb 
from itertools import combinations
import random 
import tensorflow 
from skimage import transform



def find_label(img_name):
    '''Find the label's number according to the name of the image. 

    Parameters
    ----------
    img_name: str 
        Name of the image in its directory

    Returns number associated to the label:
        Necrotic: 0
        Infectious: 1
        Fibrous: 2
        Buding: 3
        Epithelial: 4
    '''

    if img_name[0]=='e' and img_name[1]=='p':
        return 4
    elif img_name[0]=='b':
        return 3
    elif img_name[0]=='f':
        return 2    
    elif img_name[0]=='i':
        return 1
    elif img_name[0]=='n':
        return 0
    else:
        return 404


def save_in_json(data, path_to_save):
    '''Save data in json file. 
    
    Parameters
    ----------
    data: np.array
        Data to save.
    path_to_save: str
        Path where the data will be saved.
    
    Returns None'''

    with open(path_to_save, 'w') as f:
        json.dump(data, f)

def save_in_pickle(data, path_to_save):
    '''Save data in pickle file. 
    
    Parameters
    ----------
    data: np.array
        Data to save.
    path_to_save: str
        Path where the data will be saved.
    
    Returns None'''

    with open(path_to_save, 'wb') as f:
        pickle.dump(data, f)
    
def load_pkl_file(pkl_file_path):
    '''Load image tensor from pickle file.
    
    Parameters
    ----------
    pkl_file_path: str   
        Name of the file where the tensor of the images is stored.

    Returns np.array tensor 
    '''

    with open(pkl_file_path, "rb") as pkl_file:
        img_tensor = pickle.load(pkl_file)

    return img_tensor

def load_json_file(json_path):
    '''Load image tensor from pickle file.
    
    Parameters
    ----------
    pkl_file_path: str   
        Name of the file where the tensor of the images is stored.

    Returns dict labels  
    '''

    with open(json_path, "r") as json_file:
        labels = json.load(json_file)

    return labels

def disp_img_from_array(img_array):
    '''Display grayscale image from array. '''
    img = Image.fromarray(img_array)
    img = img.convert("L")
    
    img.show()    
    
    
def save_img_from_array(rf, path_to_save):
    '''Save grayscale image from array. '''
    img = Image.fromarray(rf)
    img = img.convert("L")

    img.save(path_to_save)

def create_dataset(img_directory, img_height, img_width, tensor_path, label_path, data_augmentation):
    '''Loads images from the directory and saves the tensor in a pickle file. 
    It also creates a json file with a dictionary of the labels of the images and their index in the tensor.
    
    Parameters
    ----------
    img_directory: str   
        Name of the directory where original images are stored.
    img_height: int
        Image height
    img_width: int 
        Image width
    tensor_path: str
        Path where the pickle file of the tensor will be stored
    label_path: str
        Path where the json file containing the dictionary of the labels will be stored
        The dictionary looks like {'label_nbr1': [img_index1, ..., img_indexn], ..., 'label_nbrn':[...]}
    data_augmentation: bool
        Needs data augmentation or not. False if validation dataset. 

    Returns None
    '''
    img_labels = {}
    i = 0
    #tensor = np.ones((nbr_of_img, img_height, img_width, 3))
    tensor = np.ones((0, img_height, img_width, 3))
    all_filenames = [f for f in listdir(img_directory) if isfile(join(img_directory, f))]
    for filename in all_filenames:
        path = img_directory + "/" + filename # Get image path

        # Open image and Add image to the tensor
        img_array = np.array(Image.open(path).convert("RGB"))
        tensor = np.append(tensor, np.reshape(img_array, (1, img_height, img_width, 3)), axis=0) 
        
        
        label = find_label(filename) # Get label

        if label not in img_labels.keys():
            img_labels[label] = []

        img_labels[label].append(i) # Add label to label dictionary 
        
        if data_augmentation==True:
            tensor = np.append(tensor, np.reshape(random_rotation(img_array).astype(np.uint8), (1, img_height, img_width, 3)), axis=0)
            tensor = np.append(tensor, np.reshape(np.fliplr(img_array).astype(np.uint8), (1, img_height, img_width, 3)), axis=0)
            img_labels[label].append(i)
            img_labels[label].append(i)
        i+=1 

    save_in_json(img_labels, label_path)
    print("{} images loaded".format(nbr_of_img))
    save_in_pickle(tensor, tensor_path)
    print("A {} Tensor has been saved in {}".format(tensor.shape, tensor_path))

    
def random_rotation(x, max_angle=8):
    '''Applies random rotation to the ndarray image for data augmentation. 
    
    Parameters
    ----------
    x: ndarray   
        Image array to be rotated
    max_angle: int 
        maximum used as upper boundary for the random choice of the angle

    Returns 
        ndarray Image rotated 
    '''
    angle = np.random.randint(2, max_angle) # rotation de +- 2° minimum, 8° maximum
    sens = np.random.randint(2)
    if sens==0: # on introduit une rotation dans le sens anti-horaire aléatoirement
        angle *= -1
   
    #x_rotated = x_rotated*255 à utiliser que si on part de photos
    # on multiplie pas y par 255 car on veut que ca reste entre 0 et 1
    x = transform.rotate(x, angle)
    x = transform.resize(x, (105, 105))

    return x.astype(np.uint8)
    
def create_data_augmentation(x_dataset):
    '''Appends rotated and flipped image in the dataset for data augmentation. 
    
    Parameters
    ----------
    x_dataset: ndarray   
        Image array dataset to be augmentated

    Returns 
        ndarray Image array dataset augmentated
    '''
    img_nbr = x_dataset.shape[0]
    img_height = x_dataset.shape[1]
    img_width = x_dataset.shape[2]
    img_channels = x_dataset.shape[3]

    aug_data = np.ones((0, img_height, img_width, img_channels))

    for i in range(img_nbr):
        x = x_dataset[i]
        aug_data = np.append(aug_data, np.reshape(x.astype(np.uint8), (1, img_height, img_width, img_channels)), axis=0)
        aug_data = np.append(aug_data, np.reshape(random_rotation(x).astype(np.uint8), (1, img_height, img_width, img_channels)), axis=0)
        aug_data = np.append(aug_data, np.reshape(np.fliplr(x).astype(np.uint8), (1, img_height, img_width, img_channels)), axis=0)

    return aug_data

def show_augmentated_img():
    with open("OASBUDdata/x_training_dataset/subj66_rf1.pkl", "rb") as pkl_file:
        x = pickle.load(pkl_file)
    with open("OASBUDdata/y_training_dataset/subj66.pkl", "rb") as pkl_file:
        y = pickle.load(pkl_file)
    
    (x, y) = random_rotation(x, y)
    x = Image.fromarray(np.reshape(x, (160, 160)))
    x = x.convert("L")
    x.save("OASBUDdata/x_training_dataset/subj66.jpg")
    y = Image.fromarray(np.reshape(y, (160, 160)))
    y = y.convert("L")
    y.save("OASBUDdata/y_training_dataset/subj66.jpg")



def create_pairs(tensor_path, label_path):
    '''Load image tensor from pickle file and create dataset with random pairs of images and their label. 
    Receives tensor already divided in training, validation or testing tensor. 
    
    Parameters
    ----------
    tensor_path: str
        Path where the pickle file of the tensor will be stored
    label_path: str
        Path where the json file containing the dictionary of the labels will be stored

    Returns tuple of np.array (pairs, labels)
    pairs looks like: [[img10, img20], [img11, img21], ..., [img1n, img2n]]
    labels looks like: [1, 0, ..., 0] with 0 meaning both images have a different label and 1 the same label 
    '''

    tensor = load_pkl_file(tensor_path) # load tensor 
    labels = load_json_file(label_path) # load dictionary of labels
    img_height = tensor.shape[1]
    img_width = tensor.shape[2]
    img_channels = tensor.shape[3]
    pairs = np.ones((0, 2, img_height, img_width, img_channels))
    binary_labels = np.ones((0))
    for label in labels.keys():
    # add matching images to pairs
        possible_combs = list(combinations(labels[label], 2)) # create list of possible combinations  
        # create each pair of those combinations 
        for i in range(len(possible_combs)):

            pair = np.ones((0, img_height, img_width, img_channels))
            pair = np.append(pair, np.reshape(tensor[possible_combs[i][0]], (1, img_height, img_width, img_channels)), axis=0)
            pair = np.append(pair, np.reshape(tensor[possible_combs[i][1]], (1, img_height, img_width, img_channels)), axis=0)
            pairs = np.append(pairs, np.reshape(pair, (1, 2, img_height, img_width, img_channels)), axis=0)
            binary_labels = np.append(binary_labels, np.reshape(1.0, (1)), axis=0)

    # add non matching images (random choice)
        # get rid of images from this label to avoid to create pairs of images with the same label
        random_label_dict = dict(labels) # copy the dict because pop modifies it 
        random_label_dict.pop(label, 0) 
        # create the pair of images
        
        for index in labels[label]:  
            pair = np.ones((0, img_height, img_width, img_channels))
            pair = np.append(pair, np.reshape(tensor[index], (1, img_height, img_width, img_channels)), axis=0)

            # add random image from another label 
            random_label = random.choice(list(random_label_dict.keys())) # choose random label
            random_img_index = random.choice(random_label_dict[random_label]) # choose random img in this label list 
            pair = np.append(pair, np.reshape(tensor[random_img_index], (1, img_height, img_width, img_channels)), axis=0)
            pairs = np.append(pairs, np.reshape(pair, (1, 2, img_height, img_width, img_channels)), axis=0)
            binary_labels = np.append(binary_labels, np.reshape(0.0, (1)), axis=0)

    return pairs, binary_labels
    
# Later we will create a function to find de Region of Interest (ROI) maybe with a filter that detect the wound and another one to crop the images.
# We can imagine to create a function to make and histogram equalization of the images.
# We can also imagine to create a function that find the right number of epochs according to the convergence of the model. 