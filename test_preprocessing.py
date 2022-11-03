# -*- coding: utf-8 -*-
'''Fichier de test pour preprocessing.py'''

from cgi import test
from preprocessing import *

def test_find_label():
    assert find_label("n7.jpg") == 0
    assert find_label("ep4.png") == 4
    assert find_label("caca") == 404

def test_create_dataset():
    create_dataset("Dataset/croped_training", 6, 105, 105, "training_tensor.pkl", "training_labels.json")

def test_data_augmentation():
    x_data = load_pkl_file("training_tensor.pkl")
    aug_x_data = create_data_augmentation(x_data)
    rdm_index = random.randint(0, aug_x_data.shape[0])
    img = Image.fromarray(aug_x_data[5].astype(np.uint8), "RGB")
    img.show()

def main():
    print("find_label function test \n")
    test_find_label()
    print("Test OK")
    print("create_dataset function test \n")
    test_create_dataset()
    print("Test achieved. Go check it out. ")
    print("Data augmentation function test \n")
    test_data_augmentation()
    print("Test achieved. Go check it out. ")

if __name__ == '__main__':
    main()