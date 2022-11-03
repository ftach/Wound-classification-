# -*- coding: utf-8 -*-
'''Fichier de test pour classify.py'''
import classify
import numpy as np

def test_create_pairs():
    classify.create_pairs("testing_tensor.pkl", "testing_labels.json")

def test_create_siamese_nw():
    classify.create_siamese_nw((105, 105, 3))

def main():
    #test_create_pairs()
    #test_create_siamese_nw()


if __name__ == '__main__':
    main()