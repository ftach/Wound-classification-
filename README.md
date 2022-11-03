# Wound-classification-
Try of new CNN architectures to classify wounds according to their aspect

This project was considered before going to Togo for a volunteering project where would taught to locals how to treat wounds. 
I didn't have the knowledge to understand how could I make this converge and the dataset came from different pictures from different website as I didn't find an appropriate dataset. At least I trained to make UI and image processing. 

In this repository, you can find: 
- A UI that allows to load an image, zoom +/- and crop the image with a rectangle to select Regions Of Interest (ROI): roi_selector.py 
- An image processing module to load image, affect label, make data augmentation, divide the dataset into training, testing and validation datasets, etc: preprocessing.py
- A module to build CNN architecture, train and test the model: classify.py 

Hope to take another look at this project at the end of my studies. 
