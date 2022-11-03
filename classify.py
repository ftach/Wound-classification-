# -*- coding: utf-8 -*-
'''Module which contains all the sub programs used to classify the images.
'''
import tensorflow as tf

from tensorflow.keras.activations import sigmoid 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pickle 
from preprocessing import *
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

BS = 4
EPOCHS = 10

def euclidean_distance(vects:list):
    """Find the Euclidean distance between two vectors (sqrt(sum(square(t1-t2))). 

    Parameters
    ----------
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)

    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

def create_siamese_nw(input_shape):
    '''Loads ResNet50 and uses it to create a siamese network model. 
    
    Parameters
    ----------
    input_shape: tuple 
        (height, width, channels) of the img used in the siamese network 

    Returns:
        model: keras Model object 
    '''

    model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg') # load ResNet

    # create siamese input layers
    input1 = tf.keras.layers.Input(input_shape)
    input2 = tf.keras.layers.Input(input_shape)

    tower1 = model(input1)
    tower2 = model(input2)

    merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower1, tower2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)

    siamese = tf.keras.Model(inputs=[input1, input2], outputs=output_layer)

    return siamese 

def check_resnet():
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(105, 105, 3), pooling='avg')
    #model.summary()
    
    # load 2 img
    tensor = tf.load2img("testing_tensor.pkl")
    img1 = tensor[0]
    img1 = np.expand_dims(img1, axis=0)

    img2 = tensor[2]
    img2 = np.expand_dims(img2, axis=0)

    # make predictions for the 2 img
    pred1 = model.predict(img1)
    pred2 = model.predict(img2)

    pred1 = np.reshape(pred1, (2048))
    pred2 = np.reshape(pred2, (2048))

    #distance = euclidean_dist(pred1, pred2) # calculate the euclidean distance between the 2 img

    #distance = sigmoid_function(distance) # apply sigmoid function 
def divide_val_and_train_datasets(x, y, img_nbr=12):

    slice_index = int(x.shape[0]/2)
    x_train = x[:slice_index, :, :, :, :]
    x_val = x[slice_index:, :, :, :, :]
    y_train = y[:slice_index]
    y_val = y[slice_index:]

    return [[x_train, y_train], [x_val, y_val]]

def split_pairs(pairs):
    return [pairs[:,0], pairs[:,1]]

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()

def main():
    # check_resnet()
    # do again create dataset if the dataset changes
    #create_dataset("Dataset/croped_training", 105, 105, "training_tensor.pkl", "training_labels.json", True)
    #create_dataset("Dataset/croped_validation", 105, 105, "validation_tensor.pkl", "validation_labels.json", False)
    #x_data = load_pkl_file("training_tensor.pkl")
    #aug_x_data = create_data_augmentation(x_data)
    #save_in_pickle(aug_x_data, "aug_training_tensor.pkl")
    x_train, y_train = create_pairs("training_tensor.pkl", "training_labels.json") # should be saved to avoid to to do it again 
    x_val, y_val = create_pairs("validation_tensor.pkl", "validation_labels.json")
    #datasets = divide_val_and_train_datasets(x_pairs, y_true)
    ## reshape each image 
    siamese = create_siamese_nw((105, 105, 3)) # should be saved too
    ## siamese.summary()
    siamese.compile(loss=loss(), optimizer="Adam", metrics=["accuracy"]) # compiling with contrastive loss
    history = siamese.fit(split_pairs(x_train), y_train, 
        validation_data=(split_pairs(x_val), y_val), batch_size=BS, epochs=EPOCHS) # training
# Plot the accuracy
    #plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the constrastive loss
    #plt_metric(history=history.history, metric="loss", title="Constrastive Loss")
if __name__ == '__main__':
    main()