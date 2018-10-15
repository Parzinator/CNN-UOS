import pickle

import tensorflow as tf
import numpy as np

CIFAR_PATH      = "CIFAR/cifar-10-batches-py"
NUMBER_BATCHES  = 5
LABEL_NAMES     = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_batch(filename):
    # load raw data using pickle
    with open(filename, mode = "rb") as file:
        batch = pickle.load(file, encoding = "latin1")

    # reshape row vectors into images x colors x rows x columns
    features = batch["data"].reshape((len(batch["data"]), 3, 32, 32))

    # transpose so that color channel comes after rows and cols
    features = features.transpose(0, 2, 3, 1)

    labels = batch["labels"]

    return features, labels

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)

    array = (array - min_val) / (max_val - min_val)

    return array

def one_hot_encode(array):
    encoded = np.zeros((len(array), 10))

    for idx, val in enumerate(array):
        encoded[idx][val] = 1

    return encoded

def get_training_data():
    """returns CIFAR training data prepared for use in tensorflow"""

    total_features  = []
    total_labels    = []

    for i in range(NUMBER_BATCHES):
        features, labels = load_batch(CIFAR_PATH + "/data_batch_" + str(i + 1))

        features = normalize(features)
        labels = one_hot_encode(labels)

        total_features.extend(features)
        total_labels.extend(labels)

    return np.array(features), np.array(labels)

def get_test_data():
    """returns CIFAR test data prepared for use in tensorflow"""

    features, labels = load_batch(CIFAR_PATH + "/test_batch")

    return normalize(features), one_hot_encode(labels)
