import pickle_data as p
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from preprocessing import preprocessing
'''Script for preprocessing data in gray scale'''

# Parameters
file_names = ['images.p', 'classNo.p']
path = 'datasets/'  # path to data files
dataset_path = 'datasets/modelTraining/'  # path of where to store dataset files
testRatio = 0.2  # percent of data split off for testing
validationRatio = 0.2  # percent of remaining data (after test split) for validation

# loading in data and determining number of classes
datasets = p.load_in_pickles(file_names, path)
images = datasets[0]
classNo = datasets[1]
num_of_classes = len(set(classNo))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
# split for train and test set
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
# split for validation set

X_train = np.array(list(map(preprocessing, X_train)))  # Iterates through and preprocesses all images
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Add Depth of 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# convert labels to categorical
y_train = to_categorical(y_train, num_of_classes)
y_validation = to_categorical(y_validation, num_of_classes)
y_test = to_categorical(y_test, num_of_classes)

# Storing the sets for later use
file_names_save = ["X_train.p", "X_test.p", "X_validation.p", "y_train.p", "y_test.p", "y_validation.p"]
datasets = [X_train, X_test, X_validation, y_train, y_test, y_validation]
p.save_pickles(file_names_save, datasets, dataset_path)
