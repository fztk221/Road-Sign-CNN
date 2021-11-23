from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle_data as p

'''Script for Preprocessing data in RGB'''

# Parameters
testRatio = 0.2  # percent of data split off for testing
validationRatio = 0.2  # percent of remaining data (after test split) for validation
dataset_path = 'datasets/VGG/'  # path to location to save dataset
file_names = ['images.p', 'classNo.p']
path = 'datasets/'  # path to data files

# loading in data and determining number of classes
datasets = p.load_in_pickles(file_names, path)
images = datasets[0]
classNo = datasets[1]
num_of_classes= len(set(classNo))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
# split for train and test set
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
# split for validation set

# Add Depth of 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2],
                                    X_validation.shape[3])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3])

# convert labels to categorical
y_train = to_categorical(y_train, num_of_classes)
y_validation = to_categorical(y_validation, num_of_classes)
y_test = to_categorical(y_test, num_of_classes)

# Storing the sets for later use
file_names_save = ["X_train2.p", "X_test2.p", "X_validation2.p", "y_train2.p", "y_test2.p", "y_validation2.p"]
datasets = [X_train, X_test, X_validation, y_train, y_test, y_validation]
p.save_pickles(file_names_save, datasets, dataset_path)
