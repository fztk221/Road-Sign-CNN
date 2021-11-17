from CNN import cnn_model
import talos as ta
import pickle_data as p
from tensorflow.keras.optimizers import Nadam, Adam

# Parameters
dataset_path = 'datasets/modelTraining/'  # path to location to load datasets from
scan_name = "model_trained_scan"

# loading in Data
file_names = ["X_train.p", "X_test.p", "X_validation.p", "y_train.p", "y_test.p", "y_validation.p"]
data_sets = p.load_in_pickles(file_names, dataset_path)
# Storing the Datasets into correctly name variables for ease of reading
X_train = data_sets[0]
X_test = data_sets[1]
X_validation = data_sets[2]
y_train = data_sets[3]
y_test = data_sets[4]
y_validation = data_sets[5]

# Hyper Parameter values to be tested
par = {'activation': ['relu', 'elu'],
     'dropout': [0, 0.25, 0.5],
     'nodes': [250, 500],
     'last_activation': ['sigmoid', 'softmax'],
     'optimizer': [Adam, Nadam],
     'lr': [0.001, 0.0001, 0.00001],
     'losses': ['categorical_crossentropy', 'sparse_categorical_crossentropy'],
     'steps_per_epoch': [1000, 2000, 2300],
     'epochs': [10, 15, 20],
     'batch_size': [10, 20, 50]}

# scanning
t = ta.Scan(x=X_train,
            y=y_train,
            x_val=X_validation,
            y_val=y_validation,
            model=cnn_model,
            params=par,
            experiment_name='AlexNet model Scan')

# store model and history in pickle files for later use
p.pickle_scan(scan_name, t)
