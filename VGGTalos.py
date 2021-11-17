from keras.applications.vgg16 import preprocess_input
import pickle_data as p
import talos as ta
from CNN import vgg16_custom

# Parameters
num_of_classes = 43
dataset_path = 'datasets/VGG/'  # path to location to load datasets from
model_name = "vgg16_model3"
scan_name = "vgg16_model3_scan"

# loading in Data
file_names = ["X_train2.p", "X_test2.p", "X_validation2.p", "y_train2.p", "y_test2.p", "y_validation2.p"]
data_sets = p.load_in_pickles(file_names, dataset_path)
# Storing the Datasets into correctly name variables for ease of reading
X_train = data_sets[0]
X_test = data_sets[1]
X_validation = data_sets[2]
y_train = data_sets[3]
y_test = data_sets[4]
y_validation = data_sets[5]


X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
X_validation = preprocess_input(X_validation)

# Hyper Parameter values to be tested
par = {'dropout': [0, 0.2],
     'nodes': [250],
     'steps_per_epoch': [2000],
     'epochs': [30],
     'batch_size': [100]}

# scanning
t = ta.Scan(x=X_train,
            y=y_train,
            x_val=X_validation,
            y_val=y_validation,
            model=vgg16_custom,
            params=par,
            experiment_name='VGG16 model Scan')

# store model and history in pickle files for later use
p.pickle_scan(scan_name, t)

