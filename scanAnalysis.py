import pickle_data as p
from keras.applications.vgg16 import preprocess_input
from CNN import vgg16_custom
from talos.utils.recover_best_model import recover_best_model

dataset_path = 'datasets/VGG/'  # path to location to load datasets from

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
results, models = recover_best_model(x_train=X_train,
                                     y_train=y_train,
                                     x_val=X_validation,
                                     y_val=y_validation,
                                     experiment_log='VGG16 model Scan/111721010700.csv',
                                     input_model=vgg16_custom,
                                     n_models=5,
                                     task='multi_label')
