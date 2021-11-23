from keras.applications.vgg16 import preprocess_input
import pickle_data as p
from CNN import vgg16_opt

# Parameters
num_of_classes = 43
dataset_path = 'datasets/VGG/'  # path to location to load datasets from
model_name = "vgg16_model_optimized"

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

history, model = vgg16_opt(X_train, y_train, X_validation, y_validation, X_train.shape[0])

# store model and history in pickle files for later use
model.save("models/" + model_name)
p.pickle_model_history(model_name, history.history)
