from plotting import plotting
from tensorflow import keras
import pickle_data as p

# Parameters
history_path = "models/history/"
data_path = "datasets/VGG/"
model_name = 'vgg16_model_optimized3'
model = keras.models.load_model('models/' + model_name)
file_names = [history_path + model_name + "_history.p", data_path + "X_test2.p", data_path + "y_test2.p"]
datasets = p.load_in_pickles(file_names, "")
history = datasets[0]
X_test = datasets[1]
y_test = datasets[2]

plotting(history, model, X_test, y_test)
