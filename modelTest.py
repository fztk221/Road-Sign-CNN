from plotting import plotting
import pickle_data as p

# Parameters
history_path = "models/history/"
model_name = "vgg16_model3"
data_path = "datasets/VGG/"
model_path = "models/"
file_names = [history_path + model_name+"_history.p", model_path + model_name+".p", data_path + "X_test2.p", data_path + "y_test2.p"]
datasets = p.load_in_pickles(file_names, "")
history = datasets[0]
model = datasets[1]
X_test = datasets[2]
y_test = datasets[3]

plotting(history.history, model, X_test, y_test)

