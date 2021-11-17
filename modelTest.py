from matplotlib import pyplot as plt
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

# plotting
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
