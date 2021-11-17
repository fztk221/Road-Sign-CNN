from keras.preprocessing.image import ImageDataGenerator
from CNN import cnn
import pickle_data as p

# Parameters
num_of_classes = 43
batch_size_val = 50  # how many to process together
steps_per_epoch_val = 2000
epochs_val = 15
imageDimensions = (32, 32, 3)  # size of images
testRatio = 0.2  # percent of data split off for testing
validationRatio = 0.2  # percent of remaining data (after test split) for validation
dataset_path = 'datasets/modelTraining/'  # path to location to load datasets from
model_name = "my_model_trained"

# loading in Data
file_names = ["X_train.p", "X_test.p", "X_validation.p", "y_train.p", "y_test.p", "y_validation.p"]
data_sets=p.load_in_pickles(file_names,dataset_path)
# Storing the Datasets into correctly name variables for ease of reading
X_train = data_sets[0]
X_test = data_sets[1]
X_validation = data_sets[2]
y_train = data_sets[3]
y_test = data_sets[4]
y_validation = data_sets[5]

# Augment images
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,  # 0.2 Means can go from 0.8-> 1.2
                             shear_range=0.1,  # Magnitude of shear angle
                             rotation_range=10)  # Degree max rotation
dataGen.fit(X_train)

# Training
model = cnn(imageDimensions, num_of_classes)  # refer to CNN.py
print(model.summary())
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                              validation_data=(X_validation, y_validation))

# store model and history in pickle files for later use
p.pickle_model(model_name, model)
p.pickle_model_history(model_name, history)



