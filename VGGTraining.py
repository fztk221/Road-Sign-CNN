from keras import Input, Model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import pickle_data as p

# Parameters
num_of_classes = 43
batch_size_val = 50  # how many to process together
epochs_val = 15
dataset_path = 'datasets/VGG/'  # path to location to load datasets from
model_name = "vgg16_model3"

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

steps_per_epoch_val = 2000

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
X_validation = preprocess_input(X_validation)

new_input = Input(shape=(32, 32, 3))
model = VGG16(include_top=False, input_tensor=new_input)
# mark loaded layers as not trainable
for layer in model.layers[:17]:
    layer.trainable = False
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)
print(model.summary())
flat1 = Flatten()(model.output)
class1 = Dense(256, activation='relu')(flat1)
output = Dense(43, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# Augmenting images
dataGen = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,  # 0.2 Means can go from 0.8-> 1.2
                             shear_range=0.2,  # Magnitude of shear angle
                             rotation_range=30)  # Degree max rotation
dataGen.fit(X_train)

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    epochs=epochs_val, steps_per_epoch=steps_per_epoch_val,
                    validation_data=(X_validation, y_validation))

# store model and history in pickle files for later use
p.pickle_model(model_name, model)
p.pickle_model_history(model_name, history)

