from keras import Input, Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from talos.model import lr_normalizer

"""File Defining all CNN models"""


def cnn(image_dimensions, num_of_classes):
    """Initial Lenet/AlexNet based CNN model without hyper-parameter optimziation"""
    num_of_filters = 60
    size_of_filter = (5, 5)  # note this removes 2 pixels from each border when applied to 32 by 32 images
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    num_of_nodes = 500
    model = Sequential([
        # 1st convolution block
        Conv2D(num_of_filters, size_of_filter, input_shape=(image_dimensions[0], image_dimensions[1], 1),
               activation='relu'),
        Conv2D(num_of_filters, size_of_filter, activation='relu'),
        MaxPooling2D(pool_size=size_of_pool),
        # 2nd convolution block
        Conv2D(num_of_filters // 2, size_of_filter2, activation='relu'),
        Conv2D(num_of_filters // 2, size_of_filter2, activation='relu'),
        MaxPooling2D(pool_size=size_of_pool),
        # ANN block
        Dropout(0.5),
        Flatten(),
        Dense(num_of_nodes, activation='relu'),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn_model(X_train, y_train, X_validation, y_validation, params):
    """Initial Lenet/AlexNet based CNN model with hyper-parameter optimization"""
    num_of_filters = 60
    size_of_filter = (5, 5)  # note this removes 2 pixels from each border when applied to 32 by 32 images
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    image_dimensions = [32, 32]
    num_of_classes = 43
    model = Sequential([
        # 1st convolution block
        Conv2D(num_of_filters, size_of_filter, input_shape=(image_dimensions[0], image_dimensions[1], 1),
               activation=params['activation']),
        Conv2D(num_of_filters, size_of_filter, activation=params['activation']),
        MaxPooling2D(pool_size=size_of_pool),
        # 2nd convolution block
        Conv2D(num_of_filters // 2, size_of_filter2, activation=params['activation']),
        Conv2D(num_of_filters // 2, size_of_filter2, activation=params['activation']),
        MaxPooling2D(pool_size=size_of_pool),
        # ANN block
        Dropout(params['dropout']),
        Flatten(),
        Dense(params['nodes'], activation=params['activation']),
        Dropout(params['dropout']),
        Dense(num_of_classes, activation=params['last_activation'])
    ])
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['losses'], metrics=['accuracy'])

    # Augment images
    data_gen = ImageDataGenerator(width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.2,  # 0.2 Means can go from 0.8-> 1.2
                                  shear_range=0.1,  # Magnitude of shear angle
                                  rotation_range=10)  # Degree max rotation

    history = model.fit_generator(data_gen.flow(X_train, y_train, batch_size=params['batch_size']),
                                  steps_per_epoch=params['steps_per_epoch'], epochs=params['epochs'],
                                  validation_data=(X_validation, y_validation))
    return history, model


def vgg16_custom(X_train, y_train, X_validation, y_validation, params):
    """Vgg16 transfer learned model with hyper param optimization"""
    new_input = Input(shape=(32, 32, 3))
    model = VGG16(include_top=False, input_tensor=new_input)
    # mark loaded layers as not trainable
    for layer in model.layers[:17]:
        layer.trainable = False
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    print(model.summary())
    flat1 = Flatten()(model.output)
    drop1 = Dropout(params['dropout'])(flat1)
    class1 = Dense(params['nodes'], activation='relu')(drop1)
    output = Dense(43, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    # Augmenting images
    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,  # 0.2 Means can go from 0.8-> 1.2
                                 shear_range=0.2,  # Magnitude of shear angle
                                 rotation_range=30)  # Degree max rotation
    dataGen.fit(X_train)

    history = model.fit(dataGen.flow(X_train, y_train, batch_size=params['batch_size']),
                        epochs=params['epochs'], steps_per_epoch=params['steps_per_epoch'],
                        validation_data=(X_validation, y_validation))

    return history, model
