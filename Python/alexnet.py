# Author: Matt Williams
# Version: 5/5/2022 


import tensorflow as tf
from util import *
from load_dataset import load_images_from_dataset_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from save_results import save_results
import matplotlib.pyplot as plt
import os 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy


def define_alexnet_model():
    """Defines the Edited AlexNet model for classification"""
    
    model = Sequential()

    #CNN specific layers
    model.add(Conv2D(filters = 96, kernel_size = (15,15), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (CLASSIFY_INPUT_HEIGHT,CLASSIFY_INPUT_WIDTH,1), padding = "valid", strides = 5))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 256, kernel_size = (5,5), activation = "relu", kernel_initializer ='glorot_normal', \
         input_shape = (37,37,96), padding = "same", strides = 1))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 384, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (18,18,256), padding = "same", strides = 1))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 384, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (8,8,384), padding = "same", strides = 1,))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (8,8,384), padding = "same", strides = 1))

    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))
    
    model.add(Dropout(rate = 0.5))
    
    model.add(Flatten())

    #fully connected layer
    model.add(Dense(units = 4096, activation = "relu", kernel_initializer = "glorot_normal"))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units = 512, activation = "relu", kernel_initializer = "glorot_normal"))
    model.add(Dense(units = 64,activation = "relu", kernel_initializer = "glorot_normal"))
    model.add(Dense(units = N_CLASSES, activation = "softmax", kernel_initializer = "glorot_normal"))


    opt = SGD(learning_rate = 0.001, momentum = 0.9)
    loss = CategoricalCrossentropy()
    model.compile(optimizer = opt, loss = loss) 

    return model


def train_alexnet():
    '''Train AlexNet on Training and Validation sets. Saves results on test set and saves the model weights.'''
    _, mask_imgs, labels = load_images_from_dataset_csv()

    x_train, x_valid, y_train, y_valid = train_test_split(mask_imgs, labels, test_size=TEST_SIZE + VALID_SIZE, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, random_state=42)


    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)

    model = define_alexnet_model()
    model.summary()

    stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5, restore_best_weights = True)

    training_hist = model.fit(x_train, y_train, epochs = N_EPOCHS, batch_size = CLASS_BATCH_SIZE, \
       validation_data = (x_valid, y_valid), verbose = 1, callbacks = [stop])

    file_name = "AlexNet_Backbone_{}.h5".format(len(os.listdir(MODELS_DATA_PATH)) + 1)
    model.save(os.path.join(MODELS_DATA_PATH, file_name))

    plt.plot(training_hist.history["val_loss"], label="Validation")
    plt.plot(training_hist.history["loss"], label = "Training")
    plt.legend()
    plt.show()

    predictions = model.predict(x_test, batch_size=CLASS_BATCH_SIZE)
    predictions = np.argmax(predictions, axis = -1)
    class_report = classification_report(y_test, predictions, output_dict=True)
    class_report["Model"] = "AlexNet"

    save_results(class_report)


    
def load_alexnet_model(): 
    '''Load a new AlexNet model with saved weights from previous training sessions.'''
    file_path = os.path.join(MODELS_DATA_PATH, "AlexNet_Backbone_3.h5")

    model = define_alexnet_model()
    model.load_weights(file_path)

    return model
    



if __name__ == "__main__": 
    
    train_alexnet()

    #model = load_alexnet_model()

