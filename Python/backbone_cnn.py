from time import monotonic
import tensorflow as tf
from util import *
from load_dataset import load_images_from_dataset_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from save_results import save_results
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def define_backbone_model():
    num_classes = 3

    model = Sequential()

    #CNN specific layers
    model.add(Conv2D(filters = 96, kernel_size = (15,15), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (TARGET_HEIGHT,TARGET_WIDTH,1), padding = "valid", strides = 5, kernel_regularizer = l2(5e-3)))

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 256, kernel_size = (5,5), activation = "relu", kernel_initializer ='glorot_normal', \
         input_shape = (53,53,96), padding = "same", strides = 1, kernel_regularizer = l2(5e-3)))

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 384, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (26,26,256), padding = "same", strides = 1, kernel_regularizer = l2(5e-3)))

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 384, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (26,26,384), padding = "same", strides = 1, kernel_regularizer = l2(5e-3)))

    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (26,26,384), padding = "same", strides = 1, kernel_regularizer = l2(5e-3)))

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))
    
    model.add(Dropout(rate = 0.5))
    
    model.add(Flatten())

    #fully connected layer
    model.add(Dense(units = 4096, activation = "relu", kernel_initializer = "glorot_normal"))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units = 512, activation = "relu", kernel_initializer = "glorot_normal"))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units = 64,activation = "relu", kernel_initializer = "glorot_normal"))
    model.add(Dense(units = num_classes, activation = "softmax", kernel_initializer = "glorot_normal"))


    opt = SGD(learning_rate = 0.01, momentum = 0.9)
    model.compile(optimizer = opt, loss = "categorical_crossentropy") 

    return model


if __name__ == "__main__": 
    
    base_imgs, _, labels = load_images_from_dataset_csv()

    x_train, x_valid, y_train, y_valid = train_test_split(base_imgs, labels, test_size=TEST_SIZE + VALID_SIZE)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5)


    y_train = to_categorical(y_train)
   # y_test = to_categorical(y_test)
    y_valid = to_categorical(y_valid)

    model = define_backbone_model()

    stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)

    training_hist = model.fit(x_train, y_train, epochs = N_EPOCHS, batch_size = BATCH_SIZE, \
       validation_data = (x_valid, y_valid), verbose = 1, callbacks = [stop])

    plt.plot(training_hist.history["val_loss"], label="Validation")
    plt.plot(training_hist.history["loss"], label = "Training")
    plt.legend()
    plt.show()

    predictions = model.predict(x_test, batch_size=BATCH_SIZE)
    predictions = np.argmax(predictions, axis = -1)
    class_report = classification_report(y_test, predictions, output_dict=True)
    class_report["Model"] = "AlexNet Backbone"

    save_results(class_report)


    
   
    