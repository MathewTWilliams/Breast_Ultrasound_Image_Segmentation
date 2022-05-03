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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def define_backbone_model():

    model = Sequential()

    #CNN specific layers
    model.add(Conv2D(filters = 96, kernel_size = (15,15), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (TARGET_HEIGHT,TARGET_WIDTH,1), padding = "valid", strides = 5))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 256, kernel_size = (5,5), activation = "relu", kernel_initializer ='glorot_normal', \
         input_shape = (53,53,96), padding = "same", strides = 1))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 384, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (26,26,256), padding = "same", strides = 1))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Conv2D(filters = 384, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (26,26,384), padding = "same", strides = 1,))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", kernel_initializer ='glorot_normal', \
        input_shape = (26,26,384), padding = "same", strides = 1))
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
    model.compile(optimizer = opt, loss = "categorical_crossentropy") 

    return model


def train_backbone_cnn():
    _, mask_imgs, labels = load_images_from_dataset_csv()

    x_train, x_valid, y_train, y_valid = train_test_split(mask_imgs, labels, test_size=TEST_SIZE + VALID_SIZE, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, random_state=42)


    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)

    model = define_backbone_model()
    model.summary()

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

    file_name = "AlexNet_Backbone_{}.h5".format(len(os.listdir(MODELS_DATA_PATH)) + 1)
    model.save(os.path.join(MODELS_DATA_PATH, file_name))

    
def load_image_embedding_model(): 
    file_path = os.path.join(MODELS_DATA_PATH, "AlexNet_Backbone_4.h5")

    model = define_backbone_model()
    model.load_weights(file_path)
    



if __name__ == "__main__": 
    
    train_backbone_cnn()

    #model = load_image_embedding_model()