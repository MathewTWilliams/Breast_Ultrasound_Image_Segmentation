# references: 
# - https://medium.com/analytics-vidhya/training-u-net-from-scratch-using-tensorflow2-0-fad541e2eaf1
# - https://arxiv.org/pdf/1505.04597.pdf

import tensorflow as tf
from util import *
from load_dataset import load_images_from_dataset_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from save_results import save_results
import matplotlib.pyplot as plt
import os 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Input, ZeroPadding2D
from tensorflow.keras.layers import Cropping2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def define_unet_model(): 
    
    #inputs
    inputs = Input(shape = (764, 764, 1))

    ####Encoders#####

    #Encoder 1
    cnv1 = Conv2D(filters = 64, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(inputs)
    bn1 = BatchNormalization()(cnv1)
    cnv2 = Conv2D(filters = 64, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(bn1) # This one
    bn2 = BatchNormalization()(cnv2)
    mp1 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn2)

    #Encoder 2
    cnv3 = Conv2D(filters = 128, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(mp1)
    bn3 = BatchNormalization()(cnv3)
    cnv4 = Conv2D(filters = 128, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(bn3) # This one
    bn4 = BatchNormalization()(cnv4)
    mp2 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn4)

    #Encoder 3
    cnv5 = Conv2D(filters = 256, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(mp2)
    bn5 = BatchNormalization()(cnv5)
    cnv6 = Conv2D(filters = 256, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(bn5) # This one
    bn6 = BatchNormalization()(cnv6)
    mp3 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn6)

    # Encoder 4
    cnv7 = Conv2D(filters = 512, activation = "relu", kernel_size = (3,3),  strides = (1,1))(mp3)
    bn7 = BatchNormalization()(cnv7)
    cnv8 = Conv2D(filters = 512, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(bn7) # This one
    bn8 = BatchNormalization()(cnv8)
    mp4 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn8)

    #bridge
    cnv9 = Conv2D(filters = 1024, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1))(mp4)
    bn9 = BatchNormalization()(cnv9)
    cnv10 = Conv2D(filters = 1024, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(bn9)
    bn10 = BatchNormalization()(cnv10)

    ####Decoders####

    #Decoder 1
    cnvt1 = Conv2DTranspose(filters = 512, activation = 'relu', kernel_size = (2,2), strides = (2,2), padding = "valid")(bn10)
    crop1 = Cropping2D(cropping=((4,4), (4,4)))(cnv8)
    con1 = Concatenate(axis = -1)([cnvt1, crop1])
    cnv11 = Conv2D(filters = 512, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1))(con1)
    bn11 = BatchNormalization()(cnv11)
    cnv12 = Conv2D(filters = 512, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1))(bn11)
    bn12 = BatchNormalization()(cnv12)

    #Decoder 2
    cnvt2 = Conv2DTranspose(filters = 256, activation = 'relu', kernel_size = (2,2), strides = (2,2), padding = "valid")(bn12)
    crop2 = Cropping2D(cropping = (16,16))(cnv6)
    con2 = Concatenate(axis = -1)([cnvt2, crop2])
    cnv13 = Conv2D(filters = 256, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1))(con2)
    bn13 = BatchNormalization()(cnv13)
    cnv14 = Conv2D(filters = 256, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1))(bn13)
    bn14 = BatchNormalization()(cnv14)

    #Decoder 3
    cnvt3 = Conv2DTranspose(filters = 128, activation = 'relu', kernel_size = (2,2), strides = (2,2), padding = "valid")(bn14)
    crop3 = Cropping2D(cropping = ((40,40),(40,40)))(cnv4)
    con3 = (Concatenate(axis = -1)([cnvt3, crop3]))
    cnv15 = Conv2D(filters = 128, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(con3)
    bn15 = BatchNormalization()(cnv15)
    cnv16 = Conv2D(filters = 128, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1))(bn15)
    bn16 = BatchNormalization()(cnv16)

    #Decoder 4
    cnvt4 = Conv2DTranspose(filters = 64, activation = 'relu', kernel_size = (2,2), strides = (2,2), padding = "valid")(bn16)
    crop4 = Cropping2D(cropping = (88,88))(cnv2)
    con4 = Concatenate(axis = -1)([cnvt4, crop4])
    cnv17 = Conv2D(filters = 64, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1))(con4)
    bn17 = BatchNormalization()(cnv17)
    cnv18 = Conv2D(filters = 64, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1))(bn17)
    bn18 = BatchNormalization()(cnv18)

    #output
    outputs = Conv2D(filters = 2, kernel_size = (1,1), padding = "valid", strides = (1,1), activation = "softmax")(bn18)

    model = Model(inputs = inputs, outputs = outputs, name = "U-net")

    opt = SGD(learning_rate = 0.01, momentum = 0.99)
    model.compile(optimizer = opt, loss = "categorical_crossentropy")

    return model


def train_unet_model(): 
    base_imgs, mask_imgs, _ = load_images_from_dataset_csv(segmentation=True)

    x_train, x_valid, y_train, y_valid = train_test_split(base_imgs, mask_imgs, test_size=TEST_SIZE + VALID_SIZE, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, random_state=42)

    model = define_unet_model()
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

    file_name = "Unet_{}.h5".format(len(os.listdir(MODELS_DATA_PATH)) + 1)
    model.save(os.path.join(MODELS_DATA_PATH, file_name))

def load_unet_model(): 
    pass


if __name__ == "__main__": 
    train_unet_model()