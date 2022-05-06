# Author: Matt Williams
# Version: 5/5/2022
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
import math

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Input, ZeroPadding2D
from tensorflow.keras.layers import Cropping2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU

def calc_std_dev(kernel_x, kernel_y, filters):
    '''A method used to calculate the value of the standard deviation for a layer that is 
    initialized with the Gaussian Distribution based on the kernel size and number of filters
    coming in from the previous layer. The equation given by Unet paper.'''
    num = 2 / (kernel_x * kernel_y * filters)
    return math.sqrt(num)


def define_unet_model(): 
    '''Returns a model with the Unet architecture'''
    #inputs
    inputs = Input(shape = (SEGMENT_INPUT_HEIGHT, SEGMENT_INPUT_WIDTH, 1))

    ####Encoders#####

    #Encoder 1
    rn1 = RandomNormal(stddev = calc_std_dev(1,1,1))
    cnv1 = Conv2D(filters = 64, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn1)(inputs)
    bn1 = BatchNormalization()(cnv1)

    rn2 = RandomNormal(stddev = calc_std_dev(3,3,64))
    cnv2 = Conv2D(filters = 64, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn2)(bn1) # This one
    bn2 = BatchNormalization()(cnv2)
    mp1 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn2)

    #Encoder 2
    cnv3 = Conv2D(filters = 128, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn2)(mp1)
    bn3 = BatchNormalization()(cnv3)

    rn3 = RandomNormal(stddev = calc_std_dev(3,3,128))
    cnv4 = Conv2D(filters = 128, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn3)(bn3) # This one
    bn4 = BatchNormalization()(cnv4)
    mp2 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn4)

    #Encoder 3
    cnv5 = Conv2D(filters = 256, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn3)(mp2)
    bn5 = BatchNormalization()(cnv5)

    rn4 = RandomNormal(stddev = calc_std_dev(3,3,256))
    cnv6 = Conv2D(filters = 256, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn4)(bn5) # This one
    bn6 = BatchNormalization()(cnv6)
    mp3 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn6)

    # Encoder 4
    cnv7 = Conv2D(filters = 512, activation = "relu", kernel_size = (3,3),  strides = (1,1), kernel_initializer = rn4)(mp3)
    bn7 = BatchNormalization()(cnv7)
    
    rn5 = RandomNormal(stddev = calc_std_dev(3,3,512))
    cnv8 = Conv2D(filters = 512, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn5)(bn7) # This one
    bn8 = BatchNormalization()(cnv8)
    mp4 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn8)

    #bridge
    cnv9 = Conv2D(filters = 1024, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1), kernel_initializer = rn5)(mp4)
    bn9 = BatchNormalization()(cnv9)

    rn6 = RandomNormal(stddev = calc_std_dev(3,3,1024))
    cnv10 = Conv2D(filters = 1024, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn6)(bn9)
    bn10 = BatchNormalization()(cnv10)

    ####Decoders####

    #Decoder 1
    cnvt1 = Conv2DTranspose(filters = 512, activation = 'relu', kernel_size = (2,2), strides = (2,2), padding = "valid", kernel_initializer = rn6)(bn10)
    crop1 = Cropping2D(cropping=((4,4), (4,4)))(cnv8)
    con1 = Concatenate(axis = -1)([cnvt1, crop1])

    rn7 = RandomNormal(stddev = calc_std_dev(2,2,512))
    cnv11 = Conv2D(filters = 512, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1), kernel_initializer = rn7)(con1)
    bn11 = BatchNormalization()(cnv11)

    rn8 = RandomNormal(stddev = calc_std_dev(3,3,512))
    cnv12 = Conv2D(filters = 512, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1), kernel_initializer = rn8)(bn11)
    bn12 = BatchNormalization()(cnv12)

    #Decoder 2
    cnvt2 = Conv2DTranspose(filters = 256, activation = 'relu', kernel_size = (2,2), strides = (2,2), padding = "valid", kernel_initializer = rn8)(bn12)
    crop2 = Cropping2D(cropping = (16,16))(cnv6)
    con2 = Concatenate(axis = -1)([cnvt2, crop2])

    rn9 = RandomNormal(stddev = calc_std_dev(2,2,256))
    cnv13 = Conv2D(filters = 256, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1), kernel_initializer = rn9)(con2)
    bn13 = BatchNormalization()(cnv13)

    rn10 = RandomNormal(stddev = calc_std_dev(3,3,256))
    cnv14 = Conv2D(filters = 256, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1), kernel_initializer = rn10)(bn13)
    bn14 = BatchNormalization()(cnv14)

    #Decoder 3
    cnvt3 = Conv2DTranspose(filters = 128, activation = 'relu', kernel_size = (2,2), strides = (2,2), padding = "valid", kernel_initializer = rn10)(bn14)
    crop3 = Cropping2D(cropping = (40,40))(cnv4)
    con3 = (Concatenate(axis = -1)([cnvt3, crop3]))

    rn11 = RandomNormal(stddev = calc_std_dev(2,2,128))
    cnv15 = Conv2D(filters = 128, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn11)(con3)
    bn15 = BatchNormalization()(cnv15)
    
    rn12 = RandomNormal(stddev = calc_std_dev(3,3,128))
    cnv16 = Conv2D(filters = 128, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn12)(bn15)
    bn16 = BatchNormalization()(cnv16)

    #Decoder 4
    cnvt4 = Conv2DTranspose(filters = 64, activation = 'relu', kernel_size = (2,2), strides = (2,2), padding = "valid", kernel_initializer = rn12)(bn16)
    crop4 = Cropping2D(cropping = (88,88))(cnv2)
    con4 = Concatenate(axis = -1)([cnvt4, crop4])

    rn13 = RandomNormal(stddev = calc_std_dev(2,2,64))
    cnv17 = Conv2D(filters = 64, activation = "relu", kernel_size = (3,3), padding="valid", strides = (1,1), kernel_initializer = rn13)(con4)
    bn17 = BatchNormalization()(cnv17)

    rn14 = RandomNormal(stddev = calc_std_dev(3,3,64))
    cnv18 = Conv2D(filters = 64, activation = "relu", kernel_size = (3,3), padding="valid",  strides = (1,1), kernel_initializer = rn14)(bn17)
    bn18 = BatchNormalization()(cnv18)

    #output
    outputs = Conv2D(filters = 1, kernel_size = (1,1), padding = "valid", strides = (1,1), activation = "sigmoid", kernel_initializer = rn14)(bn18)

    model = Model(inputs = inputs, outputs = outputs, name = "U-net")

    opt = SGD(learning_rate = 0.01, momentum = 0.99)
    loss = BinaryCrossentropy()
    metric = MeanIoU(num_classes = 2)
    model.compile(optimizer = opt, loss = loss, metrics = [metric])

    return model


def train_unet_model(): 
    '''Train Unet on the training set. Saves the model weights for later use.'''
    base_imgs, mask_imgs, _ = load_images_from_dataset_csv(segmentation=True)

    x_train, x_valid, y_train, y_valid = train_test_split(base_imgs, mask_imgs, test_size=TEST_SIZE + VALID_SIZE, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, random_state=42)

    model = define_unet_model()
    model.summary()

    stop = EarlyStopping(monitor = "loss", mode = "min", patience = 5, restore_best_weights = True)

    training_hist = model.fit(x_train, y_train, epochs = N_EPOCHS, batch_size = SEG_BATCH_SIZE, verbose = 1, callbacks = [stop])

    file_name = "Unet_{}.h5".format(len(os.listdir(MODELS_DATA_PATH)) + 1)
    model.save(os.path.join(MODELS_DATA_PATH, file_name))

    predictions = model.predict(x_test, batch_size=SEG_BATCH_SIZE)

    for i, pred in enumerate(predictions):
        plt.imshow(pred)
        plt.show()
        plt.imshow(y_test[i])
        plt.show()

def load_unet_model(): 
    file_path = os.path.join(MODELS_DATA_PATH, "Unet_5.h5")
    model = define_unet_model()
    model.load_weights(file_path)

    return model


if __name__ == "__main__":
    #train_unet_model()

    model = load_unet_model()

    base_imgs, mask_imgs, labels = load_images_from_dataset_csv(segmentation=True)

    malig_base_imgs, malig_mask_imgs = [] , []
    for i, label in enumerate(labels): 
        if label != LABEL_MAP['normal']:
            continue
        
        malig_base_imgs.append(base_imgs[i])
        malig_mask_imgs.append(mask_imgs[i])

    predictions = model.predict(np.array(malig_base_imgs), batch_size = SEG_BATCH_SIZE)

    for i, pred in enumerate(predictions):
        plt.title("Prediction vs Actual")
        plt.subplot(1,3,1)
        plt.imshow(malig_base_imgs[i], cmap=plt.get_cmap("gray"))
        plt.subplot(1,3,2)
        plt.imshow(malig_mask_imgs[i], cmap=plt.get_cmap("gray"))
        plt.subplot(1,3,3)
        plt.imshow(pred, cmap = plt.get_cmap("gray"))
        plt.show()

