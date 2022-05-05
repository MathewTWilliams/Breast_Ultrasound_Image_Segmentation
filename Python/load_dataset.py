from email.mime import base
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2
from util import *

def load_images_from_dataset_csv(segmentation = False):
    dataset_df = pd.read_csv(DATA_CSV_PATH, index_col=False)

    input_height = SEGMENT_INPUT_HEIGHT if segmentation else CLASSIFY_INPUT_HEIGHT
    input_width = SEGMENT_INPUT_WIDTH if segmentation else CLASSIFY_INPUT_WIDTH


    images_list = np.empty(shape=(len(dataset_df.index.values), input_height, input_width, 1))
    masks_list = np.empty(shape=(len(dataset_df.index.values), CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_HEIGHT, 1))
    labels_list = np.empty(shape=(len(dataset_df.index.values),))

    for i, row_ds in dataset_df.iterrows():

        img_class = row_ds[CLASS_NAME_COL]
        filename = row_ds[FILE_NAME_COL]
        mask_1 = row_ds[MASK_NAME_1_COL]
        mask_2 = row_ds[MASK_NAME_2_COL]
        mask_3 = row_ds[MASK_NAME_3_COL]

        base_img = cv2.imread(os.path.join(DIR_PATH_MAP[img_class], filename), cv2.IMREAD_GRAYSCALE)
        #resize to target height and width
        base_img = cv2.resize(base_img, (input_height, input_width))
        
        labels_list[i] = LABEL_MAP[img_class]
        images_list[i] = np.reshape(base_img, newshape=(input_height, input_width, 1))
    
        mask_names = [mask_1, mask_2, mask_3]
        mask_names = [mask for mask in mask_names if not pd.isna(mask)]
        cur_mask_images = []

        for mask in mask_names: 
            mask_image = cv2.imread(os.path.join(DIR_PATH_MAP[img_class], mask), cv2.IMREAD_GRAYSCALE)
            mask_image = cv2.resize(mask_image, (CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_WIDTH))

            cur_mask_images.append(mask_image)

        final_mask_image = cur_mask_images[0]
        
        if len(cur_mask_images) > 1: 
            for i in range(1, len(cur_mask_images)): 
                cv2.add(final_mask_image, cur_mask_images[i], final_mask_image)

        
        masks_list[i] = np.reshape(final_mask_image, newshape=(CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_WIDTH, 1))
        
    return (images_list / 255.0), \
            (masks_list / 255.0), \
            labels_list

        
        


        
if __name__ == "__main__": 
    load_images_from_dataset_csv()