# Author: Matt Williams
# Version: 5/5/2022
import os
from util import *
from collections import OrderedDict
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

def match_images_with_masks(dataset_path): 
    '''Given a file path to a dataset sub directory, return a Pandas Data Frame.
    Each row in the data frame contains: the class name, the file name of the base image, and
    the filenames of the associated segmented images/masks. '''
    dict_list = []
    current_dict = None
    
    for filename in os.listdir(dataset_path): 
        if filename.find("mask") == -1:

            if current_dict != None:
                dict_list.append(current_dict)
                
            current_dict = OrderedDict()
            current_dict["Class"] = filename.strip().split()[0]
            current_dict["Image"] = filename

        else: 
            current_dict["Mask {}".format(len(current_dict.keys()) - 1)] = filename

    return pd.DataFrame(dict_list)


def make_dataset_csv():
    '''Using the method above, this method constructs a large Pandas Data Frame that contains the rows
    of each Dataframe associated with a dataset sub directory. '''
    benign_df = match_images_with_masks(BENIGN_DATA_PATH)
    malig_df = match_images_with_masks(MALIGNANT_DATA_PATH)
    normal_df = match_images_with_masks(NORMAL_DATA_PATH)

    return pd.concat([benign_df, malig_df, normal_df], ignore_index = True)


def balance_dataset(dataset_df):
    '''Given a Pandas Data Frame, use a 3rd party Library to perform oversamples on our dataset'''
    oversampler = RandomOverSampler(random_state=0)
    col_names = dataset_df.columns

    labels = dataset_df.iloc[:, CLASS_NAME_COL].to_numpy()
    info = dataset_df.iloc[:, (CLASS_NAME_COL+1):].to_numpy()
    os_info, os_labels = oversampler.fit_resample(info, labels)
    
    new_data_df = pd.DataFrame(columns=col_names)
    new_data_df.iloc[:, CLASS_NAME_COL] = os_labels
    new_data_df.iloc[:, (CLASS_NAME_COL + 1):] = os_info

    return new_data_df

if __name__ == "__main__": 
    dataset_df = make_dataset_csv()
    dataset_df = balance_dataset(dataset_df)
    dataset_df.to_csv(DATA_CSV_PATH, index = False)
