import os
from util import *
from collections import OrderedDict
import pandas as pd

def match_images_with_masks(dataset_path): 
    
    dict_list = []
    current_dict = None
    
    for filename in os.listdir(dataset_path): 
        print(filename)
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
    
    benign_df = match_images_with_masks(BENIGN_DATA_PATH)
    malig_df = match_images_with_masks(MALIGNANT_DATA_PATH)
    normal_df = match_images_with_masks(NORMAL_DATA_PATH)

    pd.concat([benign_df, malig_df, normal_df], ignore_index = True).to_csv(DATA_CSV_PATH, index=False)

if __name__ == "__main__": 
    make_dataset_csv()
