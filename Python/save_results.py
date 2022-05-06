# Author: Matt Williams
# Version: 5/5/2022

import json
import os
from util import *



def save_results(dict):
    '''A Simple method to save the json results files.'''
    if not os.path.exists(RESULTS_DATA_PATH):
        os.mkdir(RESULTS_DATA_PATH)

    file_name = "results_{}.json".format(str(len(os.listdir(RESULTS_DATA_PATH)) + 1))

    path = os.path.join(RESULTS_DATA_PATH, file_name)

    with open(path, "w", encoding="utf-8") as file: 
        json.dump(dict, file, indent = 1)