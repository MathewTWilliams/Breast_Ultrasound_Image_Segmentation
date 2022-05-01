import os
import pandas as pd
ABS_CWD = os.path.abspath(os.getcwd())

DATA_SETS_PATH = os.path.join(ABS_CWD, "Datasets")
BENIGN_DATA_PATH = os.path.join(DATA_SETS_PATH, "benign")
MALIGNANT_DATA_PATH = os.path.join(DATA_SETS_PATH, "malignant")
NORMAL_DATA_PATH = os.path.join(DATA_SETS_PATH, "normal")
DATA_CSV_PATH = os.path.join(DATA_SETS_PATH, "complete_datset.csv")
RESULTS_DATA_PATH = os.path.join(ABS_CWD, "Results")



MODELS_DATA_PATH = os.path.join(ABS_CWD, "Models")



# for resizing
TARGET_WIDTH = 550
TARGET_HEIGHT = 550


#Data frame column indicies
CLASS_NAME_COL = 0
FILE_NAME_COL = 1
MASK_NAME_1_COL = 2
MASK_NAME_2_COL = 3
MASK_NAME_3_COL = 4

TEST_SIZE = 0.10
VALID_SIZE = 0.10

N_TARGET_SAMPLES = 436

N_EPOCHS = 100
BATCH_SIZE = 8

DIR_PATH_MAP = {
    "benign" : BENIGN_DATA_PATH,
    "malignant" : MALIGNANT_DATA_PATH,
    "normal" : NORMAL_DATA_PATH, 
}

LABEL_MAP = {
    "normal" : 0,
    "benign" : 1,
    "malignant" : 2,
}


if __name__ == "__main__":
    dataset_df = pd.read_csv(DATA_CSV_PATH, index_col = False)

    groupby = dataset_df.groupby(dataset_df.columns[0])
    
    for key in LABEL_MAP.keys(): 
        print(groupby.get_group(key))
    