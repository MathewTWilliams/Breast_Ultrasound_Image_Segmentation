# Author: Matt Williams
# Version: 5/5/2022
from util import CLASS_BATCH_SIZE, SEG_BATCH_SIZE, INVERSE_LABEL_MAP
from load_dataset import load_images_from_dataset_csv
from unet import load_unet_model
from alexnet import load_alexnet_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from save_results import save_results


def run_predictions():
    '''A method that runs the full segmentation and classification predicts using the two neural networks'''
    base_imgs, mask_imgs, labels = load_images_from_dataset_csv(segmentation=True)

    base_imgs, mask_imgs, labels = shuffle(base_imgs, mask_imgs, labels)
    model = load_unet_model()

    seg_predictions = model.predict(base_imgs, batch_size = SEG_BATCH_SIZE)

    del model

    model = load_alexnet_model()
    cls_predictions = model.predict(seg_predictions, batch_size = CLASS_BATCH_SIZE)

    del model

    cat_cls_predictions = np.argmax(cls_predictions, axis = -1)
    class_report = classification_report(labels, cat_cls_predictions, output_dict=True)
    class_report["Model"] = "Combined"
    save_results(class_report)

    for i, seg_pred in enumerate(seg_predictions):
        print("------------------------------------------")
        print("Percentages:")
        print("Normal: {}".format(cls_predictions[i][0]))
        print("Benign: {}".format(cls_predictions[i][1]))
        print("Malignant: {}".format(cls_predictions[i][2]))

        plt.subplot(1,3,1)
        plt.imshow(base_imgs[i], cmap=plt.get_cmap("gray"))
        plt.subplot(1,3,2)
        plt.title("Actual: {}".format(INVERSE_LABEL_MAP[labels[i]]))
        plt.imshow(mask_imgs[i], cmap = plt.get_cmap("gray"))
        plt.subplot(1,3,3)
        plt.title("Prediction: {}".format(INVERSE_LABEL_MAP[cat_cls_predictions[i]]))
        plt.imshow(seg_pred, cmap=plt.get_cmap("gray"))
        plt.show()






if __name__ == "__main__": 
    run_predictions()