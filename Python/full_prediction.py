from util import CLASS_BATCH_SIZE, SEG_BATCH_SIZE, INVERSE_LABEL_MAP
from load_dataset import load_images_from_dataset_csv
from unet import load_unet_model
from alexnet import load_alexnet_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from save_results import save_results


from tensorflow.keras.utils import to_categorical


def run_predictions(): 
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
        percentages = "Percentages: Normal: {}, Benign: {}, Malignant: {}" \
           .format(round(cls_predictions[i][0],5), round(cls_predictions[i][1], 5), round(cls_predictions[i][2], 5))
        plt.subplot(1,3,1)
        plt.title(percentages)
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