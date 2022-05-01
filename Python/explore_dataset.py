import cv2
import statistics
import matplotlib.pyplot as plt
from util import *

if __name__ == "__main__":

    widths = []
    heights = []
    aspect_ratios = []

    for dataset in [BENIGN_DATA_PATH, MALIGNANT_DATA_PATH, NORMAL_DATA_PATH]: 
        for file in os.listdir(dataset):
            image = cv2.imread(os.path.join(dataset, file))
            widths.append(image.shape[1])
            heights.append(image.shape[0])
            aspect_ratios.append(image.shape[1]/ image.shape[0])

        
    width_mean = statistics.mean(widths)
    height_mean = statistics.mean(heights)
    width_std_dev = statistics.stdev(widths)
    height_std_dev = statistics.stdev(heights)

    print("Mean width: {}".format(width_mean))
    print("Median width: {}".format(statistics.median(widths)))
    print("Std. Dev. width: {}".format(width_std_dev))
    print("Min width: {}".format(min(widths)))
    print("Max width: {}".format(max(widths)))

    print("Mean height: {}".format(height_mean))
    print("Median height: {}".format(statistics.median(heights)))
    print("Std. Dev. height: {}".format(height_std_dev))
    print("Min height: {}".format(min(heights)))
    print("Max height: {}".format(max(heights)))

    print("Aspect Ratio mean: {}".format(statistics.mean(aspect_ratios)))
    print("Aspect Ratio median: {}".format(statistics.median(aspect_ratios)))


    plt.boxplot([widths, heights], labels=["Widths", "Heights"])
    plt.show()


