"""
INFO
File: image_formatter.py
Created by: William Frieden Templeton
Date: February 1, 2023

SUMMARY
This code will grab images, cut them into user-specified sections (you state the divisibility) and then deposit
the data into the "../inputs" folder for CVAE training.

USER GUIDE
This code will look for two directories under "micrographs" -> 0 and 1
You should split your data along the lines you want to classify at when putting in these folders. It is recommended
to use folder 0 for the images with whichever criteria you're interested in finding the cause of.

For example, when investigating the cause of low-cycle fatigue failure, I will put the micrographs from samples that
failed in low-cycle conditions into the "../micrographs/0" folder, and the micrographs that failed at high-cycle into
the "../micrographs/1" folder.

Additionally, you will specify the test/train split at this step in the TrainParams class.
"""
import cv2
import tqdm
import os.path
import matplotlib.pyplot as plt


def load_micrographs(params, classification):
    # This is a simple file that loads the micrographs into an OpenCV dataset
    # if running as main, look up a folder
    if __name__ == "__main__":
        outputs = [
            cv2.imread(f"../../micrographs/{params.parent_dir}/{classification}/{file}") for file in os.listdir(
                f"../../micrographs/{params.parent_dir}/{classification}"
            ) if not file.startswith('.')
                  ]
    else:
        outputs = [
            cv2.imread(f"../micrographs/{params.parent_dir}/{classification}/{file}") for file in os.listdir(
                f"../micrographs/{params.parent_dir}/{classification}"
            ) if not file.startswith('.')
                  ]
    outputs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in outputs]
    return outputs


def crop_micrographs(inputs, params):
    # This crop seeks to remove any borders and text that may skew classification results
    # By default, this will remove the lower 10% of the image as well
    outputs = []
    for img in inputs:
        if params.bright_sample:
            _, threshold = cv2.threshold(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                thresh=100,
                maxval=255,
                type=cv2.THRESH_BINARY
            )
        else:
            _, threshold = cv2.threshold(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                thresh=100,
                maxval=255,
                type=cv2.THRESH_BINARY_INVERSE
            )
        contours, hierarchy = cv2.findContours(
            threshold,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        max_area = 0
        areas = []
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            areas.append([w*h, x, y, w, h])
        areas.sort(reverse=True)
        x, y, w, h = areas[1][1], areas[1][2], areas[1][3], areas[1][4]
        outputs.append(img[y:y+h, x:x+w])
    return outputs


if __name__ == "__main__":
    parent_dir = 'HighCycleLowCycle_Regime'
    sub_dir = 'full_test_no_borders'

    from bin.settings import TrainParams
    check_params = TrainParams(
        parent_dir=parent_dir,
        name=sub_dir,
        # If your sample is brighter than the background, make true - this influences crop_micrographs
        bright_sample=True
    )

    for criteria in [0, 1]:
        # Stacked the data loader and the cropping function into one line
        images = crop_micrographs(load_micrographs(check_params, criteria), check_params)

    plt.imshow(images[0])
    plt.show()

