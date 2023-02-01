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
import glob
import os.path


def load_micrographs(params, classification):
    # This is a simple file that loads the micrographs into an OpenCV dataset
    images = [cv2.imread(file) for file in glob.glob(f"../micrographs/{classification}/*") if not file.startswith('.')]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    return images


def crop_micrographs(params):
    # This crop seeks to remove any borders and text that may skew classification results

    return


if __name__ == "__main__":
    parent_dir = 'HighCycleLowCycleNoBorder_Regime'
    sub_dir = 'full_test_no_borders'

    from bin.settings import TrainParams
    check_params = TrainParams(
        parent_dir=parent_dir,
        name=sub_dir
    )

    for criteria in [0, 1]:
        raw_images = load_micrographs(check_params, criteria)

