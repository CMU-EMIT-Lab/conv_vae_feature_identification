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
import numpy as np
from bin.utils import loader_pbar, check_dir
import skimage
import matplotlib.pyplot as plt


def load_micrographs(params, classification):
    """
    IN
    params: training parameters with file paths specified
    classification: the current classification (0 or 1) of the files

    OUT
    outputs: A list of cv2 read-in images
    """
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


def remove_background(img, contours):
    """
    PURPOSE
    Remove the background from the sample we want to study. Mount material/background noise will also be encoded
    if it exists which takes up space that could be better used in the encoding. If we only have 1000 channels,
    we don't want 200 to be taken up replicating the texture of the mount material.

    IN
    img: The full input micrograph

    OUT
    img: The micrograph with the mount and background turned transparent
    """
    # a different color than the rest of the sample

    # Create an empty array in the shape of the image with one channel
    background = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Create a contour to match the largest object in the image
    cv2.drawContours(background, [max(contours, key=cv2.contourArea)], 0, 255, -1)

    # max the alpha channel of the non-empty area
    background = skimage.exposure.rescale_intensity(background, in_range=(127.5, 255), out_range=(0, 255))

    # Convert image to color_BGRA to include alpha
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Make everything that's not the sample invisible by setting the alpha channel to match the background mask
    img[background == 0] = (0, 0, 0, 255)
    return img


def crop_micrographs(inputs, params):
    """
    IN
    inputs: List of images from folder (.png, .tiff, .jpg, .whatever)
    params: The training parameters specified in main

    OUT
    outputs: Sectioned images with background (mounting material) removed
    """

    # This crop seeks to remove any borders and text that may skew classification results
    # By default, this will remove the lower 10% of the image as well
    outputs = []
    for img in inputs:

        # If the sample is bright, we will use a normal binary threshold to crop to the sample
        if params.bright_sample:
            _, threshold = cv2.threshold(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                thresh=100,
                maxval=255,
                type=cv2.THRESH_BINARY
            )

        # if the sample is not bright, then we'll crop to the largest bounding box (may need to tweak area[**][1] later)
        else:
            _, threshold = cv2.threshold(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                thresh=100,
                maxval=255,
                type=cv2.THRESH_BINARY_INVERSE
            )

        # Find all contours and record the hierarchy (bounding contours are placed higher)
        contours, hierarchy = cv2.findContours(
            threshold,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort through the area of each contour, and pick the second largest (the sample inside the frame & micrograph)
        areas = []
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            areas.append([w*h, x, y, w, h])
        areas.sort(reverse=True)

        # Record the x, y edges of the largest micrograph
        x, y, w, h = areas[1][1], areas[1][2], areas[1][3], areas[1][4]

        # Make the background of the sample transparent
        img = remove_background(img, contours)

        # Return the sectioned/background removed micrograph
        outputs.append(img[y:y + h, x:x + w])
    return outputs


def slice_utility(img, params):

    # Get the shape of the image
    h, w, channels = img.shape

    # This specifies the distance between width and height rectangle corners per the section divisibility
    cutter_w = w//params.section_divisibility
    cutter_h = h//params.section_divisibility
    sectors = range(1, params.section_divisibility+1)

    # Make the combinations of the widths and heights we need to chop up the image grid-wise
    slap_chop = [(w, h) for w in sectors for h in sectors]

    # Get the sections per the section interval and location on the image
    sections = [img[cutter_h * (h - 1):cutter_h * h, cutter_w * (w - 1):cutter_w * w] for w, h in slap_chop]

    # Also record the location of the images for file path data
    axes = [[w, h] for w, h in slap_chop]

    # Return the sections, the axes, and the interval
    return sections, axes, cutter_w, cutter_h


def slice_images(from_bin, inputs, crit, params):
    pbar = tqdm.tqdm(range(len(inputs)))
    for i in pbar:
        sections, axes, _, _ = slice_utility(inputs[i], params)
        count = 0
        for section, axis in zip(sections, axes):
            count += 1
            if count % params.test_train_split:
                if from_bin:
                    cv2.imwrite(
                        f'../../input/{params.parent_dir}/val/{crit}/img_{i}_section_w{axis[0]}_h{axis[1]}.png',
                        section
                    )
                else:
                    cv2.imwrite(
                        f'../input/{params.parent_dir}/val/{crit}/img_{i}_section_w{axis[0]}_h{axis[1]}.png',
                        section
                    )
            else:
                if from_bin:
                    cv2.imwrite(
                        f'../../input/{params.parent_dir}/train/{crit}/img_{i}_section_w{axis[0]}_h{axis[1]}.png',
                        section
                    )
                else:
                    cv2.imwrite(
                        f'../input/{params.parent_dir}/train/{crit}/img_{i}_section_w{axis[0]}_h{axis[1]}.png',
                        section
                    )
        loader_pbar(i, crit, pbar)


def format_images(from_bin, params):
    check_dir('input', from_bin, params.parent_dir)
    for criteria in [0, 1]:
        # Stacked the data loader and the cropping function into one line
        images = crop_micrographs(load_micrographs(params, criteria), params)
        slice_images(from_bin, images, criteria, params)
    plt.imshow(images[0])
    plt.show()


if __name__ == "__main__":
    parent_dir = 'test_dataset'
    sub_dir = 'test_model'

    from bin.settings import TrainParams
    check_params = TrainParams(
        parent_dir=parent_dir,
        name=sub_dir,
        section_divisibility=15,
        test_train_split=5,
        # If your sample is brighter than the background, make true - this influences crop_micrographs
        bright_sample=True
    )

    format_images(True, check_params)
