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
from bin.utils import loader_pbar, check_dir


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
        contours, hierarchy = cv2.findContours(
            threshold,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        areas = []
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            areas.append([w*h, x, y, w, h])
        areas.sort(reverse=True)
        x, y, w, h = areas[1][1], areas[1][2], areas[1][3], areas[1][4]
        outputs.append(img[y:y+h, x:x+w])
    return outputs


def slice_images(for_inputs, for_mapping, from_bin, inputs, subdir, params):
    pbar = tqdm.tqdm(range(len(inputs)))
    for i in pbar:
        count = 0
        h, w, channels = inputs[i].shape
        cutter_w = w//params.section_divisibility
        cutter_h = h//params.section_divisibility
        sectors = range(1, params.section_divisibility+1)
        slap_chopper = [(w, h) for w in sectors for h in sectors]
        for w, h in slap_chopper:
            count += 1
            section = inputs[i][cutter_h*(h-1):cutter_h*h, cutter_w*(w-1):cutter_w*w]
            if for_inputs:
                if count % params.test_train_split:
                    if from_bin:
                        cv2.imwrite(
                            f'../../input/{params.parent_dir}/val/{subdir}/img_{i}_section_w{w}_h{h}.png',
                            cv2.cvtColor(section, cv2.COLOR_RGB2BGR)
                        )
                    else:
                        cv2.imwrite(
                            f'../input/{params.parent_dir}/val/{subdir}/img_{i}_section_w{w}_h{h}.png',
                            cv2.cvtColor(section, cv2.COLOR_RGB2BGR)
                        )
                else:
                    if from_bin:
                        cv2.imwrite(
                            f'../../input/{params.parent_dir}/train/{subdir}/img_{i}_section_w{w}_h{h}.png',
                            cv2.cvtColor(section, cv2.COLOR_RGB2BGR)
                        )
                    else:
                        cv2.imwrite(
                            f'../input/{params.parent_dir}/train/{subdir}/img_{i}_section_w{w}_h{h}.png',
                            cv2.cvtColor(section, cv2.COLOR_RGB2BGR)
                        )
            # elif for_mapping:
            #     bre
        loader_pbar(i, subdir, pbar)


def format_images(from_bin, params):
    import matplotlib.pyplot as plt
    check_dir('input', from_bin, params.parent_dir)
    for criteria in [0, 1]:
        # Stacked the data loader and the cropping function into one line
        images = crop_micrographs(load_micrographs(params, criteria), params)
        slice_images(True, False, from_bin, images, criteria, params)
    plt.imshow(images[0])
    plt.show()


if __name__ == "__main__":
    parent_dir = 'test_dataset'
    sub_dir = 'test_model'

    from bin.settings import TrainParams
    check_params = TrainParams(
        parent_dir=parent_dir,
        name=sub_dir,
        section_divisibility=4,
        # If your sample is brighter than the background, make true - this influences crop_micrographs
        bright_sample=True
    )

    format_images(True, check_params)
