from bin.image_formatter import slice_utility, load_micrographs
from bin.utils import loader_pbar, check_dir
import tqdm
import numpy as np
import cv2
import tensorflow as tf


def map_sections(from_bin, crit, cvae_model, rf_model, params):
    inputs = load_micrographs(params, crit)
    pbar = tqdm.tqdm(range(len(inputs)))
    check_dir('maps', from_bin, params.parent_dir)
    for i in pbar:
        sections, axes, cutter_w, cutter_h = slice_utility(inputs[i], params)
        for section, axis in zip(sections, axes):
            # Get the odds of it being useful
            section = tf.keras.preprocessing.image.array_to_img(section)
            section = tf.image.resize(section, (128, 128))
            print(section.shape)

            mean, log_var = cvae_model.encode(section)
            encoding = cvae_model.re_parameterize(mean, log_var)
            prediction = rf_model.predict(encoding)  # We could convert to binary, or could leave as a % chance

            # Create a mask of the whole image
            masks = np.zeros_like(inputs[i], np.uint8)
            # Draw a rectangle over the spot of the section we just checked
            cv2.rectangle(masks,
                          (cutter_h * (axis[1] - 1), cutter_w * (axis[0] - 1)),
                          (cutter_h * axis[1], cutter_w * axis[0]),
                          (0, 0, 255),
                          cv2.FILLED
                          )
            # Convert to a bool (True/False grid)
            mask = masks.astype(bool)
            # Add a box with an alpha == the prediction (a prediction of 1 means highly likely it's useful)
            inputs[i][mask] = cv2.addWeighted(inputs[i], 1-prediction, mask, prediction, 0)[mask]
        # Save the image
        cv2.imwrite(
            f'../../maps/{params.parent_dir}/img_{i}.png',
            inputs[i]
        )
        loader_pbar(i, crit, pbar)
