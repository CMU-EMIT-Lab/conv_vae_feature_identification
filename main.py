"""
INFO
File: main.py
Created by: William Frieden Templeton
Date: January 27, 2023
"""

from bin.randomforest import *
from bin.utils import *
from bin.train import train_a_model
from bin.settings import TrainParams
from bin.image_formatter import *
from bin.image_mapper import *
from time import sleep
import datetime

# Logging start time of execution
print(f"Start Execution: {datetime.datetime.now()}")

# Display number of GPUs available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Display Tensorflow version
print('Tensorflow: %s' % tf.__version__)  # print version

parent_dir = 'fatigue_test'
sub_dir = 'fatigue_test_kfolds'
new_micrographs = False

check_params = TrainParams(
    parent_dir=parent_dir,
    name=sub_dir,
    epochs=9000,
    batch_size=256,
    image_size=128,
    latent_dim=int(1028),
    num_examples_to_generate=16,
    learning_rate=0.0005,
    section_divisibility=1,  # set to 1 if full image is needed
    bright_sample=True,
    dofolds=True,
    kfolds=5
)

# If new micrographs are present, they will be formatted
if new_micrographs:
    format_images(from_bin=False, params=check_params)  # False means we're running from main
    # Give the disk a second to notice the files
    sleep(1)

# Train a new model with the given parameters
cvae, test_ds, train_ds, ssim_scores, mean_ssim, std_ssim = train_a_model(check_params)
print(f"End of CVAE Training: {datetime.datetime.now()}")

# Get arrays of encoded data from model
train_encodings, train_labels, train_files, split_train_encodings, _ = get_encoding(cvae, train_ds)
test_encodings, test_labels, test_files, _, _ = get_encoding(cvae, test_ds)

# Run arrays through random forest regression to figure out if any can separate the labels
valuable_encodings, forest_model = random_forest(
    train_encodings,
    train_labels,
    test_encodings,
    test_labels,
    check_params
)
print(f"End of Random Forest Regression: {datetime.datetime.now()}")

# Save the useful encodings and show the tree (saved to "outputs" folder)
show_split(split_train_encodings, valuable_encodings, forest_model, check_params)
save_tree(forest_model, check_params)

# Decode useful encodings and save to folder, so you can do whatever you want with them
positive_features, negative_features = identify_files(
    0,
    train_encodings,
    train_labels,
    test_encodings,
    test_labels,
    forest_model)
pull_key_features(positive_features, negative_features, cvae, check_params.name)

# Similar to the image formatter, run sections through the CVAE, then run those sections through the random forest
# Map back to the original image and save to show where we should be looking on the samples
# map_sections(from_bin=False, crit=0, cvae_model=cvae, rf_model=forest_model, params=check_params)
