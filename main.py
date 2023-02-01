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

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('Tensorflow: %s' % tf.__version__)  # print version

parent_dir = 'HighCycleLowCycleNoBorder_Regime'
sub_dir = 'full_test_no_borders'

check_params = TrainParams(
    parent_dir=parent_dir,
    name=sub_dir,
    epochs=200,
    batch_size=16,
    image_size=64,
    latent_dim=256,
    num_examples_to_generate=16,
    learning_rate=0.0001,
    section_divisibility=10
)

cvae, test_ds, train_ds = train_a_model(check_params)

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
show_split(split_train_encodings, valuable_encodings, forest_model, check_params)
save_tree(forest_model, check_params)

positive_features, negative_features = identify_files(
    0,
    train_encodings,
    train_labels,
    test_encodings,
    test_labels,
    forest_model)
pull_key_features(positive_features, negative_features, cvae, check_params.name)
