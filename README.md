# Convolutional Variational Autoencoder (CVAE) for Feature Identification

This repository contains an implementation of a Convolutional Variational Autoencoder (CVAE), a type of generative model that's particularly effective with high-dimensional input data, such as images.

## Overview

Convolutional Variational Autoencoders (CVAEs) are a type of deep learning model that combines Variational Autoencoders (VAEs) with Convolutional Neural Networks (CNNs). They are used for generating new data that resembles some set of training data, with the particularity of preserving spatial structure in the data by leveraging the power of CNNs.

The main script `main.py` consists of the following steps:
1. Format and preprocess the image dataset.
2. Train the CVAE model.
3. Encode the images using the trained model.
4. Run the encoded data through a random forest regression to identify the valuable encodings.
5. Visualize and save the useful encodings, as well as the corresponding random forest.
6. Decode the useful encodings and save them.
7. Map the identified sections back to the original image to indicate regions of interest.

## Requirements
- See .txt files - depends on the OS (you want GPU acceleration)

## How to Run
You can run the script `main.py`

## Data Formatting
\
Data must be formatted like so for loading:\
\
Root Path - rapid_feature_identification - py files \
Root Path - Micrographs - parent_dir (see main.py) - subdir0 - 0 / subdir1 - 1 \

---> Automatically processes micrographs into the following input folders if new_micrographs = True
---> Else, manually put micrographs into input folders and set new_micrographs = False

Root Path - Inputs - train - class 0 - (class 0 training images) \
Root Path - Inputs - train - class 1 - (class 1 training images) \
Root Path - Inputs - val - class 0 - (class 0 test images) \
Root Path - Inputs - val - class 1 - (class 1 test images) \

---> Folders created automatically
Root Path - Outputs \
Root Path - Features \


## Contribution
Feel free to fork the project, open a pull request, or submit suggestions and bugs through the issue tracker.

## License
This project is licensed under the Carnegie Mellon University (CMU) License. Please see the `LICENSE` file in the root of the repository if it is there yet. 
