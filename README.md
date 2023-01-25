# rapid_feature_identification

## Main.py - Controller for all scripts, will be implemnted in GUI at later date

## train_model Segment: 
### Model - Subclassed convolutional variational autoencoder
### Engine - Applies custom training loop to model, designed to track file paths and labels into the encoded data
### Train - Defines the custom training loop and records data

## find_dimensions Segment:
### randomforest.get_encoding - Encodes all training or test data (depends on user)
### randomforest.random_forest - Splits encoding via random forest regression (default parameters currently used) and returns valuable encoded dimensions plus RF model
### utils.show_split - Visualizes the dimension and split criteria identified by random forest regression
### utils.save_tree - Shows the first tree among random forest trees
