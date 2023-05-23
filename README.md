# conv_vae_feature_identification
Created by: William Frieden Templeton \
Date: 25 January 2023 \
The purpose of this code is to rapidly identify features in micrographs (or images) that can be used to \
separate data per a user-identified classification criteria.\
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
\
## Main.py - Controller for all scripts, will be implemnted in GUI at later date
