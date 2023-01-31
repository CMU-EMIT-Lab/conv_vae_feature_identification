# https://towardsdatascience.com/how-to-build-a-weapon-detection-system-using-keras-and-opencv-67b19234e3dd
import pandas as pd
import os
import numpy as np
import cv2
from keras.preprocessing import image
import sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from skimage.segmentation import mark_boundaries


def get_array(path, params):
    '''This function will read an image and convert to a specified version and resize depending on which algorithm is being used. '''
    img = image.load_img(path, target_size=params.image_size)
    img = image.img_to_array(img)
    return img/255


def get_images(paths, params):
    output = []
    for path in paths:
        img = get_array(path, params)
        output.append(img)
    return np.array(output)


def get_tts(params):
     dim =  (params.image_size,params.image_size)
     np.random.seed(10)
     useful_paths = [
         f'../features/{params.name}/useful/{i}' for i in os.listdir(f'../features/{params.name}/useful')
     ]
     useful_labels = [1 for i in range(len(useful_paths))]

     not_useful_paths = [
         f'../features/{params.name}/not_useful/{i}' for i in os.listdir(f'../features/{params.name}/not_useful')
     ]
     not_useful_labels = [0 for i in range(len(not_useful_paths))]

     paths = useful_paths + not_useful_paths
     labels = useful_labels + not_useful_labels
     x_train, x_test, y_train, y_test = train_test_split(
         paths, labels, stratify = labels, train_size = .90, random_state = 10
     )

     new_x_train = get_images(x_train, params)
     new_x_test = get_images(x_test, params)

     print('Train Value Counts')
     print(pd.Series(y_train).value_counts())
     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
     print('Test Value Counts')
     print(pd.Series(y_test).value_counts())
     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
     print('X Train Shape')
     print(new_x_train.shape)
     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
     print('X Test Shape')
     print(new_x_test.shape)
     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

     y_train = np.array(y_train)
     y_test = np.array(y_test)
     y_test = to_categorical(y_test)
     y_train = to_categorical(y_train)
     tts = (new_x_train, new_x_test, y_train, y_test)
     return tts



def detection_model(params):
 from keras.models import Sequential
 from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Dense, Dropout, Flatten
 from keras.optimizers import Adam
 from keras import regularizers
 # This is a simple CNN we will train for image detection
 inp_shape = params.image_size
 act = 'relu'
 drop = .25
 kernel_reg = regularizers.l1(.001)
 optimizer = Adam(lr=.0001)
 model = Sequential()
 model.add(Conv2D(64, kernel_size=(3, 3), activation=act, input_shape=inp_shape, kernel_regularizer=kernel_reg,
                  kernel_initializer='he_uniform',  padding='same', name='Input_Layer'))
 model.add(MaxPooling2D(pool_size=(2, 2),  strides=(3, 3)))
 model.add(Conv2D(64, (3, 3), activation=act, kernel_regularizer=kernel_reg,
                  kernel_initializer='he_uniform', padding='same'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
 model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer=kernel_reg,
                  kernel_initializer='he_uniform', padding='same'))
 model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer=kernel_reg,
                  kernel_initializer='he_uniform', padding='same'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dense(64, activation='relu'))
 model.add(Dense(32, activation='relu'))
 model.add(Dropout(drop))
 model.add(Dense(3, activation='softmax', name='Output_Layer'))
 model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
 return model


if __name__ == "__main__":
    from train import TrainParams
    check_params = TrainParams(
     parent_dir='data_binary_watermark',
     name='watermark_test',
     epochs=50,
     batch_size=16,
     image_size=64,
     latent_dim=128,
     num_examples_to_generate=16,
     learning_rate=0.001
     # show_latent_gif=True
    )
    image_train, image_test, label_train, label_test = get_tts(check_params)
