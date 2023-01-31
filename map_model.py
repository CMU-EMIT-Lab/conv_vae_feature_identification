# https://towardsdatascience.com/how-to-build-a-weapon-detection-system-using-keras-and-opencv-67b19234e3dd
import pandas as pd
import os
import numpy as np
import keras.utils as image
import sklearn
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def get_array(path, params):
    img = image.load_img(
        path,
        target_size=(params.image_size, params.image_size),
        color_mode="grayscale"
    )
    img = image.img_to_array(img)
    return img/255


def get_images(paths, params):
    output = []
    for path in paths:
        img = get_array(path, params)
        output.append(img)
    return np.array(output)


def get_tts(params):
    np.random.seed(10)
    useful_paths = [
        f'../features/{params.name}/useful/{i}' for i in os.listdir(f'../features/{params.name}/useful')
    ]
    useful_labels = [1] * len(useful_paths)

    not_useful_paths = [
        f'../features/{params.name}/not_useful/{i}' for i in os.listdir(f'../features/{params.name}/not_useful')
    ]
    not_useful_labels = [0] * len(not_useful_paths)

    paths = useful_paths + not_useful_paths
    labels = useful_labels + not_useful_labels
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        paths, labels, stratify=labels, train_size=.90, random_state=10
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
    y_test = image.to_categorical(y_test)
    y_train = image.to_categorical(y_train)
    tts = (new_x_train, new_x_test, y_train, y_test)
    return tts


def detection_model(params):
    import tensorflow as tf
    # This is a simple CNN we will train for image detection
    inp_shape = (params.image_size, params.image_size, 1)
    act = 'relu'
    drop = .25
    initializer = 'he_uniform'
    pad = 'same'
    kernel_reg = tf.keras.regularizers.l1(.001)
    optimizer = tf.keras.optimizers.Adam(lr=.0001)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        16, kernel_size=(3, 3), activation=act, input_shape=inp_shape, kernel_regularizer=kernel_reg,
        kernel_initializer=initializer,  padding=pad, name='Input_Layer'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),  strides=(3, 3)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=act, kernel_regularizer=kernel_reg,
                                     kernel_initializer=initializer, padding=pad))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=act, kernel_regularizer=kernel_reg,
                                     kernel_initializer=initializer, padding=pad))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=act, kernel_regularizer=kernel_reg,
                                     kernel_initializer=initializer, padding=pad))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation=act))
    model.add(tf.keras.layers.Dense(16, activation=act))
    model.add(tf.keras.layers.Dense(8, activation=act))
    model.add(tf.keras.layers.Dropout(drop))
    model.add(tf.keras.layers.Dense(2, activation='softmax', name='Output_Layer'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    from train import TrainParams
    check_params = TrainParams(
     parent_dir='data_binary_watermark',
     name='watermark_test',
     epochs=50,
     batch_size=16,
     image_size=128,
     latent_dim=128,
     num_examples_to_generate=16,
     learning_rate=0.001
     # show_latent_gif=True
    )
    image_train, image_test, label_train, label_test = get_tts(check_params)

    calls = [EarlyStopping(monitor='val_loss', verbose=1, patience=10, min_delta=.00075),
             ModelCheckpoint(
                 f'../outputs/{check_params.name}/{check_params.name}_detector_ModelWeights.h5',
                 verbose=1, save_best_only=True, monitor='val_loss'),
             ReduceLROnPlateau(patience=2, mode='min')
             ]
    cnn_model = detection_model(check_params)
    model_history = cnn_model.fit(image_train, label_train, batch_size=32,
                                  epochs=500,
                                  callbacks=calls,
                                  validation_data=(image_test, label_test),
                                  verbose=1)
