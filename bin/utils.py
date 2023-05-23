"""
INFO
File: utils.py
Created by: William Frieden Templeton
Date: January 27, 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import path, mkdir, makedirs
import tensorflow as tf
import pandas as pd
mpl.pyplot.rcParams.update({'font.size': 14})
mpl.pyplot.rcParams.update({'font.family': 'serif'})


def check_dir(in_out, from_bin, name):
    if from_bin:
        if not path.isdir(f'../../{in_out}/{name}'):
            mkdir(f'../../{in_out}/{name}')
            if in_out == 'input':
                mkdir(f'../../{in_out}/{name}/val')
                mkdir(f'../../{in_out}/{name}/train')
                mkdir(f'../../{in_out}/{name}/val/0')
                mkdir(f'../../{in_out}/{name}/val/1')
                mkdir(f'../../{in_out}/{name}/train/0')
                mkdir(f'../../{in_out}/{name}/train/1')
    else:
        if not path.isdir(f'../{in_out}/{name}'):
            mkdir(f'../{in_out}/{name}')
            if in_out == 'input':
                mkdir(f'../{in_out}/{name}/val')
                mkdir(f'../{in_out}/{name}/train')
                mkdir(f'../{in_out}/{name}/val/0')
                mkdir(f'../{in_out}/{name}/val/1')
                mkdir(f'../{in_out}/{name}/train/0')
                mkdir(f'../{in_out}/{name}/train/1')


def save_reconstructed_images(model, epoch, test_sample, test_label, max_epoch, name):
    mean, log_var = model.encode(test_sample)
    z = model.re_parameterize(mean, log_var)
    listed_z = {}

    predictions = model.sample(z)
    plt.figure(figsize=(15, 15))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap=plt.get_cmap('Greys_r'))
        plt.title(int(test_label[i]), size=32)
        plt.axis('off')
        listed_z["{}_{}".format(i, int(test_label[i] + 1))] = z[i, :]

    plt.savefig(f'../outputs/{name}/output{epoch}.jpg', dpi=100)
    plt.close()

    if epoch == max_epoch:
        df = pd.DataFrame(listed_z)
        df.to_csv(f'../outputs/{name}/z_pred_sample.csv')
    return f'../outputs/{name}/output{epoch}.jpg'


def printer(model, branch, name):
    with open(f'../outputs/{name}/{branch}_summary.txt', 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def get_sample(batch_size, num_examples_to_generate, test_set, name, epochs, model):
    assert batch_size >= num_examples_to_generate
    for test_batch, test_label, test_file in test_set.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
        test_label = test_label.numpy()
    # noinspection PyUnboundLocalVariable
    input_images(test_sample, name, test_label)
    save_reconstructed_images(model, 0, test_sample, test_label, epochs, name)
    return test_sample, test_label


def input_images(image, name, label):
    plt.figure(figsize=(15, 15))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(image[i].numpy() / 255, cmap=plt.get_cmap('Greys_r'))
        plt.title(int(label[i]), size=32)
        plt.axis("off")
    plt.savefig(f'../outputs/{name}/input_example.png', dpi=100)
    plt.show()
    return f'../outputs/{name}/input_example.png'


def calculate_ssim(image1, image2):
    from skimage.metrics import structural_similarity as compare_ssim
    return compare_ssim(image1[:, :, 0], image2[:, :, 0], channel_axis=None)


def start_llist(latent_dim):
    log_lists = []
    for i in range(latent_dim):
        if i % 2 == 0:
            log_lists.append([i, i + 1])
    return log_lists


def save_loss_plot(train_loss, valid_loss, name):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss[:], color='orange', label='val loss')
    plt.plot(valid_loss[:], color='red', label='train loss')
    plt.ylim([0, 11000])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'../outputs/{name}/loss.jpg')
    plt.show()


def get_dataset(parent_dir, sub_dir, image_size, batch_size):
    datagen = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'../input/{parent_dir}/{sub_dir}/',
        color_mode='grayscale', #rgba
        labels='inferred',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        # seed=11
        seed=1
    )
    return datagen


def format_dataset(images, labels, paths):
    normalization_layer = tf.keras.layers.Rescaling(1/255)
    images = normalization_layer(images)
    labels = tf.cast(labels, tf.float32)
    return images, labels, tf.constant(paths)


def update_pbar(e_loss, r_loss, k_loss, pbar):
    return pbar.set_postfix_str(f"ELBO Loss: {e_loss} - Reconstruction Loss: {r_loss} - KL Loss: {k_loss}")


def loader_pbar(file, criteria, pbar):
    if criteria == 0:
        value = 'positive'
    else:
        value = 'negative'
    return pbar.set_postfix_str(f'Slicing file: {file} in {value} images')


def save_forest(forest, importance, f1, name):
    forest.to_csv(f'../outputs/{name}/ranked_latent_dims.csv')

    # Visualize the important encodings
    fig, ax = plt.subplots()
    plt.rcParams.update({
        'font.size': 22})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    forest.plot.bar(ax=ax, color='gray')
    ax.set_title(f"MDI - F1 = {np.round(f1, 2)}", fontsize=18)
    ax.set_ylabel("Mean decrease in impurity", fontsize=16)
    ax.set_xlabel("Latent Dimension", fontsize=16)
    ax.set_xlim([-0.5, 5.5])
    ax.set_ylim([0, np.max(importance)+np.max(importance)*0.1])
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black', labelsize=14)
    ax.tick_params(axis='y', colors='black', labelsize=14)
    fig.tight_layout()
    plt.savefig(f'../outputs/{name}/ranked_latent_dims.jpg', transparent=True, dpi=100)
    plt.show()


def show_split(parted_encodings, forest_importance, regressor, params):

    # most important feature
    trees = [tree for tree in regressor.estimators_]
    top_dims = []
    threshold = {}
    for i in range(0, 3):
        top_dims.append(forest_importance.index[i])
        threshold[i] = []
        for tree in trees:
            split_val = tree.tree_.threshold[i]
            threshold[i].append(split_val)

    # This will only every show two classifications by design
    colors = ['#595959', '#E7E7E7']

    for i in range(len(top_dims)):
        fig = plt.figure(figsize=(10, 10))
        plt.rcParams.update({
            'font.size': 22})
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        for c in parted_encodings:
            plt.hist(
                parted_encodings[c][:, top_dims[i]],
                color=colors[int(c)], alpha=0.5, edgecolor='black', bins=20, label=f'Label: {c}')
        plt.axvline(
            np.min(threshold[i]), color='k', linestyle='dashed', linewidth=3, label='Decision Threshold Limits'
        )
        plt.axvline(
            np.mean(threshold[i]), color='k', linestyle='solid', linewidth=3, label='Average Decision Threshold'
        )
        plt.axvline(
            np.max(threshold[i]), color='k', linestyle='dashed', linewidth=3
        )
        plt.xlabel(f'Latent Dimension {top_dims[i]} Values')
        plt.ylabel('Number of Images Encoded to Dimension')
        plt.legend(loc='upper left', frameon=False)
        plt.tight_layout()
        plt.savefig(f'../outputs/{params.name}/no{i}_valuable_dimension_{top_dims[i]}.jpg',
                    transparent=True, dpi=100)
        plt.show()


def save_tree(regressor, params):
    import sklearn.tree as sk_t
    import pydot  # Pull out one tree from the forest
    tree = regressor.estimators_[3]  # Export the image to a dot file
    sk_t.export_graphviz(tree, out_file=f'../outputs/{params.name}/tree.dot')
    (graph,) = pydot.graph_from_dot_file(f'../outputs/{params.name}/tree.dot')
    # Write graph to a png file
    graph.write_png(f'../outputs/{params.name}/tree.png')


def pull_key_features(useful_encodings, not_useful_encodings, model, name):
    if not path.isdir(f'../features/{name}/useful/'):
        makedirs(f'../features/{name}/useful/')
    if not path.isdir(f'../features/{name}/not_useful/'):
        makedirs(f'../features/{name}/not_useful/')

    if useful_encodings.shape[0] > 200:
        predictions = model.sample(useful_encodings[:200, :])
    else:
        predictions = model.sample(useful_encodings)
    plt.figure(figsize=(15, 15))
    count = 0
    for i in range(predictions.shape[0]):
        plt.imshow(predictions[i], cmap=plt.get_cmap('Greys_r'))
        plt.axis('off')
        plt.savefig(f'../features/{name}/useful/output_{count}.jpg', dpi=50, bbox_inches='tight')
        plt.close()
        count += 1

    if not_useful_encodings.shape[0] > 200:
        predictions = model.sample(not_useful_encodings[:200, :])
    else:
        predictions = model.sample(not_useful_encodings)
    plt.figure(figsize=(15, 15))
    count = 0
    for i in range(predictions.shape[0]):
        plt.imshow(predictions[i], cmap=plt.get_cmap('Greys_r'))
        plt.axis('off')
        plt.savefig(f'../features/{name}/not_useful/output_{count}.jpg', dpi=50, bbox_inches='tight')
        plt.close()
        count += 1
