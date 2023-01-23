import numpy as np
import matplotlib.pyplot as plt
from os import path, mkdir
import tensorflow as tf
import pandas as pd


def check_dir(name):
    if not path.isdir(f'../outputs/{name}'):
        mkdir(f'../outputs/{name}')


def save_reconstructed_images(model, epoch, test_sample, test_label, max_epoch, name):
    mean, log_var = model.encode(test_sample)
    z = model.re_parameterize(mean, log_var)
    listed_z = {}

    predictions = model.sample(z)
    plt.figure(figsize=(15, 15))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap=plt.get_cmap('Greys_r'))
        plt.title(int(test_label[i] + 1), size=32)
        plt.axis('off')
        listed_z["{}_{}".format(i, int(test_label[i] + 1))] = z[i, :]

    plt.savefig(f'../outputs/{name}/output{epoch}.jpg')
    plt.show()

    if epoch == max_epoch:
        df = pd.DataFrame(listed_z)
        df.to_csv(f'../outputs/{name}/z_pred_sample.csv')


def generate_latent_iteration(model, epoch, val_ds, log_list, name):
    z = []
    c = []
    for x, y, f in val_ds:
        mean, log_var = model.encode(x)
        # import ipdb; ipdb.set_trace()
        z.append(model.reparameterize(mean, log_var))
        c.append(y.numpy())
        # noinspection PyUnboundLocalVariable
        zp = np.concatenate(z, axis=0)
        cpn = np.concatenate(c, axis=0)
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'yellow']
        cp = [colors[int(i)] for i in cpn]

    count = 0
    for lists in log_list:
        plt.figure(figsize=(12, 10))
        # noinspection PyUnboundLocalVariable
        plt.scatter((zp[:, lists[0]]-np.min(zp[:, lists[0]]))/(np.max(zp[:, lists[0]])-np.min(zp[:, lists[0]])),
                    (zp[:, lists[1]]-np.min(zp[:, lists[1]]))/(np.max(zp[:, lists[1]])-np.min(zp[:, lists[1]])),
                    color=cp, alpha=0.75)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Vector {}'.format(lists[0]), size=25)
        plt.ylabel('Vector {}'.format(lists[1]), size=25)
        plt.grid('off')
        # ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(
            f'../outputs/{name}/Latent_Space_Vectors_{lists[0]}_{lists[1]}_Epoch_{epoch}',
            dpi=100)
        plt.close()
        count += 1


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
        plt.title(int(label[i] + 1), size=32)
        plt.axis("off")
    plt.savefig(f'../outputs/{name}/input_example.png', dpi=100)
    plt.show()


def start_llist(latent_dim):
    log_lists = []
    for i in range(latent_dim):
        if i % 2 == 0:
            log_lists.append([i, i + 1])
    return log_lists


def save_loss_plot(train_loss, valid_loss, name):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'../outputs/{name}/loss.jpg')
    plt.show()


def save_model(model, name):
    model_id = "{}.pth"
    my_model = f'saved_{name}'
    model.save_weights(f'../outputs/{name}/{model_id}')
    model.save(f'../outputs/{name}/{my_model}.h5')


def get_dataset(parent_dir, sub_dir, image_size, batch_size):
    datagen = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'../input/{parent_dir}/{sub_dir}/',
        color_mode='grayscale',
        labels='inferred',
        image_size=(image_size, image_size),
        batch_size=batch_size
    )
    return datagen


def format_dataset(images, labels, paths):
    normalization_layer = tf.keras.layers.Rescaling(1/255)
    images = normalization_layer(images)
    labels = tf.cast(labels, tf.float32)
    return images, labels, tf.constant(paths)


def update_pbar(e_loss, r_loss, k_loss, pbar):
    pbar.set_postfix_str(
        f"ELBO Loss: {e_loss} - Reconstruction Loss: {r_loss} - KL Loss: {k_loss}")
    return pbar


def load_model(name):
    my_model = f'saved_{name}'
    model = tf.keras.models.load_model(f'../outputs/{name}/{my_model}.h5')
    return model
