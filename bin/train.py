"""
INFO
File: main.py
Created by: William Frieden Templeton
Date: January 27, 2023
"""

import tqdm
from bin.engine import train, validate
from bin.utils import *
from bin.model import *
from bin.settings import TrainParams
from sklearn.model_selection import KFold

def load_data(params):
    train_loader = get_dataset(
        parent_dir=params.parent_dir,
        sub_dir='train',
        image_size=params.image_size,
        batch_size=params.batch_size
    )

    test_loader = get_dataset(
        parent_dir=params.parent_dir,
        sub_dir='val',
        image_size=params.image_size,
        batch_size=params.batch_size
    )

    train_set = train_loader.map(
        lambda images, labels: format_dataset(images, labels, paths=train_loader.file_paths)
    )
    test_set = test_loader.map(
        lambda images, labels: format_dataset(images, labels, paths=test_loader.file_paths)
    )
    return test_set, train_set


def sample_inputs(model, encoder, decoder, test_set, params):
    # Save the Starting Point
    log_lists = start_llist(params.latent_dim)

    # Saving initial images
    test_sample, test_label = get_sample(
        params.batch_size,
        params.num_examples_to_generate,
        test_set,
        params.name,
        params.epochs,
        model
    )

    # Saving model structure
    printer(
        encoder(
            latent_dim=params.latent_dim, image_size=params.image_size
        ).build(), branch="Encoder",
        name=params.name
    )
    printer(
        decoder(
            latent_dim=params.latent_dim, image_size=params.image_size, batch_size=params.batch_size
        ).build(), branch="Decoder",
        name=params.name
    )
    return log_lists, test_sample, test_label


def initialize_training(params):
    # Set up training loop
    pbar = tqdm.tqdm(range(1, params.epochs+1))
    e_loss_record = []
    r_loss_record = []
    k_loss_record = []
    ssim_record = []
    return pbar, e_loss_record, r_loss_record, k_loss_record, ssim_record


def train_model(
        model,
        test_set,
        train_set,
        test_sample,
        test_label,
        pbar,
        k_loss_record,
        r_loss_record,
        e_loss_record,
        ssim_record,
        params,
        kf,
        k):
    """
    Inputs:
    model
    test_set, train_set
    test_sample, train_sample
    pbar
    r_loss_record, e_loss_record, k_loss_record
    name
    """
    print(f"Training Started on {params.parent_dir} Dataset")
    ssim_scores = []
    mean_ssim = None
    std_ssim = None
    if kf:
        datagen = get_dataset(parent_dir=params.parent_dir, sub_dir='train',
                              image_size=params.image_size, batch_size=params.batch_size)
        images = np.concatenate([x for x, _ in datagen], axis=0)
        labels = np.concatenate([y for _, y in datagen], axis=0)

        for fold, (train_indices, val_indices) in enumerate(kf.split(images)):
            print(f"Training Fold {fold + 1}/{k}")
            train_images_fold, train_labels_fold = images[train_indices], labels[train_indices]
            val_images_fold, val_labels_fold = images[val_indices], labels[val_indices]

            train_dataset = tf.data.Dataset.from_tensor_slices((train_images_fold, train_labels_fold))
            train_dataset = train_dataset.batch(params.batch_size)

            val_dataset = tf.data.Dataset.from_tensor_slices((val_images_fold, val_labels_fold))
            val_dataset = val_dataset.batch(params.batch_size)

            # I am inputting train_labels twice because I have a vestigial function that I don't want to remove
            k_train_set = train_dataset.map(
                lambda images, labels: format_dataset(train_images_fold, train_labels_fold, val_labels_fold)
            )
            k_val_set = val_dataset.map(
                lambda images, labels: format_dataset(val_images_fold, val_labels_fold, val_labels_fold)
            )

            for epoch in pbar:
                reco_loss, elbo_loss, kl_loss = train(
                    model, k_train_set, params.learning_rate
                )

                # Validate the model
                elbo_loss, reco_loss, kl_loss = validate(
                    model, k_val_set, reco_loss, elbo_loss, kl_loss
                )

                # Record the loss
                r_loss = -reco_loss.result()
                e_loss = elbo_loss.result()
                k_loss = kl_loss.result()

                r_loss_record.append(r_loss)
                e_loss_record.append(e_loss)
                k_loss_record.append(k_loss)

                update_pbar(e_loss, r_loss, k_loss, pbar)

            val_images_fold = np.concatenate([x for x, _, _ in k_val_set], axis=0)
            import ipdb; ipdb.set_trace()
            mean, log_var = model.encode(val_images_fold)
            z = model.re_parameterize(mean, log_var)
            val_reconstructions = model.sample(
                z)

            for i in range(len(val_images_fold)):
                ssim = calculate_ssim(val_images_fold[i], val_reconstructions[i])
                ssim_scores.append(ssim)

        mean_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        np.savetxt(f'../outputs/{params.name}/ssim_scores.txt', ssim_scores)


    for epoch in pbar:
        # Train the model


        reco_loss, elbo_loss, kl_loss = train(
            model, train_set, params.learning_rate, kf
        )

        # Validate the model
        elbo_loss, reco_loss, kl_loss = validate(
            model, test_set, reco_loss, elbo_loss, kl_loss
        )

        # Record the loss
        r_loss = -reco_loss.result()
        e_loss = elbo_loss.result()
        k_loss = kl_loss.result()

        r_loss_record.append(r_loss)
        e_loss_record.append(e_loss)
        k_loss_record.append(k_loss)

        update_pbar(e_loss, r_loss, k_loss, pbar)

        # make an example of the reconstruction
        if epoch % 25 == 0 or epoch == 1 or epoch == params.epochs:
            reconimg = save_reconstructed_images(model, epoch, test_sample, test_label, params.epochs, params.name)
    # generate_latent_iteration(model, epoch, test_set, log_lists, name)
    save_loss_plot(e_loss_record, r_loss_record, params.name)
    np.savetxt('r_loss_record.txt', r_loss_record)
    np.savetxt('e_loss_record.txt', e_loss_record)
    print('TRAINING COMPLETE')
    return model, test_set, train_set, ssim_scores, mean_ssim, std_ssim


def train_a_model(train_params):
    check_dir('outputs', False, train_params.name)

    model = CVAE(
        latent_dim=train_params.latent_dim,
        batch_size=train_params.batch_size,
        image_size=train_params.image_size
    )

    if train_params.dofolds:
        kf = KFold(n_splits=train_params.kfolds, shuffle=True, random_state=42)
    else:
        kf = False


    test_set, train_set = load_data(train_params)
    log_lists, test_sample, test_label = sample_inputs(
        model, Encoder, Decoder, test_set, train_params
    )
    pbar, e_loss_record, r_loss_record, k_loss_record, ssim_record = initialize_training(train_params)

    model, test_sample, test_label, ssim_scores, mean_ssim, std_ssim = train_model(
        model=model,
        test_set=test_set,
        train_set=train_set,
        test_sample=test_sample,
        test_label=test_label,
        pbar=pbar,
        k_loss_record=k_loss_record,
        r_loss_record=r_loss_record,
        e_loss_record=e_loss_record,
        ssim_record=ssim_record,
        params=train_params,
        kf=kf,
        k=train_params.kfolds
    )
    return model, test_set, train_set, ssim_scores, mean_ssim, std_ssim


if __name__ == "__main__":
    check_params = TrainParams(
        parent_dir='HighCycleLowCycle_Regime',
        name='my_model',
        epochs=2,
        batch_size=16,
        image_size=128,
        latent_dim=32,
        learning_rate=0.001,
    )
    train_a_model(train_params=check_params)
