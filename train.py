import tqdm
from engine import train, validate
from utils import *


class TrainParams:
    def __init__(self, epochs, parent_dir, name, image_size, batch_size, latent_dim, num_examples_to_generate):
        self.epochs = epochs
        self.parent_dir = parent_dir
        self.name = name
        self.image_size = image_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.num_examples_to_generate = num_examples_to_generate


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

    train_set = train_loader.map(lambda images, labels: format_dataset(images, labels, paths=train_loader.file_paths))
    test_set = test_loader.map(lambda images, labels: format_dataset(images, labels, paths=test_loader.file_paths))
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
    return pbar, e_loss_record, r_loss_record, k_loss_record


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
        params):
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
    for epoch in pbar:
        # Train the model
        reco_loss, elbo_loss, kl_loss = train(
            model, train_set
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
        if epoch % 10 == 0 or epoch == 1 or epoch == params.epochs:
            save_reconstructed_images(model, epoch, test_sample, test_label, params.epochs, params.name)
    # generate_latent_iteration(model, epoch, test_set, log_lists, name)
    print('TRAINING COMPLETE')
    return model, test_set, train_set


if __name__ == "__main__":
    check_params = TrainParams(
        parent_dir='HighCycleLowCycle_Regime',
        name='my_model',
        epochs=2,
        batch_size=16,
        image_size=128,
        latent_dim=32,
        num_examples_to_generate=16,
        # show_latent_gif=True
    )
    from main import train_a_model
    train_a_model(parent_directory='HighCycleLowCycle_Regime', sub_directory='my_model', check_parameters=check_params)
