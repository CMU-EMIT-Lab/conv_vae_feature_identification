import tensorflow as tf
import matplotlib
from utils import check_dir, load_model
from model import CVAE, Encoder, Decoder

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('Tensorflow: %s' % tf.__version__)  # print version

matplotlib.style.use('ggplot')

new_model = True
parent_dir = 'HighCycleLowCycle_Regime'
sub_dir = 'my_model'


def train_a_model(parent_directory, sub_directory, check_parameters):
    from train import TrainParams, load_data, sample_inputs, initialize_training, train_model

    # Fill out the parameters we are testing if not already defined
    if not check_parameters:
        train_params = TrainParams(
            parent_dir=parent_directory,
            name=sub_directory,
            epochs=2,
            batch_size=16,
            image_size=128,
            latent_dim=32,
            num_examples_to_generate=16,
            # show_latent_gif=True
        )
    else:
        train_params = check_parameters

    check_dir(train_params.name)

    model = CVAE(
        latent_dim=train_params.latent_dim,
        batch_size=train_params.batch_size,
        image_size=train_params.image_size
    )

    test_set, train_set = load_data(train_params)
    log_lists, test_sample, test_label = sample_inputs(
        model, Encoder, Decoder, test_set, train_params
    )
    pbar, e_loss_record, r_loss_record, k_loss_record = initialize_training(train_params)
    model, test_sample, test_label = train_model(
        model=model,
        test_set=test_set,
        train_set=train_set,
        test_sample=test_sample,
        test_label=test_label,
        pbar=pbar,
        k_loss_record=k_loss_record,
        r_loss_record=r_loss_record,
        e_loss_record=e_loss_record,
        params=train_params
    )
    return model


def find_latent_dims(model):

    return


if __name__ == "__main__":
    if new_model:
        check_params = False
        cvae = train_a_model(parent_dir, sub_dir, check_params)
    else:
        cvae = load_model(sub_dir)
    exit()
