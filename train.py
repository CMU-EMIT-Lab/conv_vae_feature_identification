import os
import sys
import tqdm
import matplotlib
from model import CVAE, Encoder, Decoder
from engine import train, validate
from utils import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('Tensorflow: %s' % tf.__version__)  # print version


matplotlib.style.use('ggplot')
main_dir = os.path.dirname(sys.path[0])

# initialize the model
name = 'my_model'
parent_dir = 'HighCycleLowCycle_Regime'

# check if your dir exists and make along directory if not (CVAE > Output > name)
check_dir(name)

# set the learning parameters
epochs = 2
batch_size = 16
image_size = 128
latent_dim = 32
num_examples_to_generate = 16
show_latent_gif = True

train_loader = get_dataset(parent_dir=parent_dir, sub_dir='train', image_size=image_size, batch_size=batch_size)
test_loader = get_dataset(parent_dir=parent_dir, sub_dir='val', image_size=image_size, batch_size=batch_size)

train_set = train_loader.map(lambda images, labels: format_dataset(images, labels, paths=train_loader.file_paths))
test_set = test_loader.map(lambda images, labels: format_dataset(images, labels, paths=test_loader.file_paths))

# Save the Starting Point
log_lists = start_llist(latent_dim)
model = CVAE(latent_dim, image_size, batch_size)

# Saving initial images
test_sample, test_label = get_sample(batch_size, num_examples_to_generate, test_set, name, epochs, model)

# Saving model structure
printer(
    Encoder(latent_dim=latent_dim, image_size=image_size).build(), branch="Encoder",
    name=name
)
printer(
    Decoder(latent_dim=latent_dim, image_size=image_size, batch_size=batch_size).build(), branch="Decoder",
    name=name
)

# Set up training loop
pbar = tqdm.tqdm(range(1, epochs+1))
e_loss_record = []
r_loss_record = []
k_loss_record = []

print(f"Training Started on {parent_dir} Dataset")
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
    if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
        save_reconstructed_images(model, epoch, test_sample, test_label, epochs, name)

# generate_latent_iteration(model, epoch, test_set, log_lists, name)
save_model(model, name)
print('TRAINING COMPLETE')

if __name__ == "__main__":
    exit()
