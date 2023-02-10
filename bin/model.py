"""
INFO
File: main.py
Created by: William Frieden Templeton
Date: January 27, 2023
"""

import tensorflow as tf
import numpy as np


class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, image_size, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.image_size = (image_size, image_size, 1)
        self.latent_dim = latent_dim

    # encoder
    def build(self):
        layers = [
            tf.keras.layers.InputLayer(input_shape=self.image_size),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='valid',
                                   activation=tf.keras.layers.LeakyReLU(0.01)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='valid',
                                   activation=tf.keras.layers.LeakyReLU(0.01)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim/2+self.latent_dim/2, activation=None)
        ]  # *2 because number of parameters for both mean and (raw) standard deviation
        return tf.keras.Sequential(layers)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, image_size, batch_size, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.batch_size = batch_size

    # decoder
    def build(self):
        layers = [
            tf.keras.layers.InputLayer(input_shape=(int(self.latent_dim/2),)),
            tf.keras.layers.Dense(
                int(self.image_size / 4) * int(self.image_size / 4) * self.image_size,
                activation=None
            ),
            tf.keras.layers.Reshape(
                target_shape=(int(self.image_size / 4),
                              int(self.image_size / 4),
                              self.image_size)
            ),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                            activation=tf.keras.layers.LeakyReLU(0.01)),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                            activation=tf.keras.layers.LeakyReLU(0.01)),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
        ]
        return tf.keras.Sequential(layers)


# CVAE
class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, latent_dim, image_size, batch_size):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.batch_size = batch_size
        self.encoder = Encoder(
            latent_dim=self.latent_dim, image_size=self.image_size
        ).build()
        self.decoder = Decoder(
            latent_dim=self.latent_dim, image_size=self.image_size, batch_size=self.batch_size
        ).build()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, int(self.latent_dim/2)))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var

    @staticmethod
    def re_parameterize(mean, log_var):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logit = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logit)
            return probs
        return logit

    @staticmethod
    def log_normal_pdf(sample, mean, log_var, r_axis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + log2pi),
            axis=r_axis)

    @staticmethod
    def compute_loss(model, x):
        """
        ELBO loss for VAEs:
        https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed

        We want to push the model to use similar distributions for data between images
        (I don't want num_pictures * input_dimension * input_dimension distributions)
        But also introduce a weight factor to stop the VAE from just collapsing everything to the same distribution
        in order ot preserve information. This means we need two competing functions: One, kl, to
        collapse distributions in the encoder, and another, reconstruction cross-entropy, to correct for error from
        at the decoder.

        (1) Z Sampling --
            Estimate the posterior mean & variance using model.encode(x)

        (2) Reconstruction --
            Generate reconstruction by passing Z to decoder.
            This will be a number of distributions equal to the product of input_dim * input_dim

        (3) KL divergence between Z and Normal Probability --
            pz <- p(z) log normal probability of z under a normal distribution
            qzx <- p(z|x) log normal probability of z under the encoder's distribution
            By minimizing this difference, we will increase the odds over time that any z sampled will fall under
            the normalized 0-1 distribution.


            Why Log Probability:
            Theoretical - Probabilities of two independent events A and B co-occurring together is given by P(A).P(B).
            This easily gets mapped to a sum if we use log, i.e. log(P(A)) + log(P(B)). It is thus easier to address
            the neuron firing 'events' as a linear function.
            Practical - The probability values are in [0, 1]. Hence, multiplying two or more such small numbers could
            easily lead to an underflow in a floating point precision arithmetic (e.g. consider multiplying
            0.0001*0.00001). A practical solution is to use the logs to get rid of the underflow.

            Since we’re trying to compute a loss, we need to penalize bad predictions, right? If the probability
            associated with the true class is 1.0, we need its loss to be zero. Conversely, if that probability is
            low, say, 0.01, we need its loss to be HUGE!


        (4) Probability of reconstruction --
            pxz <- p(x|z) Sample z from q (the encoder distribution) and use that z to calculate the probability
            of seeing the input x. This goes pixel by pixel then is summed.

        (5) Combine for ELBO --
            First, each image will end up with its own q.
            The KL term will push all the z distributions (Posteriors) towards the same p distribution Norm(0,1) (Prior)
            **If all the z distributions collapse to the p distribution then the network can cheat by just mapping
            everything to zero which collapses the VAE**
            The reconstruction term (pxz), forces each q to be unique and spread out so that the image can be
            reconstructed correctly. This keeps all the z distributions from collapsing onto each other.
        """
        # (1) Sampling
        mean, log_var = model.encode(x)
        z_sample = model.re_parameterize(mean, log_var)

        # (2) Reconstruction
        x_recon_logit = model.decode(z_sample)

        # (3) KL Divergence
        log_qzx = CVAE.log_normal_pdf(z_sample, mean, log_var)
        #   a) the log probability of z under the q distribution \(norm gaussian about input)
        log_pz = CVAE.log_normal_pdf(z_sample, 0., 2.)
        #   b) the log probability of z under the p distribution (norm gaussian about 0-1)
        #   c) I want KL loss to be underrepresented in this model because we would like a distribution to form
        kl = log_qzx-log_pz

        # (4) Probability of reconstruction
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon_logit, labels=x)
        log_pxz = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

        return tf.reduce_mean(kl-log_pxz), tf.reduce_mean(log_pxz), tf.reduce_mean(kl)

    @staticmethod
    def train_step(model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss, _, _ = CVAE.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
