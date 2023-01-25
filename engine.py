import tensorflow as tf


def train(model, train_set, lr):
    for train_x, train_y, train_f in train_set:
        model.train_step(model=model, x=train_x, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    reco_loss = tf.keras.metrics.Mean()
    elbo_loss = tf.keras.metrics.Mean()
    kl_loss = tf.keras.metrics.Mean()
    return reco_loss, elbo_loss, kl_loss


def validate(model, test_set, elbo_loss, reco_loss, kl_loss):
    for test_x, test_y, test_file in test_set:
        e, l, kl = model.compute_loss(model=model, x=test_x)
        elbo_loss(e)
        reco_loss(l)
        kl_loss(kl)
    return elbo_loss, reco_loss, kl_loss
