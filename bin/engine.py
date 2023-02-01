import tensorflow as tf


def train(model, train_set, lr):
    for train_x, train_y, train_f in train_set:
        model.train_step(model=model, x=train_x, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    reco_loss = tf.keras.metrics.Mean()
    elbo_loss = tf.keras.metrics.Mean()
    kl_loss = tf.keras.metrics.Mean()
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    return reco_loss, elbo_loss, kl_loss, train_loss, val_loss


def calculate_train_loss(model, train_set, train_loss):
    for train_x, _, _ in train_set:
        loss, _, _ = model.compute_loss(model=model, x=train_x)
        train_loss(loss)
    return train_loss


def validate(model, test_set, elbo_loss, reco_loss, kl_loss, val_loss):
    for test_x, test_y, test_file in test_set:
        e, l, kl = model.compute_loss(model=model, x=test_x)
        elbo_loss(e)
        reco_loss(l)
        kl_loss(kl)
        val_loss(e)
    return elbo_loss, reco_loss, kl_loss, val_loss
