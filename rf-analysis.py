import numpy as np
import sklearn.ensemble as sk_e
import pandas as pd
import matplotlib.pyplot as plt


def get_encoding(model, ds):
    encoding, all_encodings, label, all_labels, all_files = [], [], [], [], []
    # Encode each batch of the train or test set, save the encoding, label, and filenames
    for img, lab, file in ds:
        mean, log_var = model.encode(img)  # encode the batch of images
        encoding.append(model.re_parameterize(mean, log_var))  # append the encoding into a list
        label.append(lab.numpy())  # append the corresponding label into a list
        all_encodings = np.concatenate(encoding, axis=0)  # stack list into an array (n-images by n-latent_dimensions)
        all_labels = np.concatenate(label, axis=0)  # stack labels into an array
        all_files = file.numpy()  # stack files into an array

    # At this point, we should have three arrays:
    #    all_encodings - an array of shape [n-images x n-latent_dimensions]
    #    all_labels - an array of shape [n-images x 1]
    #    all_files - an array of shape [n-images x 1]

    return all_encodings, all_labels, all_files


def random_forest(x_train, y_train, x_test, y_test):
    # x == encodings, y == labels, f == filenames
    regressor = sk_e.RandomForestRegressor()  # You can edit the regressor if you want, running with default for now
    regressor.fit(x_train, y_train)
    find_importance = regressor.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)

    feature_names = [int(i) for i in list(np.linspace(0, x_train.shape[1], x_train.shape[1]-1))]
    forest_importance = pd.Series(find_importance, index=feature_names)

    fig, ax = plt.subplots()
    forest_importance.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importance using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    return


if __name__ == "__main__":
    import sys
    from train import TrainParams
    from main import train_a_model

    new_model = True
    parent_dir = 'HighCycleLowCycleNoBorder_Regime'
    sub_dir = 'my_model'

    check_params = TrainParams(
        parent_dir=parent_dir,
        name=sub_dir,
        epochs=1,
        batch_size=16,
        image_size=32,
        latent_dim=32,
        num_examples_to_generate=16,
        # show_latent_gif=True
    )

    if new_model:
        # Train a model
        cvae, test_ds, train_ds = train_a_model(check_params)

        # Get arrays of encoded data from model
        train_encodings, train_labels, train_files = get_encoding(cvae, train_ds)
        test_encodings, test_labels, test_files = get_encoding(cvae, test_ds)

        # Run arrays through random forest regression to figure out if any can separate the labels
        random_forest(train_encodings, train_labels, test_encodings, test_labels)

    else:
        sys.exit()
