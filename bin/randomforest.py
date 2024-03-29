"""
INFO
File: main.py
Created by: William Frieden Templeton
Date: January 30, 2023
"""

import sklearn.ensemble as sk_e
import sklearn.metrics as sk_m
from bin.utils import *
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score


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
    categories = list(np.unique(all_labels))
    partitioned_encodings = {}
    partitioned_files = {}
    for c in categories:
        partitioned_encodings[c] = all_encodings[all_labels == c, :]
        partitioned_files[c] = all_files[all_labels == c]

    return all_encodings, all_labels, all_files, partitioned_encodings, partitioned_files


def random_forest(x_train, y_train, x_test, y_test, params):
    from bin.utils import save_forest

    if params.dofolds:
        # Define the number of folds
        k = 5
        # Initialize lists to store evaluation metrics
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        # Iterate over the folds
        for train_index, test_index in kf.split(x_train, y_train):
            # Split the data into train and test sets for the current fold
            X_train, X_test = x_train[train_index], x_train[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]

            # Initialize the Random Forest model
            kfoldmodel = sk_e.RandomForestClassifier(
                n_estimators=2000, max_depth=100, random_state=random.seed(1234)
            )

            # Fit the model on the training data
            kfoldmodel.fit(X_train, Y_train)
            # Predict on the test data
            y_pred = kfoldmodel.predict(X_test)

            # Calculate evaluation metrics
            accuracy = np.mean(y_pred == Y_test)
            precision = precision_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)

            # Append scores to the respective lists
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        # Calculate the mean and standard deviation of the evaluation metrics
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        mean_precision = np.mean(precision_scores)
        std_precision = np.std(precision_scores)
        mean_recall = np.mean(recall_scores)
        std_recall = np.std(recall_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        with open(f'../outputs/{params.name}/rf_ssim.txt', 'w') as f:
            f.write('%s:%s\n' % ('mean_accuracy', mean_accuracy))
            f.write('%s:%s\n' % ('std_accuracy', std_accuracy))
            f.write('%s:%s\n' % ('mean_precision', mean_precision))
            f.write('%s:%s\n' % ('std_precision', std_precision))
            f.write('%s:%s\n' % ('mean_recall', mean_recall))
            f.write('%s:%s\n' % ('std_recall', std_recall))
            f.write('%s:%s\n' % ('mean_f1', mean_f1))
            f.write('%s:%s\n' % ('std_f1', std_f1))

    # x == encodings, y == labels, f == filenames
    # We want a low depth because we're trying to identify the key features - low overall RF accuracy is unimportant
    regressor = sk_e.RandomForestClassifier(
        n_estimators=2000, max_depth=100, random_state=random.seed(1234)
    )
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Importance is Mean Decrease in Impurity
    # Mean decrease in impurity is the total decrease in node impurity
    # meaning the proportion of samples that reach that node averaged over all trees
    # which should boost the identified node to the most important spot on the tree
    find_importance = regressor.feature_importances_

    # Feature names will be the number of latent dimensions we encoded to (0 index)
    feature_names = [int(i) for i in list(np.linspace(0, x_train.shape[1]-1, x_train.shape[1]))]
    forest_importance = pd.Series(find_importance, index=feature_names).sort_values(ascending=False)

    save_forest(forest_importance, find_importance, f1, params.name)
    return forest_importance, regressor


def identify_files(classification, x_train, y_train, x_test, y_test, regressor):
    classification = classification
    # Reduce this to the classification we're interested in identifying (we want to know what made something happen,
    # not why something didn't make something happen
    train_classified = y_train == classification
    test_classified = y_test == classification

    neg_classified_train_idx = [i for i, x in enumerate(train_classified[:]) if not x]
    neg_classified_test_idx = [i for i, x in enumerate(test_classified[:]) if not x]
    neg_clas_train = x_train[neg_classified_train_idx, :]
    neg_clas_test = x_test[neg_classified_test_idx, :]

    pos_classified_train_idx = [i for i, x in enumerate(train_classified[:]) if x]
    pos_classified_test_idx = [i for i, x in enumerate(test_classified[:]) if x]
    x_train = x_train[pos_classified_train_idx, :]
    y_train = y_train[pos_classified_train_idx]
    x_test = x_test[pos_classified_test_idx, :]
    y_test = y_test[pos_classified_test_idx]

    train_predict = np.round(regressor.predict(x_train), 0)  # we want the final prediction, not a ranking
    test_predict = np.round(regressor.predict(x_test), 0)

    # Make an array that shows where it predicted correctly
    train_correct = train_predict == y_train
    test_correct = test_predict == y_test

    # Record the index where it is true (meaning it got it correct)
    pos_train_idx = [i for i, x in enumerate(train_correct[:]) if x]
    pos_test_idx = [i for i, x in enumerate(test_correct[:]) if x]
    # Make array out of correct images
    pos_enc_train = x_train[pos_train_idx, :]
    pos_enc_test = x_test[pos_test_idx, :]

    neg_train_idx = [i for i, x in enumerate(train_correct[:]) if not x]
    neg_test_idx = [i for i, x in enumerate(test_correct[:]) if not x]
    # Make array out of correct images
    neg_enc_train = x_train[neg_train_idx, :]
    neg_enc_test = x_test[neg_test_idx, :]

    positive_stacked = np.vstack((pos_enc_train, pos_enc_test))
    negative_stacked = np.vstack((neg_clas_train, neg_clas_test, neg_enc_train, neg_enc_test))

    return positive_stacked, negative_stacked


if __name__ == "__main__":
    from train import TrainParams
    from main import train_a_model

    new_model = True
    parent_dir = 'data_binary_watermark'
    sub_dir = 'my_model'

    check_params = TrainParams(
        parent_dir=parent_dir,
        name=sub_dir,
        epochs=15,
        batch_size=64,
        image_size=32,
        latent_dim=32,
        num_examples_to_generate=16,
        learning_rate=0.001
        # show_latent_gif=True
    )

    if new_model:
        # Train a model
        cvae, test_ds, train_ds = train_a_model(check_params)

        # Get arrays of encoded data from model
        train_encodings, train_labels, train_files, split_train_encodings, _ = get_encoding(cvae, train_ds)
        test_encodings, test_labels, test_files, _, _ = get_encoding(cvae, test_ds)

        # Run arrays through random forest regression to figure out if any can separate the labels
        valuable_encodings, forest_model = random_forest(
            train_encodings,
            train_labels,
            test_encodings,
            test_labels,
            check_params
        )
        show_split(split_train_encodings, valuable_encodings, forest_model, check_params)
        save_tree(forest_model, check_params)
        positive_features, negative_features = identify_files(
            0,
            train_encodings,
            train_labels,
            test_encodings,
            test_labels,
            forest_model)
        pull_key_features(positive_features, negative_features, forest_model, check_params.name)
