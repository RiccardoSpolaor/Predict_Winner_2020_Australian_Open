from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def get_tree_variance_bias_decomposition(dataframe):
    n_tests = 5
    stats = np.array([])

    x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                          dataframe['Winner'], test_size=0.33, shuffle=False,
                                                          stratify=None)

    depth = range(2, 20)
    for n in depth:
        y_preds = np.array([])

        for i in range(n_tests):
            xs, ys = resample(x_train, y_train, n_samples=int(0.67 * len(y_train)))

            # train a decision tree classifier
            estimator = tree.DecisionTreeClassifier(max_depth=n)
            estimator.fit(xs, ys)

            y_pred = estimator.predict(x_valid)
            y_preds = np.column_stack([y_preds, y_pred]) if y_preds.size else y_pred

        est_bias = (y_valid - np.mean(y_preds, axis=1)) ** 2
        est_variance = np.var(y_preds, axis=1)
        est_error = (y_preds - y_valid.values.reshape(-1, 1)) ** 2

        run_stats = np.array([est_error.mean(), est_bias.mean(), est_variance.mean()])

        stats = np.column_stack([stats, run_stats]) if stats.size else run_stats

    plot_estimator_variance_bias_decomposition(depth, stats, 'Depth')


def get_bagged_tree_variance_bias_decomposition(dataframe, dt):
    n_tests = 3
    stats = np.array([])

    x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                          dataframe['Winner'], test_size=0.33, shuffle=False,
                                                          stratify=None)

    n_estimators = range(50, 301, 25)
    for n in n_estimators:
        y_preds = np.array([])

        for i in range(n_tests):
            xs, ys = resample(x_train, y_train, n_samples=int(0.67 * len(y_train)))

            # train a decision tree classifier
            estimator = BaggingClassifier(dt, n_estimators=n, n_jobs=-1)
            estimator.fit(xs, ys)

            y_pred = estimator.predict(x_valid)
            y_preds = np.column_stack([y_preds, y_pred]) if y_preds.size else y_pred

        est_bias = (y_valid - np.mean(y_preds, axis=1)) ** 2
        est_variance = np.var(y_preds, axis=1)
        est_error = (y_preds - y_valid.values.reshape(-1, 1)) ** 2

        run_stats = np.array([est_error.mean(), est_bias.mean(), est_variance.mean()])

        stats = np.column_stack([stats, run_stats]) if stats.size else run_stats

    plot_estimator_variance_bias_decomposition(n_estimators, stats, 'Estimators')


def get_boosted_tree_variance_bias_decomposition(dataframe, dt):
    n_tests = 3
    stats = np.array([])

    x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                          dataframe['Winner'], test_size=0.33, shuffle=False,
                                                          stratify=None)

    n_estimators = range(2, 21)
    for n in n_estimators:
        y_preds = np.array([])

        for i in range(n_tests):
            xs, ys = resample(x_train, y_train, n_samples=int(0.67 * len(y_train)))

            # train a decision tree classifier
            estimator = AdaBoostClassifier(dt, n_estimators=n)
            estimator.fit(xs, ys)

            y_pred = estimator.predict(x_valid)
            y_preds = np.column_stack([y_preds, y_pred]) if y_preds.size else y_pred

        est_bias = (y_valid - np.mean(y_preds, axis=1)) ** 2
        est_variance = np.var(y_preds, axis=1)
        est_error = (y_preds - y_valid.values.reshape(-1, 1)) ** 2

        run_stats = np.array([est_error.mean(), est_bias.mean(), est_variance.mean()])

        stats = np.column_stack([stats, run_stats]) if stats.size else run_stats

    plot_estimator_variance_bias_decomposition(n_estimators, stats, 'Estimators')


def get_forest_variance_bias_decomposition(dataframe, best_estimators=None):
    n_tests = 3
    stats = np.array([])

    x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                          dataframe['Winner'], test_size=0.33, shuffle=False,
                                                          stratify=None)
    if best_estimators is None:
        n_estimators = range(50, 301, 25)
        for n in n_estimators:
            y_preds = np.array([])

            for i in range(n_tests):
                xs, ys = resample(x_train, y_train, n_samples=int(0.67 * len(y_train)))

                # train a decision tree classifier
                estimator = RandomForestClassifier(n_estimators=n, random_state=13, n_jobs=-1)
                estimator.fit(xs, ys)

                y_pred = estimator.predict(x_valid)
                y_preds = np.column_stack([y_preds, y_pred]) if y_preds.size else y_pred

            est_bias = (y_valid - np.mean(y_preds, axis=1)) ** 2
            est_variance = np.var(y_preds, axis=1)
            est_error = (y_preds - y_valid.values.reshape(-1, 1)) ** 2

            run_stats = np.array([est_error.mean(), est_bias.mean(), est_variance.mean()])

            stats = np.column_stack([stats, run_stats]) if stats.size else run_stats

        plot_estimator_variance_bias_decomposition(n_estimators, stats, 'Estimators')
    else:
        depth = range(2, 20)
        for n in depth:
            y_preds = np.array([])

            for i in range(n_tests):
                xs, ys = resample(x_train, y_train, n_samples=int(0.67 * len(y_train)))

                # train a decision tree classifier
                estimator = RandomForestClassifier(n_estimators=best_estimators, max_depth=n,
                                                   random_state=13, n_jobs=-1)
                estimator.fit(xs, ys)

                y_pred = estimator.predict(x_valid)
                y_preds = np.column_stack([y_preds, y_pred]) if y_preds.size else y_pred

            est_bias = (y_valid - np.mean(y_preds, axis=1)) ** 2
            est_variance = np.var(y_preds, axis=1)
            est_error = (y_preds - y_valid.values.reshape(-1, 1)) ** 2

            run_stats = np.array([est_error.mean(), est_bias.mean(), est_variance.mean()])

            stats = np.column_stack([stats, run_stats]) if stats.size else run_stats

        plot_estimator_variance_bias_decomposition(depth, stats, 'Depth')


def plot_estimator_variance_bias_decomposition(n_inst, stats, x_label):
    fig, ax = plt.subplots(figsize=(15, 10))

    fig.suptitle('Bias$^2$-Variance Decomposition')

    ax.plot(n_inst, stats[0, :], 'o:', label='Error')
    ax.plot(n_inst, stats[1, :], 'o:', label='Bias$^2$')
    ax.plot(n_inst, stats[2, :], 'o:', label='Variance')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Error')
    ax.grid()
    ax.legend()


def plot_estimator_accuracy(ax, accuracies, hyperparameter):
    accuracies = np.array(accuracies)
    ax.plot(accuracies[:, 2], accuracies[:, 1], "x:", label="Train")
    ax.plot(accuracies[:, 2], accuracies[:, 0], "o-", label="Validation")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(hyperparameter)
    ax.grid()
    ax.legend()


def get_best_dataframe_split(dataframe):
    def get_splitted_dataframe_accuracy(index, dataframe):

        x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                              dataframe['Winner'], test_size=0.33, shuffle=False,
                                                              stratify=None)
        accuracies = []

        for max_depth in range(2, 20):
            dt = tree.DecisionTreeClassifier(max_depth=max_depth)

            dt.fit(x_train, y_train)

            train_acc = accuracy_score(y_true=y_train, y_pred=dt.predict(x_train))
            valid_acc = accuracy_score(y_true=y_valid, y_pred=dt.predict(x_valid))

            accuracies += [[valid_acc, train_acc]]

        best_valid_accuracy, best_train_accuracy = max(accuracies)
        return [[best_valid_accuracy, best_train_accuracy, index]]

    cut_dataframes = [dataframe.drop(dataframe[dataframe['csvID'].isin(range(0, i))].index.values, axis=0)
                      for i in range(0, int(max(dataframe['csvID'])))]
    accuracies = []

    for i, df in enumerate(cut_dataframes):
        accuracies += get_splitted_dataframe_accuracy(i, df)

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("Accuracy From a Certain csvID onwards")
    plot_estimator_accuracy(ax, accuracies, 'Minimum csvID')

    best_accuracy, _, best_max_csv_id = max(accuracies)
    print("Best Max csvID:", best_max_csv_id, '- Accuracy:', best_accuracy)

    return best_max_csv_id


def validate_tree_classifier(dataframe):
    def tune_tree_depth(ax):
        x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                              dataframe['Winner'],
                                                              test_size=0.33,
                                                              stratify=None, shuffle=False)

        accuracies = []

        for max_depth in range(2, 20):
            dt = tree.DecisionTreeClassifier(max_depth=max_depth)

            dt.fit(x_train, y_train)

            train_acc = accuracy_score(y_true=y_train, y_pred=dt.predict(x_train))
            valid_acc = accuracy_score(y_true=y_valid, y_pred=dt.predict(x_valid))
            # print ("Depth: {:2d} - Train Accuracy: {:.3f} - Validation Accuracy: {:.3f} ".format(
            #    max_depth,  train_acc, valid_acc))

            accuracies += [[valid_acc, train_acc, max_depth]]

        plot_estimator_accuracy(ax, accuracies, "Depth")

        best_accuracy, _, best_max_depth = max(accuracies)
        print("Best Max Depth:", best_max_depth, '- Accuracy:', best_accuracy)
        return best_max_depth

    def tune_tree_max_features(ax, depth):
        x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                              dataframe['Winner'],
                                                              test_size=0.33,
                                                              stratify=None, shuffle=False)

        accuracies = []

        for max_features in range(1, x_train.shape[1]):
            dt = tree.DecisionTreeClassifier(max_depth=depth, max_features=max_features)

            dt.fit(x_train, y_train)

            train_acc = accuracy_score(y_true=y_train, y_pred=dt.predict(x_train))
            valid_acc = accuracy_score(y_true=y_valid, y_pred=dt.predict(x_valid))
            # print ("Max Features: {:2d} - Train Accuracy: {:.3f} - Validation Accuracy: {:.3f} ".format(
            #    max_features,  train_acc, valid_acc))

            accuracies += [[valid_acc, train_acc, max_features]]

        plot_estimator_accuracy(ax, accuracies, "Max Features")

        best_accuracy, _, best_max_features = max(accuracies)
        print("Best Max Features:", best_max_features, '- Accuracy:', best_accuracy)
        return best_max_features

    fig, ax = plt.subplots(2, figsize=(15, 10))
    fig.suptitle("Accuracy based on Hyperparameters Tuning")

    hyper_parameters = {}
    hyper_parameters['depth'] = tune_tree_depth(ax[0])
    hyper_parameters['max_features'] = tune_tree_max_features(ax[1], depth=hyper_parameters['depth'])
    return hyper_parameters


def validate_bagged_tree_classifier(dataframe, tree_best_features):
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("Accuracy based on Hyperparameters Tuning")

    dt = tree.DecisionTreeClassifier(max_depth=tree_best_features['depth'],
                                     max_features=tree_best_features['max_features'])

    x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                          dataframe['Winner'],
                                                          test_size=0.33,
                                                          stratify=None, shuffle=False)

    accuracies = []

    for n_estimators in range(50, 301, 25):
        bagged_dt = BaggingClassifier(dt, n_estimators=n_estimators, n_jobs=-1)

        bagged_dt.fit(x_train, y_train)

        train_acc = accuracy_score(y_true=y_train, y_pred=bagged_dt.predict(x_train))
        valid_acc = accuracy_score(y_true=y_valid, y_pred=bagged_dt.predict(x_valid))
        # print ("Estimators: {:2d} - Train Accuracy: {:.3f} - Validation Accuracy: {:.3f} ".format(
        #    n_estimators,  train_acc, valid_acc))

        accuracies += [[valid_acc, train_acc, n_estimators]]

    plot_estimator_accuracy(ax, accuracies, "Estimators")

    best_accuracy, _, best_n_estimators = max(accuracies)
    print("Best Number of Estimators:", best_n_estimators, '- Accuracy:', best_accuracy)
    return best_n_estimators


def get_best_boosted_tree_n_estimators(dataframe, tree_best_features):
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("Accuracy based on Hyperparameters Tuning")

    dt = tree.DecisionTreeClassifier(max_depth=tree_best_features['depth'],
                                     max_features=tree_best_features['max_features'])

    x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                          dataframe['Winner'],
                                                          test_size=0.33,
                                                          stratify=None, shuffle=False)

    accuracies = []

    for n_estimators in range(2, 21):
        boosted_dt = AdaBoostClassifier(dt, n_estimators=n_estimators)

        boosted_dt.fit(x_train, y_train)

        train_acc = accuracy_score(y_true=y_train, y_pred=boosted_dt.predict(x_train))
        valid_acc = accuracy_score(y_true=y_valid, y_pred=boosted_dt.predict(x_valid))

        accuracies += [[valid_acc, train_acc, n_estimators]]

    plot_estimator_accuracy(ax, accuracies, "Estimators")

    best_accuracy, _, best_n_estimators = max(accuracies)
    print("Best Number of Estimators:", best_n_estimators, '- Accuracy:', best_accuracy)
    return best_n_estimators


def validate_forest_classifier(dataframe):
    def tune_forest_estimators(ax):
        x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                              dataframe['Winner'],
                                                              test_size=0.33,
                                                              stratify=None, shuffle=False)

        accuracies = []

        for estimators in range(50, 301, 25):
            rf = RandomForestClassifier(n_estimators=estimators, n_jobs=-1,
                                        random_state=13)  # Traininig su più core n_jobs -1
            rf.fit(x_train, y_train)

            # compute Accuracy
            train_acc = accuracy_score(y_true=y_train, y_pred=rf.predict(x_train))
            valid_acc = accuracy_score(y_true=y_valid, y_pred=rf.predict(x_valid))
            accuracies += [[valid_acc, train_acc, estimators]]
            # print ("\t Estimators: {:2d} - Validation Accuracy: {:.3f}".format(
            #    estimators, valid_acc))

        plot_estimator_accuracy(ax, accuracies, "Estimators")

        best_accuracy, _, best_estimators = max(accuracies)
        print("Best Estimators Number", best_estimators, '- Accuracy:', best_accuracy)
        return best_estimators

    def tune_forest_depth(ax, n_estimators):
        x_train, x_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                              dataframe['Winner'],
                                                              test_size=0.33,
                                                              stratify=None, shuffle=False)

        accuracies = []

        for max_depth in range(2, 20):
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        n_jobs=-1, random_state=13)
            rf.fit(x_train, y_train)

            # compute Accuracy
            train_acc = accuracy_score(y_true=y_train, y_pred=rf.predict(x_train))
            valid_acc = accuracy_score(y_true=y_valid, y_pred=rf.predict(x_valid))
            accuracies += [[valid_acc, train_acc, max_depth]]
            # print ("\t Depth: {:2d} - Validation Accuracy: {:.3f}".format(
            #    max_depth, valid_acc))

        plot_estimator_accuracy(ax, accuracies, "Depth")
        best_accuracy, _, best_depth = max(accuracies)
        print("Best Depth", best_depth, '- Accuracy:', best_accuracy)
        return best_depth

    fig, ax = plt.subplots(2, figsize=(15, 10))
    fig.suptitle("Accuracy based on Hyperparameters Tuning")

    hyper_parameters = {}
    hyper_parameters['n_estimators'] = tune_forest_estimators(ax[0])
    hyper_parameters['max_depth'] = tune_forest_depth(ax[1], n_estimators=hyper_parameters['n_estimators'])
    return hyper_parameters
