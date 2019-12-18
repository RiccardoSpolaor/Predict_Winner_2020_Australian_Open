from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import BaggingClassifier

def plot_estimator_variance_bias_decomposition(dataframe, estimator, N_TESTS = 20):
    stats = np.array([])

    X_train, X_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                            dataframe['Winner'], test_size=0.33, shuffle = False, stratify=None)

    n_inst = range(2,100,5)
    for n in n_inst:
        y_preds = np.array([])

        for i in range(N_TESTS):
            Xs, ys = resample(X_train,y_train, n_samples=n)

            # train a decision tree classifier
            estimator.fit(Xs,ys)

            y_pred = estimator.predict(X_valid)
            y_preds = np.column_stack( [y_preds, y_pred] ) if y_preds.size else y_pred

        est_bias     = (y_valid-np.mean(y_preds,axis=1))**2
        est_variance = np.var(y_preds,axis=1)
        est_error    = (y_preds - y_valid.values.reshape(-1,1))**2

        run_stats = np.array([est_error.mean(), est_bias.mean(), est_variance.mean()])

        stats = np.column_stack( [stats, run_stats]) if stats.size else run_stats

    fig, ax = plt.subplots(figsize=(15,10))

    fig.suptitle('Bias$^2$-Variance Decomposition')

    ax.plot(n_inst,stats[0,:], 'o:', label='Error')
    ax.plot(n_inst,stats[1,:], 'o:', label='Bias$^2$')
    ax.plot(n_inst,stats[2,:], 'o:', label='Variance')
    ax.set_xlabel('Number of instances')
    ax.grid()
    ax.legend()

def plot_estimator_accuracy(ax, accuracies, hyperparameter):
    accuracies = np.array(accuracies)
    ax.plot(accuracies[:,2], accuracies[:,1], "x:", label="Train")
    ax.plot(accuracies[:,2], accuracies[:,0], "o-", label="Validation")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(hyperparameter)
    ax.grid()
    ax.legend()

def validate_tree_classifier(dataframe):
    def tune_tree_depth(ax):
        X_train, X_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                              dataframe['Winner'],
                                                              test_size=0.33,
                                                              stratify=None, shuffle=False)

        accuracies = []

        for max_depth in range(2, 20):
            dt = tree.DecisionTreeClassifier(max_depth=max_depth)

            dt.fit(X_train, y_train)

            train_acc = accuracy_score(y_true=y_train, y_pred=dt.predict(X_train))
            valid_acc = accuracy_score(y_true=y_valid, y_pred=dt.predict(X_valid))
            # print ("Depth: {:2d} - Train Accuracy: {:.3f} - Validation Accuracy: {:.3f} ".format(
            #    max_depth,  train_acc, valid_acc))

            accuracies += [[valid_acc, train_acc, max_depth]]

        plot_estimator_accuracy(ax, accuracies, "Depth")

        best_accuracy, _, best_max_depth = max(accuracies)
        print("Best Max Depth:", best_max_depth)
        return best_max_depth

    def tune_tree_max_features(ax, depth):
        X_train, X_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                              dataframe['Winner'],
                                                              test_size=0.33,
                                                              stratify=None, shuffle=False)

        accuracies = []

        for max_features in range(1, X_train.shape[1]):
            dt = tree.DecisionTreeClassifier(max_depth=depth, max_features=max_features)

            dt.fit(X_train, y_train)

            train_acc = accuracy_score(y_true=y_train, y_pred=dt.predict(X_train))
            valid_acc = accuracy_score(y_true=y_valid, y_pred=dt.predict(X_valid))
            # print ("Max Features: {:2d} - Train Accuracy: {:.3f} - Validation Accuracy: {:.3f} ".format(
            #    max_features,  train_acc, valid_acc))

            accuracies += [[valid_acc, train_acc, max_features]]

        plot_estimator_accuracy(ax, accuracies, "Max Features")

        best_accuracy, _, best_max_features = max(accuracies)
        print("Best Max Features:", best_max_features)
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

    X_train, X_valid, y_train, y_valid = train_test_split(dataframe.loc[:, dataframe.columns != 'Winner'],
                                                          dataframe['Winner'],
                                                          test_size=0.33,
                                                          stratify=None, shuffle=False)

    accuracies = []

    for n_estimators in range(50, 301, 25):
        bagged_dt = BaggingClassifier(dt, n_estimators=n_estimators, n_jobs=-1)

        bagged_dt.fit(X_train, y_train)

        train_acc = accuracy_score(y_true=y_train, y_pred=bagged_dt.predict(X_train))
        valid_acc = accuracy_score(y_true=y_valid, y_pred=bagged_dt.predict(X_valid))
        print ("Estimators: {:2d} - Train Accuracy: {:.3f} - Validation Accuracy: {:.3f} ".format(
            n_estimators,  train_acc, valid_acc))

        accuracies += [[valid_acc, train_acc, n_estimators]]

    plot_estimator_accuracy(ax, accuracies, "Estimators")

    best_n_estimators = max(accuracies)[2]
    print("Best Number of Estimators", best_n_estimators)
    return best_n_estimators