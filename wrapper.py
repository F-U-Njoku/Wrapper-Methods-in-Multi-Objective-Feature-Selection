import time
import random
import numpy as np
import openml as oml
import pandas as pd
import stability as stb
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC
from operator import itemgetter
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from skfeature.function.wrapper import svm_forward
from skfeature.function.wrapper import decision_tree_forward
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score


def fetch_data(number):
    dataset = oml.datasets.get_dataset(number)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute)
    if number == 1061:
        return X, y.astype("int"), attribute_names, len(attribute_names)
    else:
        return X, y.astype("category"), attribute_names, len(attribute_names)


def randomize(X, y):
    features = list(X.columns)
    instances = list(range(len(X)))
    random.shuffle(features)
    X1 = X.loc[:, features]
    random.shuffle(instances)
    newX = X1.iloc[instances]
    newY = y.iloc[instances]
    return newX, newY, features


def skfeature_exp(X, y, subset_size, classifier):
    if isinstance(classifier, DecisionTreeClassifier):
        # "=====TREE====="
        start = time.time()
        idx_tree = decision_tree_forward.decision_tree_forward(X.values, y.values, n_selected_features=subset_size)
        end = time.time()
        print("Feature selection time for TREE is: ", (end - start))
        return idx_tree

    elif isinstance(classifier, SVC):
        # "=====SVM====="
        start = time.time()
        idx_svm = svm_forward.svm_forward(X.values, y.values, n_selected_features=subset_size)
        end = time.time()
        print("Feature selection time for SVM is: ", (end - start))
        return idx_svm
    else:
        return [0, 1, 2]


def mlxtend_exp(classifier, subset_size, forward, X, y):
    mlx_sfs = SFS(estimator=classifier,
                  k_features=subset_size,
                  forward=forward,
                  scoring='accuracy',
                  cv=10)
    start = time.time()
    mlx_sfs.fit(X, y)
    end = time.time()
    print("Feature selection time is: ", (end - start))
    idx = mlx_sfs.k_feature_idx_
    print("Performance of selected feature subset: ", mlx_sfs.k_score_)
    return idx


def sklearn_exp(classifier, subset_size, direction, X, y):
    skl_sfs = SequentialFeatureSelector(estimator=classifier,
                                        n_features_to_select=subset_size,
                                        direction=direction,
                                        scoring='accuracy',
                                        cv=10)

    start = time.time()
    skl_sfs.fit(X, y)
    end = time.time()
    print("Feature selection time is: ", (end - start))
    idx = skl_sfs.get_support(indices=True)
    # print("Performance of selected feature subset: ", base_exp(X.iloc[:, idx], y, classifier))
    return idx


def return_subset(feature_dict, res_list, subset_list, id):
    all_index = list(range(len(feature_dict)))
    selected_idx = [key for key, value in feature_dict.items() if value in res_list]
    subset = [1 if index in selected_idx else 0 for index in all_index]
    # subset_list[id].append(subset) #uncomment
    return subset  # remove ubset from return statement


def accuracy_change(clf, X, y, idx, acc_change, id):
    base = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    new = cross_val_score(clf, X.iloc[:, idx], y, cv=5, scoring='accuracy')
    acc_delta = (new.mean() - base.mean()) / base.mean()
    acc_change[id].append(round(acc_delta, 4))
    return


def get_stability(subset_list):
    stability = stb.getStability(subset_list)
    return stability


def run_experiment(classifier):
    size = 0.5
    column_names = ['ml_min', 'ml_med', 'ml_avg', 'ml_max',
                    'sf_min', 'sf_med', 'sf_avg', 'sf_max',
                    'sk_min', 'sk_med', 'sk_avg', 'sk_max',
                    'ml_s', 'sf_s', 'sk_s']
    result_dataframe = pd.DataFrame(columns=column_names)
    datasets = [793]
    seeds = [(2 * n) + 1 for n in range(10)]

    for data in datasets:
        print(data)
        result = []
        subset_list = [[], [], []]
        acc_change = [[], [], []]

        X, y, attribute_names, feat_len = fetch_data(data)
        attribute_dict = {index: value for index, value in enumerate(attribute_names)}
        subset_size = round(feat_len ** size)

        for seed in seeds:
            random.seed(seed)
            X, y, features = randomize(X, y)

            # =====MLXTEND=====
            idx = list(mlxtend_exp(classifier, subset_size, True, X, y))
            print(idx)
            res_list = list(itemgetter(*idx)(features))
            return_subset(attribute_dict, res_list, [], 0)
            # accuracy_change(classifier, X, y, idx, acc_change, 0)

            # # SK-FEATURE
            # idx = list(skfeature_exp(X, y, subset_size, classifier))
            # res_list = list(itemgetter(*idx)(features))
            # return_subset(attribute_dict, res_list, subset_list, 1)
            # accuracy_change(classifier, X, y, idx, acc_change, 1)
            #
            # # SK-LEARN
            # idx = list(sklearn_exp(classifier, subset_size, 'forward', X, y))
            # res_list = list(itemgetter(*idx)(features))
            # return_subset(attribute_dict, res_list, subset_list, 2)
            # accuracy_change(classifier, X, y, idx, acc_change, 2)

        # for i in acc_change:
        #     result.append(min(i))
        #     result.append(np.median(i))
        #     result.append(np.average(i))
        #     result.append(max(i))
        #
        # for i in [subset_list]:
        #     for j in range(3):
        #         result.append(get_stability(i[j]))
        # result_dataframe.loc[len(result_dataframe)] = result
        # result_dataframe.to_csv("results/" + str(classifier), index=False)

    print(result_dataframe.head())
    result_dataframe.to_csv("results/" + str(classifier), index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nb = GaussianNB()
    knn = KNeighborsClassifier()
    baseTree = DecisionTreeClassifier(random_state=0)
    baseSVM = svm = SVC()
    # mlp = MLPClassifier(random_state=0, alpha=0.0, solver="sgd")
    run_experiment(baseTree)
