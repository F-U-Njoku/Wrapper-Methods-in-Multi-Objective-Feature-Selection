# Import packages
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from main import randomize, fetch_data, return_subset, get_stability

import random
import numpy as np
import pandas as pda
import openml as oml
from time import time
from sklearn.svm import SVC
from operator import itemgetter
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split, StratifiedKFold

from skfeature.function.sparse_learning_based import RFS
from skfeature.function.similarity_based import SPEC, reliefF
from skfeature.function.statistical_based import gini_index, CFS
from skfeature.function.information_theoretical_based import MRMR, JMI, CMIM
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking

# - Define variables -
size = 0.5
# - Datasets
datasets = [1015, 793, 1021, 819, 1004, 41966, 995, 41158,
            1464, 931, 40983, 841, 1061, 834, 40666, 41145]
# - Seeds
seeds = [(2 * n) + 1 for n in range(10)]
# - Classifiers
svm = SVC()
nb = MultinomialNB()
dt = DecisionTreeClassifier(random_state=0)
mlp = MLPClassifier(random_state=1, alpha=0.0)
knn = KNeighborsClassifier()
# Transformers and estimators
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
cv = StratifiedKFold(n_splits=4, shuffle=False)
# - Feature selectors

# - Data structures to hold results
column_names = ['sk_s', 'svm_t_min', 'svm_t_med', 'svm_t_avg', 'svm_t_max',
                'nb_t_min', 'nb_t_med', 'nb_t_avg', 'nb_t_max',
                'dt_t_min', 'dt_t_med', 'dt_t_avg', 'dt_t_max',
                'knn_t_min', 'knn_t_med', 'knn_t_avg', 'knn_t_max',

                'svm_v_min', 'svm_v_med', 'svm_v_avg', 'svm_v_max',
                'nb_v_min', 'nb_v_med', 'nb_v_avg', 'nb_v_max',
                'dt_v_min', 'dt_v_med', 'dt_v_avg', 'dt_v_max',
                'knn_v_min', 'knn_v_med', 'knn_v_avg', 'knn_v_max']
result_dataframe = pd.DataFrame(columns=column_names)


def base_exp(x_train, x_test, y_train, y_test, classifier):
    acc = 0
    val = 0
    cv = StratifiedKFold(n_splits=4, shuffle=False)
    for train, test in cv.split(x_train, y_train):
        classifier.fit(x_train[train], y_train[train])
        y_predict = classifier.predict(x_train[test])
        acc_tmp = accuracy_score(y_train[test], y_predict)

        y_val = classifier.predict(x_test)
        val_tmp = accuracy_score(y_test, y_val)

        acc += acc_tmp
        val += val_tmp
    acc = float(acc) / 4
    val = float(val) / 4

    return acc, val


def test_subset(x_train, x_test, y_train, y_test, classifier, idx, training, validation):
    accuracies = base_exp(x_train[:, idx], x_test[:, idx], y_train, y_test, classifier)
    training.append(accuracies[0])
    validation.append(accuracies[1])
    return


def run_experiment(datasets, seeds):
    for data in datasets:
        result = {"svm": [[], []],
                  "nb": [[], []],
                  "dt": [[], []],
                  "knn": [[], []]}
        subset_list = [[]]
        result_list = []
        X, y, attribute_names, feat_len = fetch_data(data)
        attribute_dict = {index: value for index, value in enumerate(attribute_names)}
        subset_size = round(feat_len ** size)
        for seed in seeds:
            random.seed(seed)
            X, y, features = randomize(X, y)

            # Discretize
            k = max(min((X.shape[0]) / 3, 10), 2)
            disc = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
            X2 = disc.fit_transform(X.values.astype('float32'))
            y2 = le.fit_transform(y.values)

            x_train, x_test, y_train, y_test = train_test_split(X2, y2, shuffle=False, test_size=0.2)

            for train, test in cv.split(x_train, y_train):
                print(train)
                print(test)
                new_x_train, new_y_train = x_train[train], y_train[train]
                # score = reliefF.reliefF(new_x_train, new_y_train)
                # idx = list(reliefF.feature_ranking(score))[:subset_size]
                idx, _, _ = MRMR.mrmr(new_x_train, new_y_train, n_selected_features=subset_size)
                res_list = list(itemgetter(*idx)(features))
                return_subset(attribute_dict, res_list, subset_list, 0)

                for clf in [(svm, "svm"), (nb, "nb"), (dt, "dt"), (knn, "knn")]:
                    test_subset(x_train, x_test, y_train, y_test, clf[0], idx, result[clf[1]][0], result[clf[1]][1])

        result_list.append(get_stability(subset_list[0]))
        for key, value in result.items():
            print(value)
            for i in range(2):
                result_list.append(min(value[i]))
                result_list.append(np.median(value[i]))
                result_list.append(np.average(value[i]))
                result_list.append(max(value[i]))

        result_dataframe.loc[len(result_dataframe)] = result_list

    print(result_dataframe.head())
    result_dataframe.to_csv(str("results/" + "mrmr"), index=False)


if __name__ == '__main__':
    run_experiment(datasets, seeds)
