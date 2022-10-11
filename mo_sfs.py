import numpy as np
import openml as oml
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, cross_val_score


class MultiObjSFS:
    """
    A class that implements the Muti-objective Sequential Forward Selection feature selection method

    ...

    Attributes
    ----------
    estimator : obj
        an instance of a classification algorithm class
    number_of_features : int
        the number of features to be selected
    cv : int
        the number of folds for cross-validation

    Method
    -------
    fit(self, X, y)
        Selects a subset of features from a dataset
    """

    def __init__(self, estimator, number_of_features, cv):
        self.estimator = estimator
        self.number_of_features = number_of_features
        self.cv = cv

    def _get_score(self, X, y):
        scores = cross_validate(estimator=self.estimator, X=X, y=y, cv=self.cv,
                                scoring=('accuracy', 'roc_auc'))
        acc = np.mean(scores['test_accuracy'])
        auc = np.mean(scores['test_roc_auc'])
        return acc + auc

    def fit(self, X, y):

        if self.number_of_features < X.shape[1]:
            count = 0
            subset = []
            score = 0
            modified = True
            while count < self.number_of_features and modified:
                modified = False
                for i in range(X.shape[1]):
                    if i not in subset:
                        metric = self._get_score(X.iloc[:, subset + [i]], y)
                        if metric >= score:
                            modified = True
                            score = metric
                            max_feature = i
                            subset.append(max_feature)
                        count += 1
        return subset


def fetch_data(number):
    dataset = oml.datasets.get_dataset(number)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute)
    if number == 1061:
        return X, y.astype("int"), attribute_names, len(attribute_names)
    else:
        return X, y.astype("category"), attribute_names, len(attribute_names)


def mlxtend_acc(classifier, subset_size, forward, X, y):
    mlx_sfs = SFS(estimator=classifier,
                  k_features=subset_size,
                  forward=forward,
                  scoring='accuracy',
                  cv=10)
    mlx_sfs.fit(X, y)
    idx = mlx_sfs.k_feature_idx_
    print("Performance of selected feature subset: ", mlx_sfs.k_score_)
    return idx


def mlxtend_auc(classifier, subset_size, forward, X, y):
    mlx_sfs = SFS(estimator=classifier,
                  k_features=subset_size,
                  forward=forward,
                  scoring='roc_auc',
                  cv=10)
    mlx_sfs.fit(X, y)
    idx = mlx_sfs.k_feature_idx_
    print("Performance of selected feature subset: ", mlx_sfs.k_score_)
    return idx


def get_metric(classifier, idx, X, y):
    scores = cross_validate(estimator=classifier, X=X.iloc[:, idx], y=y, cv=10,
                            scoring=('accuracy', 'roc_auc'))
    return np.mean(scores['test_accuracy']), np.mean(scores['test_roc_auc'])


def experiment(classifier):
    size = 0.5

    column_names = ['acc_acc', 'acc_auc', 'auc_acc', 'auc_auc', 'sum_acc', 'sum_auc']
    result_dataframe = pd.DataFrame(columns=column_names)
    datasets = [1015, 793, 1021, 819, 1004, 41966, 995, 41158,
            1464, 931, 40983, 841, 1061, 834, 40666, 41145]
    for data in datasets:
        print(data)
        result = []
        X, y, attribute_names, feat_len = fetch_data(data)
        subset_size = round(feat_len ** size)
        idx_acc = list(mlxtend_acc(classifier, subset_size, True, X, y))
        idx_auc = list(mlxtend_auc(classifier, subset_size, True, X, y))
        test_class = MultiObjSFS(classifier, subset_size, 10)
        idx_sum = test_class.fit(X, y)
        result += get_metric(classifier, idx_acc, X, y)
        result += get_metric(classifier, idx_auc, X, y)
        result += get_metric(classifier, idx_sum, X, y)

        result_dataframe.loc[len(result_dataframe)] = result
        result_dataframe.to_csv("multi_nb0.csv", index=False)

    print(result_dataframe.head())
    result_dataframe.to_csv("multi_nb0.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nb = GaussianNB()
    experiment(nb)
