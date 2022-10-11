import time
import random
import numpy as np
import openml as oml
import pandas as pd
import stability as stb
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.naive_bayes import  GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.svm import SVC
from operator import itemgetter
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from skfeature.function.sparse_learning_based import RFS
from skfeature.function.similarity_based import SPEC, reliefF
from skfeature.function.statistical_based import gini_index, CFS
from skfeature.function.information_theoretical_based import MRMR, JMI, CMIM
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score


# - Define variables 
size = 0.5
kwargs = {'style': 0}
# - Datasets
datasets = [1015, 793, 1021, 819, 1004, 41966, 995, 41158,
            1464, 931, 40983, 841, 1061, 834, 40666, 41145]
# - Seeds
seeds = [(2 * n) + 1 for n in range(10)]
# - Classifiers
svm = SVC()
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state=0)
mlp = MLPClassifier(random_state=1, alpha=0.0)
knn = KNeighborsClassifier()
# Transformers and estimators
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
cv = StratifiedKFold(n_splits=10, shuffle=False)

# Data structures to hold results
column_names = ['stab_filt', 'svm_acc_med', 'svm_acc_avg', 'svm_mcc_med', 'svm_mcc_avg','svm_auc_med', 'svm_auc_avg',
                'nb_acc_med', 'nb_acc_avg', 'nb_mcc_med', 'nb_mcc_avg','nb_auc_med', 'nb_auc_avg',
                'dt_acc_med', 'dt_acc_avg', 'dt_mcc_med', 'dt_mcc_avg','dt_auc_med', 'dt_auc_avg',
                'knn_acc_med', 'knn_acc_avg', 'knn_mcc_med', 'knn_mcc_avg','knn_auc_med', 'knn_auc_avg']
result_dataframe = pd.DataFrame(columns=column_names)


# Retrieve data from OpenML
def fetch_data(number):
    dataset = oml.datasets.get_dataset(number)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute)
    if number == 1061:
        return X, y.astype("int"), attribute_names, len(attribute_names)
    else:
        return X, y.astype("category"), attribute_names, len(attribute_names)


# Shuffle dataset rows and columns
def randomize(X, y):
    features = list(X.columns)
    instances = list(range(len(X)))
    random.shuffle(features)
    X1 = X.loc[:, features]
    random.shuffle(instances)
    newX = X1.iloc[instances]
    newY = y.iloc[instances]
    return newX, newY, features


# Get subset of selected features
def return_subset(feature_dict, res_list, subset_list):
    all_index = list(range(len(feature_dict)))
    selected_idx = [key for key, value in feature_dict.items() if value in res_list]
    subset = [1 if index in selected_idx else 0 for index in all_index]
    subset_list.append(subset)
    return

# Get the stability score of a set o feature subsets
def get_stability(subset_list):
    stability = stb.getStability(subset_list)
    return stability

scoring = {'accuracy': 'accuracy',
           'mcc': make_scorer(matthews_corrcoef),
           'roc_auc': 'roc_auc'}

# Retrieves the performance of a classifer for given metrics
def get_performance(clf, X, y, scoring=scoring):
    scores_dict = cross_validate(clf, X, y, scoring=scoring, cv=5)
    return {metric: round(np.mean(scores), 5) for metric, scores in scores_dict.items()}

# Gets the change in predictive performance - accuracy and AUC
def accuracy_change(clf, X, y, idx, acc_change, mcc_change, auc_change):
    base = get_performance(clf, X, y)
    new = get_performance(clf, X.iloc[:, idx], y)
    acc_delta = (base['test_accuracy'] - new['test_accuracy']) / base['test_accuracy']
    mcc_delta = (base['test_mcc'] - new['test_mcc']) / base['test_mcc']
    auc_delta = (base['test_roc_auc'] - new['test_roc_auc']) / base['test_roc_auc']

    acc_change.append(round(acc_delta, 4))
    mcc_change.append(round(mcc_delta, 4))
    auc_change.append(round(auc_delta, 4))
    return


def run_experiment(datasets):
  
    for data in datasets:
        print(data)
        subset_list = []
        acc_change = {"svm":[], "nb":[], "dt":[], "knn":[]}
        mcc_change = {"svm":[], "nb":[], "dt":[], "knn":[]}
        auc_change = {"svm":[], "nb":[], "dt":[], "knn":[]}
        result_list = []
        

        X, y, attribute_names, feat_len = fetch_data(data)
        attribute_dict = {index: value for index, value in enumerate(attribute_names)}
        subset_size = round(feat_len ** size)

        for seed in seeds:
            random.seed(seed)
            X, y, features = randomize(X, y)

            # Discretize when required by the filter method
            k = max(min((X.shape[0]) / 3, 10), 2)
            disc = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
            X2 = disc.fit_transform(X.values.astype('float32'))
            y2 = le.fit_transform(y.values)
            
            idx,_,_ = MRMR.mrmr(X2, y2, n_selected_features=subset_size)
            
            #idx,_,_ = JMI.jmi(X2, y2, n_selected_features=subset_size)
            
            #idx,_,_ = CMIM.cmim(X2, y2, n_selected_features=subset_size)
            
            #score = gini_index.gini_index(X2, y2)
            #idx = gini_index.feature_ranking(score)[:subset_size]
            
            #score = reliefF.reliefF(X.values, y.values)
            #idx = list(reliefF.feature_ranking(score))[:subset_size]
            
            #score = SPEC.spec(X.values, **kwargs)
            #idx = SPEC.feature_ranking(score, **kwargs)[:subset_size]
            
            res_list = list(itemgetter(*idx)(features))
            return_subset(attribute_dict, res_list, subset_list)
            
            for clf in [(svm, "svm"), (nb, "nb"), (dt, "dt"), (knn, "knn")]:
              accuracy_change(clf[0], X, y, idx, acc_change[clf[1]], mcc_change[clf[1]], auc_change[clf[1]])
      
        
        result_list.append(get_stability(subset_list))
        print(acc_change)   
        for k,v in acc_change.items():
            result_list.append(np.median(v))
            result_list.append(np.average(v))
        for k,v in mcc_change.items():
            result_list.append(np.median(v))
            result_list.append(np.average(v))
        for k,v in auc_change.items():
            result_list.append(np.median(v))
            result_list.append(np.average(v))    
              
        result_dataframe.loc[len(result_dataframe)] = result_list
        
    print(result_dataframe.head())
    result_dataframe.to_csv("MRMR.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_experiment(datasets)

