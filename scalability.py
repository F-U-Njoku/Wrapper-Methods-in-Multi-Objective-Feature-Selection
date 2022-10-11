import time
import tracemalloc
import pandas as pd
from multi import MultiObjSFS
from skfeature.function.information_theoretical_based import MRMR

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from skfeature.function.wrapper import decision_tree_forward
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from skfeature.function.sparse_learning_based import RFS
from skfeature.function.statistical_based import gini_index
from skfeature.function.similarity_based import SPEC, reliefF
from skfeature.function.information_theoretical_based import MRMR, JMI, CMIM
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking

# Get dataset
le = LabelEncoder()
data = pd.read_csv('Synthetic_data.csv')

nb = GaussianNB()
knn = KNeighborsClassifier()
baseTree = DecisionTreeClassifier(random_state=0)


# Create various partitions for testing


def tracing_start():
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())


def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak / (1024 * 1024)
    result = round(peak, 4)
    print("Peak Size in MB - ", result)
    return result


def scalability(method, X, y):
    tracing_start()
    start = time.time()
    method.fit(X, y)
    end = time.time()
    total_time = round((end - start), 4)
    peak_mem = tracing_mem()
    return total_time, peak_mem


def experiment():
    size = 0.5
    start_feat = 201
    start_inst = 2000
    column_names = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for i in range(5):
        result_time = []
        result_mem = []
        # Constant features and changing instances
        df = data.iloc[:start_inst, :]
        X = df.loc[:, df.columns != 'label']
        y = df.loc[:, 'label']
        subset_size = round(X.shape[1] ** size)
        print(df.shape)

        mlx_sfs = SFS(estimator=baseTree,
                      k_features=subset_size,
                      forward=True,
                      scoring='accuracy',
                      cv=2)

        skl_sfs = SequentialFeatureSelector(estimator=baseTree,
                                            n_features_to_select=subset_size,
                                            direction='forward',
                                            scoring='accuracy',
                                            cv=2)

        skf_sfs = decision_tree_forward.decision_tree_forward(X.values, y.values, n_selected_features=subset_size)

        multi_sfs = MultiObjSFS(baseTree, subset_size, 2, method="sum")
        methods = [mlx_sfs, skl_sfs, multi_sfs]
        for method in methods:
            result = scalability(method, X, y)
            result_time.append(result[0])
            result_mem.append(result[1])
        print("Time:", result_time)
        print("Memory:", result_mem)
        with open('scalability.txt', 'a') as f:
            l1 = str(i)
            l2 = "\n"
            l3 = "Time:" + str(result_time)
            l4 = "Memory:" + str(result_mem)
            f.writelines([l1, l2, l3, l2, l4, l2])
        start_inst += 2000
    for i in range(4):
        result_time = []
        result_mem = []
        # Constant instances and changing features
        df = data.iloc[:, :start_feat]
        X = df.loc[:, df.columns != 'label']
        y = df.loc[:, 'label']
        subset_size = round(X.shape[1] ** size)
        print(df.shape)

        mlx_sfs = SFS(estimator=baseTree,
                      k_features=subset_size,
                      forward=True,
                      scoring='accuracy',
                      cv=2)

        skl_sfs = SequentialFeatureSelector(estimator=baseTree,
                                            n_features_to_select=subset_size,
                                            direction='forward',
                                            scoring='accuracy',
                                            cv=2)

        skf_sfs = decision_tree_forward.decision_tree_forward(X.values, y.values, n_selected_features=subset_size)

        multi_sfs = MultiObjSFS(baseTree, subset_size, 2, method="sum")
        methods = [mlx_sfs, skl_sfs, multi_sfs]
        for method in methods:
            result = scalability(method, X, y)
            result_time.append(result[0])
            result_mem.append(result[1])

        print("Time:", result_time)
        print("Memory:", result_mem)
        with open('scalability.txt', 'a') as f:
            l1 = str(i)
            l2 = "\n"
            l3 = "Time:" + str(result_time)
            l4 = "Memory:" + str(result_mem)
            f.writelines([l1, l2, l3, l2, l4, l2])
        start_feat += 200



if __name__ == '__main__':
    experiment()
