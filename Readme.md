# Wrapper Methods for Multi-Objective Feature Selection
In machine learning, models cannot be fully described by a single metric; this is why so many metrics capturing the various aspects of the model exist and are used in their evaluation. Given this, we cannot rely on a single metric to select relevant features for building the models. We should consider multiple metrics when measuring the features’ relevance, which motivates multi-objective feature selection. Hence, we propose a Multi-Objective Sequential Forward Selection (MO-SFS) feature selection method, which uses a linear combination of any metrics to select relevant features.

To evaluate MO-SFS, we perform extensive experiments using 17 datasets (16 from OpemML and one which we generate synthetically), four classification algorithm and 10 other feature selection methods. We provide the scripts for running the experiments as well as supplimentary results from our experiments.

## Prerequisites
The libraries used in this work includes Mlxtend, Scikit-Learn, and Scikit-Feature. In the requirements.txt files are these libraries as well as their dependencies used in the experiments. All of which can simply be installed by executing the following command:
```
pip install -r requirements.txt
```
It is advisable to create a [Virtual Environment(VE)](https://docs.python.org/3/library/venv.html) for this experiments in order to avoid possible conflicts with your current setup. You can also setup a VE using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
## Execution
We have seperated the experiments into four each of which test various aspects of our work. We explain them below as well as show how to run these scripts and the results to expect 
* Filter methods - predictive performance & stability

In this work, we considered six filter method - MRMR, JMI, CMIM, Gini, ReliefF, and SPEC. The ```filter.py``` file contains the code for the experiment on filter methods. To execute for each method, we uncomment the line associated with the method between lines 138 and 151 and then set an appropriate name for the output file on line 175. Afterwards, the experiment can be executed through an IDE or on the command line with the command ```python filter.py```.

* Wrapper methods - predictive performance & stability

The ```wrapper.py``` file is used to measure the predictive performance as well as the stability of the several wrapper methods. For each run, the performance metric (accuracy or AUC) on lines [79,94,115,116,147,153,159] and classifier of interest on line 183 should be specified. Subsequently, the experiment can be executed through an IDE or on the command line with the command ```python wrapper.py```. The output file is set to the name of the classifier.


* MO-SFS - predictive performance

MO-SFS is our proposed multi-objective wrapper method which used an equal weights scalarization approach to combine the objective funtion linearly. The experiments executed by ```mo_sfs.py``` compares the predictive performance of MO-SFS against the the mono-objective sequential forward seletion for the Naive Bayes classifier (this can be modified on line 135). The experiments can also be run through an IDE or on the command line with the command ```python mo_sfs.py```.

* Wrapper methods - scalability

Finally, the scalability of wrapper methods is evaluated with the ```scalability.py``` script. Here the runtime as well as the peak memory usage are measured for the all wrapper methods considering the DecisionTree classifier (this can be modified on line 28) and the ```Synthetic_data.csv```. As in the previous, the experiments can also be run through an IDE or on the command line with the command ```python scalability.py```.
## Supplementary Results
* Accuracy change (K-Nearest Neighbour)		

| 	Data | 	CMIM   |	MLX_SFS	|	SKL_SFS |	Gini	|JMI	|MRMR	|Relief	|SPEC|
|-------|---------|--- |--- |--- |--- |--- |--- |---|
| 1015  | -1.563  |1.354|1.354|-0.856|-1.563|-0.856|-0.856|-0.856|
|  793   | 3.173   |31.707|31.707|2.877|2.877|3.173|2.877|-17.776| 
| 1021  | 0.009   |0.606|0.606|1.701|0.009|1.701|-26.39|3.065| 
|  819   | -0.232  |0.632|0.632|-0.232|-0.232|-0.232|-0.232|-30.77| 
|  1004  | 0       |-0.034|-0.034|0|0|0|-27.855|0| 
|  41966 | -1.691  |0.167|0.167|0.373|0.833|0.602|0.714|-6.713| 
|  995   | 1.268   |-0.21|-0.21|1.23|1.308|1.337|1.117|-9.54| 
|  41158 | 2.188   |9.965|9.965|0.677|1.651|3.567|1.283|-15.811| 
|  1464  | 0.846   |-0.665|-0.665|0.846|0.846|0.846|-11.6|-2.15| 
|  931   | 0.254   |7.27|7.27|0.254|0.254|-2.259|-3.71|0.254| 
|  40983 | -12.145 |-3.685|-3.685|-12.145|-12.145|-14.004|-6.837|-14.657| 
|  841   | -6.084  |1.523|1.523|-3.614|-3.379|-0.67|-6.084|1.505| 
|  1061  | 0.999   |2.704|2.704|3.438|4.213|-4.535|-10.19|-4.936| 
|  834   | 12.254  |43.762|43.762|17.724|10.351|14.855|10.73|-15.203| 
|  40666 | 7.928   |1.766|1.766|6.351|7.671|4.096|8.609|-8.298| 
|  41145 | 2.618   |36.231|36.231|-0.597|2.114|1.461|-5.838|-2.119| 
| 	| 2	      | **9**		| **9**	| 2	| 4	| 3	| 1	| 2|

* Accuracy change (Decision Tree)

| Data  | CMIM    | MLX_SFS | SKL_SFS | SKF_SFS | Gini    | JMI     | MRMR     | Relief  | SPEC     |
|-------|---------|---------|---------|---------|---------|---------|----------|---------|----------|
| 1015  | -8.461  | 0.462   | 0.462   | 0.79    | 5.888   | -8.461  | 5.888    | 3.454   | 3.454    |
| 793   | -9.365  | -0.647  | -0.312  | -0.324  | 2.637   | 2.276   | -9.851   | 2.975   | -105.927 |
| 1021  | -9.083  | -0.127  | -0.127  | -0.104  | -0.999  | -9.083  | -1.018   | -26.139 | -6.848   |
| 819   | 8.628   | 3.383   | 3.383   | 3.383   | 8.628   | 8.628   | 8.628    | 8.628   | -85.302  |
| 1004  | -0.495  | 0.32    | 0.421   | 0.37    | 0.213   | 0.071   | 0.317    | -10.519 | 1.349    |
| 41966 | 2.467   | 1.771   | 1.807   | 1.771   | 1.295   | 2.564   | 2.296    | 1.836   | -41.955  |
| 995   | 0.846   | 0.5     | 0.459   | 0.429   | 0.55    | 0.374   | 0.04     | 1.017   | -64.335  |
| 41158 | -6.866  | -2.024  | -2.252  | -3.675  | -13.772 | -14.461 | -6.877   | -9.094  | -48.253  |
| 1464  | 36.273  | 6.251   | 6.251   | 6.251   | 36.273  | 36.273  | 36.273   | -9.987  | -34.287  |
| 931   | 350.112 | 2.354   | 2.354   | 1.727   | 350.112 | 350.112 | -176.273 | 316.959 | 350.112  |
| 40983 | -90.094 | -0.373  | -0.373  | -0.371  | -90.094 | -90.094 | -90.866  | -4.122  | -90.721  |
| 841   | -2.734  | 1.349   | 1.527   | 1.248   | -3.403  | 3.273   | -6.988   | -2.734  | -16.521  |
| 1061  | 47.141  | 9.743   | 9.585   | 10.581  | 77.302  | 58.004  | -36.72   | 6.034   | 3.054    |
| 834   | -4.801  | 10.941  | 11.106  | 10.51   | 23.298  | 24.203  | 17.832   | 18.344  | -99.251  |
| 40666 | 0       | 0       | 0       | 0       | 0       | 0       | 0        | 0       | -17.679  |
| 41145 | 34.769  | 14.749  | 13.865  | 14.52   | 4.973   | 44.252  | -0.086   | -45.779 | -27.513  |
|       | 4       | 2       | 1       | 3       | 6       | **8**       | 4        | 4       | 1        |

* Accuracy change (SVM)

| Data  | CMIM   | MLX_SFS | SKL_SFS | SKF_SFS | Gini   | JMI    | MRMR    | Relief | SPEC    |
|-------|--------|---------|---------|---------|--------|--------|---------|--------|---------|
| 1015  | 6.311  | 6.311   | 6.311   | 6.311   | 0      | 6.311  | 0       | 0      | 0       |
| 793   | 5.541  | 23.39   | 23.39   | 23.39   | 23.39  | 23.39  | 5.541   | 23.39  | -31.384 |
| 1021  | 1.134  | 4.482   | 4.482   | 4.482   | 4.272  | 1.134  | 4.272   | -0.082 | 4.554   |
| 819   | 2.034  | 2.033   | 2.033   | 2.033   | 2.034  | 2.034  | 2.034   | 2.034  | -27.218 |
| 1004  | 0      | -0.017  | -0.017  | -0.017  | 0      | 0      | 0       | -1.599 | -0.051  |
| 41966 | -1.068 | 0.049   | 0.049   | 0.049   | -1.456 | -0.418 | -0.418  | -0.601 | -12.377 |
| 995   | -0.4   | -0.11   | -0.11   | -0.11   | -0.43  | -0.39  | -0.43   | -0.285 | -9.445  |
| 41158 | -1.403 | 3.115   | 3.115   | 3.115   | -4.908 | -4.651 | -2.291  | -2.461 | -19.497 |
| 1464  | 0.14   | 0.105   | 0.105   | 0.105   | 0.14   | 0.14   | 0.14    | -0.018 | -0.088  |
| 931   | 5.229  | 3.017   | 3.017   | 3.017   | 5.229  | 5.229  | -3.253  | -3.131 | 5.229   |
| 40983 | 0      | 2.128   | 2.128   | 2.128   | 0      | 0      | 0       | 3.544  | 0       |
| 841   | -5.041 | 4.647   | 4.647   | 4.647   | -8.702 | 4.634  | -8.959  | -5.041 | -13.499 |
| 1061  | -0.979 | 2.003   | 2.003   | 2.003   | -0.102 | -0.742 | 0       | 0      | -0.199  |
| 834   | 7.867  | 28.84   | 28.84   | 28.84   | 20.647 | 11.957 | 16.216  | 14.445 | -7.667  |
| 40666 | -9.952 | 6.436   | 6.436   | 6.436   | -7.158 | -9.775 | 5.302   | -9.744 | -1.096  |
| 41145 | -1.51  | -       | -       | -       | 4.831  | -1.544 | -13.233 | -0.637 | -12.863 |
|       | 5      | **9**   | **9**   | **9**   | 5      | 6      | 3       | 3      | 2       |

* AUC change (Naive Bayes)	

| Data  | CMIM    | MLX_SFS | SKL_SFS | Gini    | JMI     | MRMR    | Relief  | SPEC    |   |
|-------|---------|---------|---------|---------|---------|---------|---------|---------|---|
| 1015  | -23.932 | -0.847  | -0.847  | -6.037  | -23.932 | -6.037  | -6.037  | -6.037  |   |
| 793   | -7.931  | 4.624   | 4.624   | -13.317 | -13.317 | -7.931  | -13.317 | -82.285 |   |
| 1021  | 36.857  | 3.065   | 3.065   | 64.63   | 36.857  | 64.63   | -79.506 | 48.612  |   |
| 819   | -0.937  | -0.232  | -0.232  | -0.937  | -0.937  | -0.937  | -0.937  | -69.911 |   |
| 1004  | 0       | -0.001  | -0.001  | 0       | 0       | 0       | -83.491 | 0       |   |
| 41966 | -11.46  | 1.096   | 1.096   | 3.277   | 5.684   | 6.129   | 3.639   | -24.459 |   |
| 995   | 25.594  | 1.472   | 1.472   | 25.933  | 28.342  | 32.006  | 33.282  | -37.57  |   |
| 41158 | 6.147   | 7.682   | 7.682   | 2.036   | 4.414   | 11.604  | 2.192   | -48.413 |   |
| 1464  | 3.611   | 0.846   | 0.846   | 3.611   | 3.611   | 3.611   | -16.356 | -100    |   |
| 931   | 10.17   | 0.254   | 0.254   | 10.17   | 10.17   | -42.793 | -55.194 | 10.17   |   |
| 40983 | -100    | -4.678  | -4.678  | -100    | -100    | -100    | -98.07  | -100    |   |
| 841   | -25.74  | 4.366   | 4.366   | 11.789  | 4.414   | -1.911  | -25.74  | 2.499   |   |
| 1061  | -8.461  | 3.341   | 3.341   | 9.081   | -11.12  | -7.73   | -3.227  | -4.592  |   |
| 834   | 36.314  | 18.815  | 18.815  | 84.634  | 33.132  | 57.136  | 35.611  | -52.705 |   |
| 40666 | 39.619  | 8.931   | 8.931   | 15.034  | 36.478  | 1.626   | 47.452  | -34.48  |   |
| 41145 | 2.436   | 5.841   | 5.841   | -1.823  | -0.832  | 5.157   | -27.342 | -7.951  |   |
| Count | 3       | 5       | 5       | **7**       | 3       | 5       | 2       | 2       |   |

* AUC change (Decision Tree)	

| Data  | CMIM    | MLX_SFS | SKL_SFS | SKF_SFS | Gini   | JMI     | MRMR    | Relief | SPEC    |
|-------|---------|---------|---------|---------|--------|---------|---------|--------|---------|
| 1015  | -13.207 | 1.413   | 1.413   | 2.435   | -3.646 | -13.207 | -3.646  | -3.646 | -3.646  |
| 793   | 7.593   | -0.541  | -0.84   | -0.293  | 22.313 | 22.313  | 7.593   | 22.313 | -36.703 |
| 1021  | -2.159  | 0.866   | 0.866   | 0.203   | 2.834  | -2.159  | 2.834   | 0.179  | -2.314  |
| 819   | 2.284   | 8.359   | 8.359   | 8.359   | 2.284  | 2.284   | 2.284   | 2.284  | -28.263 |
| 1004  | 0       | 0.697   | 0.892   | 0.678   | 0      | 0       | 0       | -0.105 | 0       |
| 41966 | -0.069  | 1.857   | 1.807   | 1.771   | -0.049 | -0.018  | -0.012  | -0.015 | -4.961  |
| 995   | -0.378  | 1.28    | 1.257   | 0.981   | -0.494 | -0.466  | -0.35   | -0.677 | -15.654 |
| 41158 | -1.648  | -3.657  | -4.011  | -4.101  | -4.312 | -3.611  | -1.892  | -1.757 | -17.014 |
| 1464  | -6.792  | 5.305   | 5.305   | 4.933   | -6.792 | -6.792  | -6.792  | -4.82  | -1.046  |
| 931   | 3.611   | 2.273   | 2.273   | 1.644   | 3.611  | 3.611   | -1.606  | -3.659 | 3.611   |
| 40983 | -38.01  | -1.828  | -1.828  | -1.808  | -38.01 | -38.01  | -37.265 | -0.139 | -36.861 |
| 841   | -0.752  | 1.361   | 1.541   | 1.257   | -2.084 | 0.456   | -12.734 | -0.752 | -9.941  |
| 1061  | 12.052  | 9.269   | 12.191  | 15.267  | 10.468 | 18.234  | 0.734   | 0.179  | 8.787   |
| 834   | 8.359   | 10.457  | 10.989  | 10.459  | 26.51  | 17.999  | 22.237  | 18.015 | -14.591 |
| 40666 | 1.897   | 0       | 0       | 0       | -0.671 | 0.079   | 3.325   | -0.622 | -4.31   |
| 41145 | -2.033  | 14.65   | 13.941  | 14.52   | 3.465  | -2.053  | -13.914 | -0.842 | -11.122 |
| Count | 2       | **5 (7)**   | **4(7)**    | **2(7)**    | 3      | 2       | 2       | 2      | 0       |
