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
* Wrapper methods - predictive performance & stability
* MO-SFS - predictive performance
* Wrapper methods - scalability


