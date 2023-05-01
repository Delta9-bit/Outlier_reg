# Non-parametric Outlier and Influential Points Regularization

This is the last version of our *multi-stage* inflential point detection algorithm. 
Install via typical `pip install` commands is not available yet. To use this code, download the source file and import. For a detailed installation guide, 
please read the instructions below. 

From the files in Outliers_reg/Code/ you have access to all the source code. The main function is the `multistage_npor` function.

⚠️ The `multispage_npor` function internally scales the features with MinMax scaling. Features **do not** need to be scaled.

---

## Quick Installation Guide

1. download the source code from GitHub (either copy-paste the code in an empty .py file and save it or clone the repository)
2. access the source file from your current project by changing the current working directory: `os.chdir(r"path_to_source_code.py")`
>> make sure to import os before using* `os.chdir`
3. you can now import the kstage function vith a standrad import: `import npor` **or** `from npor import multistage_npor`

---

## npor.multistage_npor(*features*, *labels*, *classifier*, *centroid="mean"*, *kernel_params={}*, *k=10*, *q=0.99*, *steps=10*)

### Args

- **features** : (n*p) numpy ndarray of features (n samples, p features).
- **labels** : (n*1) numpy array of labels.
- **classifier** : any classifier from SkLearn which supports the support_vectors, support_,intercept_ and dual_coef_ methods.
- **centroid** : str. centroid estimation method. {*"mode"*, *"mean"*, *"median"*}.
>> default : "median"
- **kernel_params** (optional) : required if style="mode". Dictionnary of parameters for the Kernel Density Estimation. See [Kernel Density Estimation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html).
- **k** : int. Number of outliers to search for in each class. Total number of outliers to search for is k * # classes.
>> default : 10
- **q** : float. [0 : 1]. Quantile threhsold.
>> default : 0.99
- **steps** : int. Numbers of bootstrap re-sampling for the quantile estimation.
>> default : 10

### Returns

Pandas dataframe with the index and influence values for influential points.
