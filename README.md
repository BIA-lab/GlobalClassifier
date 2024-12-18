# GlobalClassifier Usage Guide

## Installation

To install the required library, run the following command:

```bash
pip install git+https://github.com/BIA-lab/GlobalClassifier.git
```

## Importing Required Libraries

```python
from GlobalClassifier import GlobalClassifier, plot_metrics, labels_per_level
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
```

## Initializing the Classifier

```python
classifier = GlobalClassifier(target_column="CLASS", folds=10, cores=6)
```

## Preprocessing the Data

```python
classifier.preprocess(filepath="CATH_balanced.csv", columns_drop=['Unnamed: 0'], sep=",", nrows=500)
```

## Defining Classifiers

```python
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    SGDClassifier(),
    GaussianNB(),
    MultinomialNB()
]
```

## Running the Classifiers

```python
results = classifier.run(classifiers)
```

## Plotting Metrics and Labels

```python
plot_metrics(results)
labels_per_level(results)
```

## Execution Using Configuration File

```python
from GlobalClassifier import GlobalClassifier
classifier = GlobalClassifier(config_path='config.yaml')
classifier.run_yaml()
```

