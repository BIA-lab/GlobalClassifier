from GlobalClassifier import GlobalClassifier, plot_metrics, labels_per_level
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

classifier = GlobalClassifier(target_column="CLASS", folds=10, cores=6)


classifier.preprocess(filepath="CATH_balanced.csv", columns_drop=['Unnamed: 0'], sep=",", nrows=None)

classifiers = [DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(), AdaBoostClassifier(), SGDClassifier(), GaussianNB(), MultinomialNB()]

results = classifier.run(classifiers)

# Mostrar resultados do treinamento
plot_metrics(results)
labels_per_level(results)

# from GlobalClassifier import GlobalClassifier
# classifier = GlobalClassifier(config_path='config.yaml')
# classifier.run_yaml()
