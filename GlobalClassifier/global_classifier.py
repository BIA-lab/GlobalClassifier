import logging
import pandas as pd
import importlib.resources as pkg_resources
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from joblib import Parallel, delayed
import time
import psutil
import json
import numpy as np
import yaml

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from .charts import plot_metrics, labels_per_level

class GlobalClassifier:
    def __init__(self, target_column=None, folds=2, cores=6, config_path=None):
        """
        Inicializa a classe GlobalClassifier.

        Args:
            target_column (str): Nome da coluna alvo.
            folds (int): Número de folds para validação cruzada.
            cores (int): Número de núcleos a serem usados no processamento paralelo.
            config_path (str): Caminho para o arquivo de configuração YAML (opcional).
        """
        if config_path:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.target_column = config['model']['target_column']
            self.folds = config['model']['folds']
            self.cores = config['model']['cores']
            self.classifiers = config['model']['classifiers']
            self.data_config = config['data']
            self.output_config = config['output']
        else:
            self.target_column = target_column
            self.folds = folds
            self.cores = cores if cores is not None else psutil.cpu_count(logical=True)
            self.classifiers = []
            self.data_config = {}
            self.output_config = {}

        self.label_encoder = LabelEncoder()
        
        logging.basicConfig(filename='log.log', level=logging.INFO, format='%(message)s')

        with pkg_resources.open_text('GlobalClassifier', 'model_params.json') as f:
            self.model_params = json.load(f)

    def class_distribution_by_level(self, data, level):
        """
        Conta as instâncias de cada classe no nível especificado.

        Args:
            data (pd.DataFrame): DataFrame contendo os dados.
            level (str): Nível para agrupar e contar as instâncias.

        Returns:
            pd.DataFrame: DataFrame com a contagem e porcentagem das classes.
        """
        count = pd.eval('data.groupby([level]).count()')
        name = count.columns[0]
        count = count[count[name]!=0].copy()
        count = count.drop(count.columns[1:], axis=1)
        count.columns = pd.eval('[\'Quantity\']')
        count = count.sort_values(['Quantity'], ascending=False)
        count['Percent'] = pd.eval('count[\'Quantity\'].div(count[\'Quantity\'].sum())*100')
        
        print(level+": "+ str(pd.eval('count.shape[0]')))
        return pd.eval('count')

    def preprocess(self, filepath=None, columns_drop=None, nrows=None, sep=",", levelDrop=None):
        """
        Executa o pré-processamento: binariza a coluna alvo e remove colunas indesejadas.

        Args:
            filepath (str): Caminho para o arquivo CSV.
            columns_drop (list): Lista de colunas a serem removidas.
            nrows (int): Número de linhas a serem lidas do arquivo.
            sep (str): Separador do arquivo CSV.
            levelDrop (list): Lista de níveis a serem removidos.

        Returns:
            tuple: Tupla contendo X (features), y (labels) e y_bin (labels binarizados).
        """
        print("Running pre-processing")
        logging.info("Running pre-processing")
        logging.info("Reading file")

        if filepath is None:
            filepath = self.data_config.get('filepath')
        if columns_drop is None:
            columns_drop = self.data_config.get('columns_drop', [])
        if nrows is None:
            nrows = self.data_config.get('nrows')
        sep = self.data_config.get('sep', sep)
        if levelDrop is None:
            levelDrop = self.data_config.get('levelDrop', [])

        self.data = pd.read_csv(filepath, sep=sep, dtype={self.target_column: 'category'}, nrows=nrows)
        taxonomy_final_aux = pd.eval('self.data.copy()')
        taxonomy_final_aux = taxonomy_final_aux.drop(levelDrop, axis=1).drop(columns=columns_drop)

        self.count_data = self.class_distribution_by_level(self.data, self.target_column)

        y = taxonomy_final_aux[self.target_column]
        self.y_true = y
        y_bin = label_binarize(y, classes=self.count_data.index)
        
        self.X = taxonomy_final_aux.drop(columns=[self.target_column])
        self.y = self.label_encoder.fit_transform(y)
        self.y_bin = y_bin
        logging.info("Done pre-processing")
        return self.X, self.y, self.y_bin

    def run_yaml(self):
        """
        Executa o processo de treino e avaliação dos classificadores definidos no arquivo de configuração.
        """
        self.preprocess()
        classifiers = [eval(f'{clf}()') for clf in self.classifiers]
        results = self.run(classifiers)
        
        plot_metrics(results, save_path=self.output_config.get('metrics_plot'))
        labels_per_level(results, save_path=self.output_config.get('labels_plot'))


    def run(self, classifiers):
        """
        Executa o processo de treino com folds e retorna os resultados em um objeto.

        Args:
            classifiers (list): Lista de classificadores a serem treinados.

        Returns:
            dict: Dicionário contendo scores, y_true e resultados dos classificadores.
        """
        all_scores = pd.DataFrame()
        classifiers_results = []

        for classifier in classifiers:
            classifier_name = type(classifier).__name__
            print(f"\nRunning classifier: {classifier_name}")
            logging.info(f"Running classifier: {classifier_name}")
            
            predict_df = pd.DataFrame()
            prob_df = pd.DataFrame()

            scores, predict_classifier, prob_classifier = self._model_selection(
            self.X, self.y, self.folds, classifier)
                
            all_scores = pd.concat([all_scores, scores], ignore_index=True)
                
            predict_df[f'fold_{self.folds}'] = predict_classifier['predictions']
            prob_df[f'fold_{self.folds}'] = prob_classifier.mean(axis=1)  
                
            classifier_result = {
                'classifier': classifier_name,
                'predictions': predict_df,
                'probabilities': prob_df
            }
            classifiers_results.append(classifier_result)

        print("Training completed.")
        
        all_scores.columns = ["model", "fold", "time_train", "time_test", "accuracy_train", "accuracy_test", "precision", "recall", "f1"]
        results = {
            'scores': all_scores,
            'y_true':self.y_true,
            'classifiers': classifiers_results
        }

        return results

    def _model_selection(self, X, Y, folds, classifier):
        """
        Configura o treinamento e validação cruzada com um modelo específico.

        Args:
            X (pd.DataFrame): DataFrame contendo as features.
            Y (pd.Series): Series contendo os labels.
            folds (int): Número de folds para validação cruzada.
            classifier (object): Classificador a ser treinado.

        Returns:
            tuple: Tupla contendo DataFrames com scores, previsões e probabilidades.
        """
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        skf.get_n_splits(X, Y)

        df_predict_classifier = pd.DataFrame()
        df_predict_prob_classifier = pd.DataFrame()

        classifier_name = type(classifier).__name__
        if classifier_name in self.model_params:
            params = self.model_params[classifier_name]
            if 'n_jobs' in params:
                params['n_jobs'] = self.cores
            model = classifier.set_params(**params)
        else:
            model = classifier  
        print("\nModelo:", model)
        logging.info("\nModelo: %s", model)

        out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(
            delayed(self._cross_val)(train_index, test_index, model, X, Y, folds)
            for train_index, test_index in skf.split(X, Y)
        )

        y_predicts = [d['y_predict'] for d in out]
        y_predicts_aux = [val for sublist in y_predicts for val in sublist]

        index = [d['index'] for d in out]
        index = [val for sublist in index for val in sublist]

        y_predicts_inv = self.label_encoder.inverse_transform(y_predicts_aux)

        aux = pd.DataFrame()
        aux['predictions'] = y_predicts_inv
        aux.index = index
        aux.sort_index(inplace=True)

        df_predict_classifier['predictions'] = aux['predictions']

        y_predict_probs = [d['y_predict_prob'] for d in out]
        y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

        df_predict_probs = pd.DataFrame(y_predict_probs_aux)
        df_predict_probs.index = index
        df_predict_probs.sort_index(inplace=True)
        df_predict_prob_classifier = pd.concat([df_predict_prob_classifier, df_predict_probs], axis=1)
        df_predict_prob_classifier.index = df_predict_probs.index

        scores = [d['scores'] for d in out]
        df_scores_aux = pd.DataFrame(scores)

        return df_scores_aux, df_predict_classifier, df_predict_prob_classifier

    def _cross_val(self, train_index, test_index, model, X, Y, fold):
        """
        Treina e avalia o modelo com um conjunto de treino e teste.

        Args:
            train_index (np.array): Índices do conjunto de treino.
            test_index (np.array): Índices do conjunto de teste.
            model (object): Modelo a ser treinado.
            X (pd.DataFrame): DataFrame contendo as features.
            Y (pd.Series): Series contendo os labels.
            fold (int): Número do fold atual.

        Returns:
            dict: Dicionário contendo scores, previsões, probabilidades e índices.
        """
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        countkingdom = self.class_distribution_by_level(self.data, self.target_column)
        
        y_predict = np.zeros(len(y_test))
        y_predict_prob = np.zeros((len(y_test), countkingdom.shape[0]))
        
        time_start_train = time.time()
        model.fit(X_train, y_train)
        time_elapsed_train = time.time() - time_start_train

        accuracy_train = model.score(X_train, y_train)

        time_start_test = time.time()
        y_predict = model.predict(X_test)
        time_elapsed_test = time.time() - time_start_test

        y_predict_prob = model.predict_proba(X_test)

        accuracy_test = model.score(X_test, y_test)
        precision_test = precision_score(y_test, y_predict, average='weighted', zero_division=0)
        recall_test = recall_score(y_test, y_predict, average='weighted', zero_division=0)
        f1_test = f1_score(y_test, y_predict, average='weighted', zero_division=0)
        scores = pd.Series([type(model).__name__, fold, time_elapsed_train, time_elapsed_test, accuracy_train, accuracy_test, precision_test, recall_test, f1_test])
        
        return {
            'scores': scores,
            'y_predict': y_predict,
            'y_predict_prob': y_predict_prob,
            'index': test_index 
        }