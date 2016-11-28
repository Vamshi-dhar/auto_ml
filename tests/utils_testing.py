import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from auto_ml import Predictor

def get_boston_regression_dataset():
    boston = load_boston()
    df_boston = pd.DataFrame(boston.data)
    df_boston.columns = boston.feature_names
    df_boston['MEDV'] = boston['target']
    df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.33, random_state=42)
    return df_boston_train, df_boston_test

def get_titanic_binary_classification_dataset(basic=True):
    try:
        df_titanic = pd.read_csv(os.path.join('tests', 'titanic.csv'))
    except Exception as e:
        print('Error')
        print(e)
        dataset_url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv'
        df_titanic = pd.read_csv(dataset_url)
        # Do not write the index that pandas automatically creates
        df_titanic.to_csv(os.path.join('tests', 'titanic.csv'), index=False)
    # print(df_titanic)
    df_titanic = df_titanic.drop(['boat', 'body'], axis=1)

    if basic == True:
        df_titanic = df_titanic.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)

    df_titanic_train, df_titanic_test = train_test_split(df_titanic, test_size=0.33, random_state=42)
    return df_titanic_train, df_titanic_test


def train_basic_binary_classifier(df_titanic_train):
    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train)

    return ml_predictor


def train_basic_regressor(df_boston_train):
    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, verbose=False)
    return ml_predictor

def calculate_rmse(actuals, preds):
    return mean_squared_error(actuals, preds)**0.5 * -1

def calculate_brier_score_loss(actuals, probas):
    return -1 * brier_score_loss(actuals, probas)


def make_titanic_ensemble(df_titanic_train):
    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ensemble_list = [
        {
            'column_descriptions': column_descriptions
            , 'name': 'LogisticRegression'
            , 'model_names': ['LogisticRegression']
        }

        , {
            'column_descriptions': column_descriptions
            , 'name': 'GradientBoostingClassifier'
            , 'model_names': ['GradientBoostingClassifier']
        }

        # , {
        #     'column_descriptions': column_descriptions
        #     , 'name': 'Perceptron'
        #     , 'model_names': ['Perceptron']
        # }

        , {
            'column_descriptions': column_descriptions
            , 'name': 'RandomForestClassifier'
            , 'model_names': ['RandomForestClassifier']
        }

        # , {
        #     'column_descriptions': column_descriptions
        #     , 'name': 'PassiveAggressiveClassifier'
        #     , 'model_names': ['PassiveAggressiveClassifier']
        # }

        # , {
        #     'column_descriptions': column_descriptions
        #     , 'name': 'RidgeClassifier'
        #     , 'model_names': ['RidgeClassifier']
        # }

        , {
            'column_descriptions': column_descriptions
            , 'name': 'AdaBoostClassifier'
            , 'model_names': ['AdaBoostClassifier']
        }

        , {
            'column_descriptions': column_descriptions
            , 'name': 'ExtraTreesClassifier'
            , 'model_names': ['ExtraTreesClassifier']
        }
    ]

    ml_predictor.train_ensemble(data=df_titanic_train, ensemble_training_list=ensemble_list)

    return ml_predictor

