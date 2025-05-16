import os
import sys

import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from src.logger import logging

import optuna
from scipy.sparse import issparse
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

def save_object(file_path, obj):
    """
    Save the object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param_spaces, n_trials=20):
    try:
        report = {}

        # Convert y_train and y_test once to dense if sparse
        y_train_dense = y_train.toarray().ravel() if issparse(y_train) else y_train.ravel()
        y_test_dense = y_test.toarray().ravel() if issparse(y_test) else y_test.ravel()

        for name, model in models.items():
            def objective(trial):
                params = {key: trial.suggest_categorical(key, values) for key, values in param_spaces[name].items()}
                model.set_params(**params)
                return cross_val_score(model, X_train, y_train_dense, cv=3, scoring='f1').mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            model.set_params(**best_params)
            model.fit(X_train, y_train_dense)

            # Train predictions and metrics
            y_train_pred = model.predict(X_train)
            f1_train = f1_score(y_train_dense, y_train_pred)
            precision_train = precision_score(y_train_dense, y_train_pred)
            recall_train = recall_score(y_train_dense, y_train_pred)

            # Test predictions and metrics
            y_test_pred = model.predict(X_test)
            f1_test = f1_score(y_test_dense, y_test_pred)
            precision_test = precision_score(y_test_dense, y_test_pred)
            recall_test = recall_score(y_test_dense, y_test_pred)

            report[name] = {
                'best_params': best_params,
                'train': {
                    'f1_score': f1_train,
                    'precision': precision_train,
                    'recall': recall_train
                },
                'test': {
                    'f1_score': f1_test,
                    'precision': precision_test,
                    'recall': recall_test
                }
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


