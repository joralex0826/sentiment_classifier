import os
import sys
from dataclasses import dataclass

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")

            train_array = train_array.tocsr()
            test_array = test_array.tocsr()

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "LogisticRegression": LogisticRegression(),
                "RandomForestClassifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier()
            }

            param_spaces = {
                "LogisticRegression": {
                    "C": [10.0],
                    "solver": ["lbfgs"]
                },
                "RandomForestClassifier": {
                    "n_estimators": [50],
                    "max_depth": [None],
                    "criterion": ["gini"]
                },
                "XGBClassifier": {
                    "n_estimators": [50],
                    "max_depth": [3],
                    "learning_rate": [0.01]
                }
            }


            logging.info("Starting model training")
            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models, param_spaces, n_trials=2)

            best_model_name = max(model_report, key=lambda x: model_report[x]['test']['f1_score'])
            best_model_score = model_report[best_model_name]['test']['f1_score']
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            report = classification_report(y_test.toarray().ravel(), predicted, output_dict=True)
            f1_score = report["weighted avg"]["f1-score"]

            return f1_score

        except Exception as e:
            raise CustomException(e, sys)
