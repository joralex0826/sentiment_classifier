import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import hstack 
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_file_path = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            pipeline = Pipeline(steps=[
                ('tfidf', TfidfVectorizer())
            ])
            logging.info("TF-IDF pipeline created")
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'sentiment'
            text_column = 'cleaned_review'

            X_train = train_df[text_column]
            y_train = train_df[target_column]

            X_test = test_df[text_column]
            y_test = test_df[target_column]

            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            logging.info("Applying preprocessor object on training and testing dataframes")

            X_train_transformed = preprocessor_obj.fit_transform(X_train)

            logging.info("Transforming test data")
            X_test_transformed = preprocessor_obj.transform(X_test)

            logging.info("Combining transformed data with target variable")
            y_train_sparse = np.array(y_train_encoded).reshape(-1, 1)
            y_test_sparse = np.array(y_test_encoded).reshape(-1, 1)

            train_arr = hstack((X_train_transformed, y_train_sparse))
            test_arr = hstack((X_test_transformed, y_test_sparse))
            
            logging.info("Saving preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            save_object(
                file_path=self.data_transformation_config.label_encoder_file_path,
                obj=label_encoder
            )

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                    self.data_transformation_config.label_encoder_file_path
                    )

        except Exception as e:
            raise CustomException(e, sys)

