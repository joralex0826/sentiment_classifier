import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, clean_text


class InferencePipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            label_encoder_path = "artifacts/label_encoder.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            label_encoder = load_object(file_path=label_encoder_path)
            features['cleaned_review'] = features['review'].apply(clean_text)
            features.drop(columns=['review'], inplace=True)
            data = preprocessor.transform(features['cleaned_review'])
            preds = model.predict(data)
            preds = preds.astype(int)
            preds = label_encoder.inverse_transform(preds)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, review: str):
        self.review = review

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "review": [self.review]
            }
            df = pd.DataFrame(custom_data_input_dict)
            return df
        
        except Exception as e:
            raise CustomException(e, sys)