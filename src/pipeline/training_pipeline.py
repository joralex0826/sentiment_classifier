import sys
from src.logger import logging
from src.exception import CustomException
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion

class TrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            logging.info("Starting training pipeline...")

            obj=DataIngestion()

            logging.info("Starting data ingestion...")
            train_data,test_data=obj.initiate_data_ingestion()

            logging.info("Starting data transformation.")
            data_transformer = DataTransformation()
            train_arr, test_arr, preproc_path, label_enc_path = data_transformer.initiate_data_transformation(
                train_data,
                test_data
            )

            logging.info("Starting model training.")
            model_trainer = ModelTrainer()
            f1_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training completed with F1-score: {f1_score:.4f}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.run()