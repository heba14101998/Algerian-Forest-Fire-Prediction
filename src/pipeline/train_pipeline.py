import sys
import pandas as pd
from src.components.data_ingestion import DataIngestor
from src.components.data_factory import DataPreprocessor
from src.components.model_training import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml


class TrainPipeline:
    def __init__(self):
        self.configs, self.model_params = read_yaml('params.yaml')

    def run_pipeline(self):
        try:
            logging.info(f"Data Ingestion")
            data_ingestion = DataIngestor(self.configs)

            logging.info(f"Data Preparation")
            data_process = DataPreprocessor(self.configs)
            X_train, X_test, y_train, y_test = data_process.preprocess()

            logging.info(f"Training Model")
            model = ModelTrainer(self.configs, self.model_params)
            model.train(X_train, y_train)

            logging.info(f"Calculate performance on training dataset")
            metrics, cm, classification_rep = model.evaluate(X_train, y_train)
            
            logging.info(f"Calculate performance on testing dataset")
            metrics, cm, classification_rep = model.evaluate(X_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Error running the training pipeline: {e}")
        raise e