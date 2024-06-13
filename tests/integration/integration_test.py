import pytest
from src.pipeline.train_pipeline import TrainPipeline

def test_train_pipeline_integration(mocker):  # Example
    # Mock external dependencies (e.g., data loading, model saving)
    mocker.patch("src.components.data_ingestion.DataIngestor.download_dataset")
    mocker.patch("src.components.data_factory.DataPreprocessor.preprocess")
    mocker.patch("src.components.model_training.ModelTrainer.train")
    
    configs = {}  # Provide the necessary configurations
    pipeline = TrainPipeline(configs)
    pipeline.run()
    # Assert your expectations about the pipeline's behavior
    # (e.g., that files were created, models trained, etc.)


import unittest
import pandas as pd
from src.components.data_ingestion import DataIngestor
from src.components.data_factory import DataPreprocessor
from src.components.model_training import ModelTrainer
from src.utils import read_yaml

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.configs, self.model_params = read_yaml('params.yaml')

    def test_end_to_end_pipeline(self):
        # 1. Data Ingestion
        ingestor = DataIngestor(self.configs)
        ingestor.download_dataset() 

        # 2. Data Preprocessing
        preprocessor = DataPreprocessor(self.configs)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()

        # 3. Model Training and Evaluation
        trainer = ModelTrainer(self.configs, self.model_params)
        trainer.train(X_train, y_train)
        metrics, cm, classification_rep = trainer.evaluate(X_test, y_test)

        # 4. Assertions (Examples)
        self.assertGreater(metrics['accuracy'], 0.7)  # Adjust the threshold as needed
        self.assertIsNotNone(cm)
        self.assertIsNotNone(classification_rep) 