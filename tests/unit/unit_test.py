import pytest
from src.components.data_factory import DataPreprocessor
# ... import other components you need to test ...

def test_data_preprocessing_clean_data(mocker):  # Example
    # Mock loading data from CSV
    mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    mocker.patch("src.components.data_factory.pd.read_csv", return_value=mock_df)
    
    configs = {"raw_data_dir": "./data/raw", 
               "data_file_name": "Algerian_forest_fires_dataset.csv",
               "processed_data_dir": "./data/processed",
               "target_column": "Classes",
               "scaling_method": "minmax"}
    
    preprocessor = DataPreprocessor(configs)
    cleaned_df = preprocessor.clean_data(mock_df)
    # Assert your expected data cleaning results
    assert cleaned_df.shape == (3, 3)  # Or whatever your expected shape is
    # ... Add more assertions as needed ...


import unittest
import pandas as pd
from src.components.data_factory import DataPreprocessor
from src.utils import read_yaml

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.configs, _ = read_yaml('params.yaml')
        self.preprocessor = DataPreprocessor(self.configs)

    def test_clean_data(self):
        # Load a sample dataset (replace with your actual data)
        sample_data = pd.DataFrame({
            'Temperature': [10, 20, 30],
            'RH': [50, 60, 70],
            'wind': [1, 2, 3],
            'rain': [0, 1, 0],
            'Classes': ['not fire', 'fire', 'fire'],
            'Region': ['A', 'B', 'C'],
        })
        cleaned_data = self.preprocessor.clean_data(sample_data.copy())

        # Assert checks (example)
        self.assertEqual(cleaned_data.shape[0], 3)
        self.assertIn('Region', cleaned_data.columns) 

    def test_select_features(self):
        # Load your data or create mock data for testing
        # ...

        selected_features = self.preprocessor.select_features(X_train, y_train)
        self.assertIsNotNone(selected_features)
        self.assertGreater(len(selected_features), 0)

    # Add more unit tests for other methods in DataPreprocessor