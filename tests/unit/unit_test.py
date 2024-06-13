import pytest
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
            'Region': [0, 1, 1],
        })
        cleaned_data = self.preprocessor.clean_data(sample_data.copy())

        # Assert checks (example)
        self.assertEqual(cleaned_data.shape[0], 3)
        self.assertIn('Region', cleaned_data.columns) 

    def test_select_features(self):


        selected_features = self.preprocessor.select_features(X_train, y_train)
        self.assertIsNotNone(selected_features)
        self.assertGreater(len(selected_features), 0)
