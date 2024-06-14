import unittest
from unittest.mock import patch, MagicMock
import os
from src.components.data_ingestion import DataIngestor
from src.components.data_factory import DataPreprocessor
from src.utils import read_yaml
from dotenv import load_dotenv
import numpy as np
import pandas as pd
# Load environment variables
load_dotenv()

class TestDataIngestor(unittest.TestCase):

    def setUp(self):
        self.configs, _ = read_yaml('params.yaml')
        os.makedirs(self.configs.raw_data_dir, exist_ok=True)
        self.dataset_api = os.getenv("DATASET_API")
        self.data_ingestor = DataIngestor(self.configs)

    @patch('src.components.data_ingestion.os.system')
    def test_download_dataset(self, mock_os_system):
        mock_os_system.return_value = 0  # Mock the os.system call to always succeed
        self.data_ingestor.download_dataset()

        # Check if the directory and the file are created
        self.assertTrue(os.path.exists(self.configs.raw_data_dir))
        self.assertTrue(os.path.exists(os.path.join(self.configs.raw_data_dir,
                                                     self.configs.data_file_name)))

        # Check if os.system was called with the correct command
        expected_command = f"kaggle datasets download -d {self.dataset_api} -p {self.configs.raw_data_dir} -f {self.configs.data_file_name}"
        mock_os_system.assert_called_with(expected_command)

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        """Set up for testing"""
        configs, _ = read_yaml('params.yaml')
        self.preprocessor = DataPreprocessor(configs)
        self.sample_data = {
            'day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'month': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            'year': [2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012],
            'Temperature': [29, 29, 26, 25, 27, 31, 33, 30, 25, 28],
            'RH': [57, 61, 82, 89, 77, 67, 54, 73, 88, 79],
            'Ws': [18, 13, 22, 13, 16, 14, 13, 15, 13, 12],
            'Rain': [0, 1.3, 13.1, 2.5, 0, 0, 0, 0, 0.2, 0],
            'FFMC': [65.7, 64.4, 47.1, 28.6, 64.8, 82.6, 88.2, 86.6, 52.9, 73.2],
            'DMC': [3.4, 4.1, 2.5, 1.3, 3, 5.8, 9.9, 12.1, 7.9, 9.5],
            'DC': [7.6, 7.6, 7.1, 6.9, 14.2, 22.2, 30.5, 38.3, 38.8, 46.3],
            'ISI': [1.3, 1, 0.3, 0, 1.2, 3.1, 6.4, 5.6, 0.4, 1.3],
            'BUI': [3.4, 3.9, 2.7, 1.7, 3.9, 7, 10.9, 13.5, 10.5, 12.6],
            'FWI': [0.5, 0.4, 0.1, 0, 0.5, 2.5, 7.2, 7.1, 0.3, 0.9],
            'Classes': ['not fire', 'not fire', '  not fire', 'not fire  ', 'not fire', ' fire', 'fire', 'fire', 'not fire', 'not fire']
        }
        self.sample_data = pd.DataFrame(self.sample_data) 

        # Introduce some nulls randomly
        null_indices = np.random.choice(self.sample_data.index, size=int(len(self.sample_data) * 0.2), replace=False)
        for col in self.sample_data.columns:
            self.sample_data.loc[null_indices, col] = np.nan
        
    def test_clean_data(self):
        """Test if clean_data handles missing values and inconsistencies correctly."""
        data_cols = self.sample_data.columns
        self.preprocessor.clean_data(self.sample_data.copy())
        # Assert that the data has the expected shape after cleaning
        self.assertListEqual(list( self.sample_data.columns), list(data_cols))
        # Assert that the target column is correctly mapped to numerical values
        self.assertTrue(self.sample_data['Classes'].isin([0, 1]).all())
        # self.sample_data = cleaned_data

    def test_select_features(self):
        """Test if select_features correctly selects features using RFE."""
        X = self.sample_data.drop('Classes', axis=1)
        y = self.sample_data['Classes']

        selected_features = self.preprocessor.select_features(X, y)
        # Assert that at least one feature is selected
        self.assertTrue(len(selected_features) > 0)

    def test_create_pipeline(self):
        """Test if create_pipeline creates the correct pipeline based on configuration."""
        pipeline = self.preprocessor.create_pipeline()
        # Assert that the pipeline has the expected steps
        self.assertEqual(len(pipeline.steps), 1)  # Check for scaling step

    def test_preprocess(self):
        """Test if preprocess runs without errors and returns expected data."""
        try:
            X_train, X_test, y_train, y_test = self.preprocessor.preprocess()
            # Assert that the returned data has the expected shapes
            self.assertTrue(X_train.shape[0] > 0)
            self.assertTrue(X_test.shape[0] > 0)
            self.assertTrue(y_train.shape[0] > 0)
            self.assertTrue(y_test.shape[0] > 0)
            
        except CustomException as e:
            self.fail(f"Test failed with exception: {e}")


if __name__ == '__main__':
    unittest.main()