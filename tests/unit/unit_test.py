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