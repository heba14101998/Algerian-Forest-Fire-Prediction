"""
This module handles data preprocessing tasks for a machine learning project.
The `DataPreprocessor` class performs data cleaning, feature selection, and scaling.

Key functionalities:
    - clean data inconsistency: Cleans the data by handling missing values, inconsistencies, 
                                and encoding the target variable.
    - Feature Selection: Selects relevant features using Recursive Feature Elimination (RFE).
    - create preprocessing pipline: Creates a data preprocessing pipeline for scaling.
"""
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFECV

from src.logger import logging
from src.exception import CustomException
from src.utils import save_artifact, read_yaml

SEED = 42

class DataPreprocessor:
    """
    Performs data preprocessing tasks such as cleaning, feature selection, and scaling.
    Attributes:
        configs (SimpleNamespace): An object containing configuration parameters.
    Methods:
        clean_data(): Cleans the data by handling missing values and inconsistencies, 
                      encoding the target variable, and saving the cleaned DataFrame.
        select_features(): Selects features using Recursive Feature Elimination (RFE).
        create_pipeline(): Creates a data preprocessing pipeline for scaling.
        preprocess(): Orchestrates the preprocessing steps, including cleaning, feature 
                      selection, and splitting the data into training and testing sets.
    """
    def __init__(self, args):
        """
        Initializes the DataPreprocessor with configuration parameters.
        """
        self.configs = args
        file_path = os.path.join(self.configs.raw_data_dir, self.configs.data_file_name)
        self.data = pd.read_csv(file_path)

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans the data by handling missing values and inconsistencies. 
        Steps:
            1. Strip column names
            2. Solve inconsistencies
            3. Drop NULL values
            4. Strip whitespace from target values
            5. Map target column values
        """
        self.data.columns = self.data.columns.str.strip()
        self.data.iloc[168, -2] = np.nan
        self.data.iloc[168, -1] = 'fire'
        self.data['Region'] = np.nan
        self.data.loc[:122, "Region"] = 0
        self.data.loc[125:, "Region"] = 1

        self.data = self.data.dropna()
        self.data[self.configs.target_column] = self.data[self.configs.target_column].str.strip()
        self.data[self.configs.target_column] = self.data[self.configs.target_column].\
                                                map({'not fire': 0, 'fire': 1})

        # Save the DataFrame as a CSV
        path = os.path.join(self.configs.processed_data_dir, 
                            f"cleaned_{self.configs.data_file_name}")
        save_artifact(path, self.data)


    def select_features(self, X: pd.DataFrame, y: pd.Series):
        """
        Selects features using Recursive Feature Elimination (RFE).

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target variable.
        Returns:
            np.ndarray: An array of selected features.
        """
        rfecv = RFECV(
            estimator=LogisticRegression(),
            step=1,
            cv=self.configs.cv,
            scoring=self.configs.scoring,
            n_jobs=2,
        )
        rfecv.fit(X, y)
        selected_features = X.columns[rfecv.support_]

        logging.info(f"The selected features in feature selection process are: {list(selected_features)}")
        path = os.path.join(self.configs.artifacts_path, "selected_features.json")
        save_artifact(file_path=path, artifact={'selected_features': list(selected_features)})

        return selected_features

    def create_pipeline(self) -> Pipeline:
        """
        Creates a data preprocessing pipeline.
        Returns:
            Pipeline: The preprocessing pipeline.
        """
        if self.configs.scaling_method == 'minmax':
            pipeline = Pipeline([('scaling', MinMaxScaler()),])
        elif self.configs.scaling_method == 'standard':
            pipeline = Pipeline([('scaling', StandardScaler()),])
        else:
            raise CustomException(f"Invalid scaling method: {self.configs.scaling_method}")

        return pipeline

    def preprocess(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs data preprocessing, including cleaning, feature selection, and splitting.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  
                The preprocessed data as four NumPy arrays: X_train, X_test, y_train, y_test.
        """
        # Clean the data by handling missing values and inconsistencies.
        self.clean_data()

        # Split dataset to train and evaluation sets.
        X = self.data.drop(self.configs.target_column, axis=1)
        y = self.data[self.configs.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.configs.test_size,
                                                            random_state=SEED)
        logging.info("Split dataset to train and evaluation completed.")

        # Select features using Recursive Feature Elimination (RFE).
        selected_features = self.select_features(X_train, y_train)

        # fit the pipline on the train dataset and apply it in both the train and test sets
        pipeline = self.create_pipeline()
        X_train_arr = pipeline.fit_transform(X_train[selected_features])
        X_test_arr = pipeline.transform(X_test[selected_features])

        # Save trained pipline as pickle file
        path = os.path.join(self.configs.checkpoints, 'preprocessor.pkl')
        save_artifact(path, pipeline)

        # Save processed data as numpy
        np.save(os.path.join(self.configs.processed_data_dir, 'X_train.npy'), X_train_arr)
        np.save(os.path.join(self.configs.processed_data_dir, 'X_test.npy'), X_test_arr)
        np.save(os.path.join(self.configs.processed_data_dir, 'y_train.npy'), np.array(y_train))
        np.save(os.path.join(self.configs.processed_data_dir, 'y_test.npy'), np.array(y_test))
        logging.info("Saving data in four numpy files in artifacts directory")

        return X_train_arr, X_test_arr, np.array(y_train), np.array(y_test)

if __name__ == '__main__':

    logging.info("Data Preparation")
    configs, _ = read_yaml('params.yaml')
    run = DataPreprocessor(configs)
    X_train, X_test, y_train, y_test = run.preprocess()

    