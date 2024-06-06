import os
import sys
# import argparse
# import yaml
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_artifact, read_yaml
from sklearn.model_selection import train_test_split

SEED = 42

class DataIngestor:
    def __init__(self, configs, params):
        self.configs = configs
        self.params  = params

    def data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.configs.raw_data_path)
            logging.info(f"Data read successfully. Shape: {df.shape}")
            
            # Create output directories if they don't exist
            os.makedirs(os.path.dirname(self.configs.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.configs.eval_data_path), exist_ok=True)
            
            train_set, eval_set = train_test_split(df, test_size=self.params.test_size, random_state=SEED)
            logging.info(f"Splitted raw data to training set with shape: {train_set.shape} ; and evaluation set with shape: {eval_set.shape}")
            
            # Save the training and evaluation sets
            train_set.to_csv(self.configs.train_data_path, index=False, header=True)
            eval_set.to_csv(self.configs.eval_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

        except FileNotFoundError as e:
            raise CustomException("Raw data file not found!") from e
        except ValueError as e:
            raise CustomException(f"Error while reading data: {e}") from e

        return self.configs.train_data_path, self.configs.eval_data_path

if __name__ == '__main__':

    try:
        args_paths, args_params = read_yaml('params.yaml')
        print(args_params)
        run = DataIngestor(args_paths, args_params)
        train_data, eval_data = run.data_ingestion()
        print(f"Training data saved at: {train_data}")
        print(f"Evaluation data saved at: {eval_data}")

    except Exception as e:
        raise CustomException("Error during data ingestion", e)
