"""
This script handles the downloading of a dataset from Kaggle using the Kaggle API.
It utilizes environment variables for authentication and reads configuration from a YAML file.
    - The script ensures the existence of the raw data directory before downloading the dataset.
    - It leverages the `kaggle.api.authenticate` and `kaggle.api.dataset_download_files` functions 
      for authentication and downloading the dataset.
    - Exceptions are handled with logging for error tracking.
"""
import os
import sys
import kaggle
from dotenv import load_dotenv

from src.utils import read_yaml
from src.exception import CustomException
from src.logger import logging

load_dotenv()

class DataIngestor:
    """
    A class responsible for downloading datasets from Kaggle.
    Attributes:
        configs (SimpleNamespace): An object containing configuration parameters.
    Methods:
        download_dataset(): Downloads the specified dataset from Kaggle using the Kaggle API.
    """
    def __init__(self, args):

        self.configs = args
        # self.download_dataset()

    def download_dataset(self):
        """
        Uses the DATASET_API environment variable to specify the dataset to download.
        Creates the raw data directory if it doesn't exist.
        
        Raises:
            CustomException: If an unexpected error occurs during the download.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(self.configs.raw_data_dir, exist_ok=True)
        try:
            dataset_api = os.environ.get("DATASET_API")
            if dataset_api is None:
                logging.error("DATASET_API environment variable is not set.")
                sys.exit(1)
            else:
                logging.info(f"Using dataset API: {dataset_api}")

            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset_api,  # This is the full dataset URL
                self.configs.raw_data_dir,
                unzip=True,
            )

        except CustomException as e:
            logging.error(f"An unexpected error occurred during download: {e}")

if __name__ == '__main__':
    configs, _ = read_yaml('params.yaml')
    run = DataIngestor(configs)
    run.download_dataset()
    
    print("HEllO")