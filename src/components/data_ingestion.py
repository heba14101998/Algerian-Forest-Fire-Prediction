import os
import sys
import shutil
import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml

# Load environment variables
load_dotenv()
DATASET_API = os.environ.get("DATASET_API")
SEED = os.environ.get("SEED")

print(f"PYTHONPATH: {os.environ['PYTHONPATH']}") 

class DataIngestor:
    def __init__(self, configs):

        self.configs = configs
        self.download_dataset()

    def download_dataset(self):
        # Download the dataset
        os.makedirs(self.configs.raw_data_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        try:
            os.system(f"kaggle datasets download -d {DATASET_API} -p {self.configs.raw_data_dir} -f {self.configs.data_file_name}")

        except:
            os.system(f"kaggle datasets download -d {DATASET_API} -p {self.configs.raw_data_dir}")
            
            # Find the downloaded zip file and extract it
            for filename in os.listdir(self.configs.raw_data_dir):

                zip_path = os.path.join(self.configs.raw_data_dir, filename)
                logging.info(f"File '{filename}' downloaded to '{self.configs.raw_data_dir}'")
                
                shutil.unpack_archive(zip_path, self.configs.raw_data_dir, 'zip')
                logging.info(f"Unpack zip file '{filename}'")
                
                os.remove(zip_path) 
                logging.info(f"Remove zip file '{filename}'")
                
                os.rename(os.path.join(self.configs.raw_data_dir, filename), os.path.join(self.configs.raw_data_dir, self.configs.data_file_name))
                logging.info(f"Rename csv file to '{self.configs.data_file_name}'")

        logging.info(f"Dataset '{DATASET_API}' downloaded to '{self.configs.raw_data_dir}'")


    # def download_dataset(self):
    #     os.makedirs(self.configs.raw_data_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    #     # Get owner_slug
    #     owner_slug = DATASET_API.split('/')[0]
    #     # Get the dataset name
    #     dataset_name = DATASET_API.split('/')[1]

    #     try:
    #         # Authenticate to Kaggle API
    #         api = KaggleApi()
    #         api.authenticate()

    #         # Download the dataset
    #         api.datasets_download(owner_slug, dataset_name)
    #         logging.info(f"Dataset '{dataset_name}' downloaded to '{self.configs.raw_data_dir}'")

    #         # Extract the downloaded ZIP file (if needed)
    #         for filename in os.listdir(self.configs.raw_data_dir):
    #             if filename.endswith(".zip"):
    #                 zip_path = os.path.join(self.configs.raw_data_dir, filename)
    #                 shutil.unpack_archive(zip_path, self.configs.raw_data_dir, 'zip')
    #                 os.remove(zip_path) 

    #         logging.info(f"Unpack '{dataset_name}' to '{self.configs.raw_data_dir}'")

    #     except Exception as e:
    #         raise CustomException(e, sys)

if __name__ == '__main__':

    configs, _ = read_yaml('params.yaml')
    run = DataIngestor(configs)
    run.download_dataset()
    