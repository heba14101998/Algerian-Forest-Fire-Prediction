import os
import sys
import shutil
import pandas as pd
from dotenv import load_dotenv

from src.exception import CustomException
from src.logger import logging
from src.utils import save_artifact, read_yaml
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()
DATASET_API = os.environ.get("DATASET_API")
SEED = os.environ.get("SEED")


class DataIngestor:
    def __init__(self):
        self.configs, _ = read_yaml('params.yaml')
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

    # def data_ingestion(self):

    #     logging.info("Entered the data ingestion method or component")

    #     try:
    #         file_path = os.path.join(self.configs.raw_data_dir, self.configs.data_file_name)
    #         df = pd.read_csv(file_path)
    #         logging.info(f"Data read successfully. Shape: {df.shape}")
            
    #         # Create output directories if they don't exist
    #         os.makedirs(os.path.dirname(self.configs.processed_data_dir), exist_ok=True)            
    #         train_set, eval_set = train_test_split(df, test_size=self.configs.test_size, random_state=SEED)
    #         logging.info(f"Splitted raw data to training set with shape: {train_set.shape} ; and evaluation set with shape: {eval_set.shape}")
            
    #         # Save the training and evaluation sets
    #         train_set.to_csv(self.configs.train_data_path, index=False, header=True)
    #         eval_set.to_csv(self.configs.eval_data_path, index=False, header=True)

    #         logging.info("Ingestion of the data is completed")

    #     except FileNotFoundError as e:
    #         raise CustomException("Raw data file not found!") from e
    #     except ValueError as e:
    #         raise CustomException(f"Error while reading data: {e}") from e

    #     return self.configs.train_data_path, self.configs.eval_data_path

# if __name__ == '__main__':

#     run = DataIngestor()
#     train_data, eval_data = run.data_ingestion()
#     print(f"Training data saved at: {train_data}")
#     print(f"Evaluation data saved at: {eval_data}")