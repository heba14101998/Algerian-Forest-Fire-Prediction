import os
import yaml
import pickle
from types import SimpleNamespace
from src.exception import CustomException
from src.logger import logging

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            args = yaml.safe_load(file)

        logging.info(f"Configuration loaded successfully from {file_path}")
    except Exception as e:
        raise CustomException(f"Error during loading configuration from {file_path}", e)

    args_paths = SimpleNamespace(**args['paths'])
    args_params = SimpleNamespace(**args['hyperparams'])
    logging.info(f"Configuration and Hyperparameters loaded successfully from {file_path}")
    
    return args_paths, args_params

def save_artifact(filename, artifact):
    
    os.makedirs("artifacts/", exist_ok=True)
    file_path = os.path.join("artifacts", filename)
    try:
        with open(file_path, "wb") as file:
            pickle.dump(artifact, file)
        logging.info(f"saving {filename} in {file_path}")

    except Exception as e:
        raise CustomException(e, sys)