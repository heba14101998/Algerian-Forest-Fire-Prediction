import os
import sys
import yaml
import json
import pickle
from types import SimpleNamespace
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            args = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from {file_path}")
    
    except Exception as e:
        raise CustomException(f"Error during loading configuration from {file_path}", e)

    configs = SimpleNamespace(**args['configs'])
    logging.info(f"Configuration and Hyperparameters loaded successfully from {file_path}")
    
    return configs, args['model_params']

def save_artifact(filename, artifact, artifact_type="pkl"):
    """Saves an artifact to disk based on the specified type.

    Args:
        filename (str): The name of the file to save.
        artifact (object): The artifact to save.
        artifact_type (str, optional): The type of artifact to save. Defaults to "pickle". 
            Supported types: "pickle", "json", "image", "text"
    """
    try:
        os.makedirs("artifacts/", exist_ok=True)
        file_path = os.path.join("artifacts", filename)
        artifact_type = filename.split('.')[-1]

        if artifact_type == "pkl":
            with open(file_path, "wb") as file:
                pickle.dump(artifact, file)
            logging.info(f"Saving {filename} as pickle file in {file_path}")

        elif artifact_type == "json":
            with open(file_path, "w") as file:
                json.dump(artifact, file)
            logging.info(f"Saving {filename} as JSON file in {file_path}")

        elif artifact_type == "png":
            artifact.savefig(file_path)
            logging.info(f"Saving {filename} as image file in {file_path}")

        elif artifact_type == "txt":
            with open(file_path, "w") as file:
                file.write(artifact)
            logging.info(f"Saving {filename} as text file in {file_path}")

        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

    except Exception as e:
        raise CustomException(e, sys)

