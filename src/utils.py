import os
import sys
import yaml
import json
import pickle
from types import SimpleNamespace
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.exception import CustomException
from src.logger import logging

def read_yaml(file_path):
    """Reads YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        tuple: Configuration dictionary and model parameters dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            args = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from {file_path}")
    except Exception as e:
        raise CustomException(f"Error during loading configuration from {file_path}", e)

    configs = SimpleNamespace(**args['configs'])
    logging.info(f"Configuration and Hyperparameters loaded successfully from {file_path}")

    return configs, args['model_params']

def save_artifact(file_path, artifact, artifact_type="pkl"):
    """Saves an artifact to disk based on the specified type.

    Args:
        file_path (str): The full path to the file to save.
        artifact (object): The artifact to save.
        artifact_type (str, optional): The type of artifact to save. Defaults to "pickle".
            Supported types: "pickle", "json", "image", "text", "csv".
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        filename = os.path.basename(file_path)
        artifact_type = filename.split('.')[-1]

        if artifact_type == "pkl":
            with open(file_path, "wb") as file:
                pickle.dump(artifact, file)
            logging.info(f"Artifact '{filename}' saved as pickle file.")

        elif artifact_type == "json":
            with open(file_path, "w") as file:
                json.dump(artifact, file)
            logging.info(f"Artifact '{filename}' saved as JSON file.")

        elif artifact_type == "png":
            artifact.savefig(file_path)
            logging.info(f"Artifact '{filename}' saved as PNG image.")

        elif artifact_type == "txt":
            with open(file_path, "w") as file:
                file.write(artifact)
            logging.info(f"Artifact '{filename}' saved as text file.")

        elif artifact_type == "csv":
            artifact.to_csv(file_path, index=False, header=True)
            logging.info(f"Artifact '{filename}' saved as CSV file.")

        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

        return os.path.basename(file_path)

    except Exception as e:
        raise CustomException(e, sys)

def plot_confusion_matrix(cm, artifacts_path):
    """Plots and saves a confusion matrix."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        save_artifact(os.path.join(artifacts_path, "confusion_matrix.png"), plt, artifact_type="png")
        logging.info(f"Confusion Matrix saved to directory: {artifacts_path}/confusion_matrix.png")
    except Exception as e:
        raise CustomException(e, sys)

def plot_roc_curve(fpr, tpr, auc, artifacts_path):
    """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(auc))
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        save_artifact(os.path.join(artifacts_path, "auc_plot.png"), plt, artifact_type="png")
        logging.info(f"ROC Curve saved to directory: {artifacts_path}/auc_plot.png")
    except Exception as e:
        raise CustomException(e, sys)

def plot_precision_recall_curve(precision, recall, thresholds, artifacts_path):
    """Plots and saves the Precision-Recall curve."""
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(thresholds, precision[1:], label="Precision")
        plt.plot(thresholds, recall[1:], label="Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Precsion/Recall")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower right")
        save_artifact(os.path.join(artifacts_path, "pr_curve.png"), plt, artifact_type="png")
        logging.info(f"Precision-Recall Curve saved to directory: {artifacts_path}/pr_curve.png")
    except Exception as e:
        raise CustomException(e, sys)

