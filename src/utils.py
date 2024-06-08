import os
import yaml
import pickle
from types import SimpleNamespace
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_squared_error, r2_score

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            args = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from {file_path}")
    
    except Exception as e:
        raise CustomException(f"Error during loading configuration from {file_path}", e)

    configs = SimpleNamespace(**args['configs'])
    model_params = SimpleNamespace(**args['model_params'])
    logging.info(f"Configuration and Hyperparameters loaded successfully from {file_path}")
    
    return configs, model_params

def save_artifact(filename, artifact):
    
    os.makedirs("artifacts/", exist_ok=True)
    file_path = os.path.join("artifacts", filename)
    try:
        with open(file_path, "wb") as file:
            pickle.dump(artifact, file)
        logging.info(f"saving {filename} in {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def model_evaluate(X_true, y_pred, y_pred_proba=None):
    try:
        # Check if predicted probabilities are available (for AUC-ROC calculation)
        if y_pred_proba is not None:
            auc_roc = roc_auc_score(X_true, y_pred_proba[:, 1])  # Assuming binary classification
        else:
            auc_roc = None

        accuracy = accuracy_score(X_true, y_pred)
        precision = precision_score(X_true, y_pred)
        recall = recall_score(X_true, y_pred)
        f1 = f1_score(X_true, y_pred)
        eval_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc
        }
        return eval_metrics

    except Exception as e:
        raise CustomException(f"Error during model evaluation for classification: {e}")