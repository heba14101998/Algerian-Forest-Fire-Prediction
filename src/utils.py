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
    model = SimpleNamespace(**args['model'])
    logging.info(f"Configuration and Hyperparameters loaded successfully from {file_path}")
    
    return configs, model

def save_artifact(filename, artifact):
    
    os.makedirs("artifacts/", exist_ok=True)
    file_path = os.path.join("artifacts", filename)
    try:
        with open(file_path, "wb") as file:
            pickle.dump(artifact, file)
        logging.info(f"saving {filename} in {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def model_evaluate(X_true, y_pred, y_pred_proba=None, task='classification'):
    pass

# def evaluate_model(self, model):
#         try:
#             y_pred = model.predict(self.X_eval)
#             y_pred_proba = model.predict_proba(self.X_eval) if hasattr(model, 'predict_proba') else None
#             eval_performance = model_evaluate(self.X_eval, y_pred, y_pred_proba, self.configs['task'])
#             return eval_performance

#         except Exception as e:
#             raise CustomException("Error during model evaluation", e)
