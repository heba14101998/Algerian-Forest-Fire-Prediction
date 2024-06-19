import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import logging
from src.exception import CustomException
from src.utils import (save_artifact, read_yaml, plot_precision_recall_curve,
                         plot_roc_curve, plot_confusion_matrix)

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, 
precision_recall_fscore_support, precision_recall_curve, auc)

class ModelTrainer:
    def __init__(self, configs,  model_params):
        self.configs, self.model_params = configs,  model_params        
        self.model = None

        logging.info(f"Model type: {self.configs.model_name}")
        self.models_map = {
                "LogisticRegression": LogisticRegression(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SVC": SVC(),
                "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
                "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
                "AdaBoostClassifier": AdaBoostClassifier()
            }

    def train(self, X_train, y_train):
        try:
            self.model = self.models_map[self.configs.model_name]
            self.model.set_params(**self.model_params)  # Set hyperparameters
            self.model.fit(X_train, y_train)
            logging.info(f"Training model: {self.configs.model_name} with parameters: {self.model_params}")
            
            path = os.path.join(self.configs.checkpoints, f'{self.configs.model_name}.pkl')
            save_artifact(path, self.model)
            logging.info(f"Model saved as pickle file in Artifacts")
        
        except Exception as e:
            raise CustomException(f"Error during training the model: {e}")


    def evaluate(self, X_test, y_test):
        """Evaluates the trained model on the test set."""
        try:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # For AUC calculation
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
            
            # Save metrics as json file
            metrics = { "accuracy": acc, "auc": auc, "precision": precision, 
                        "recall": recall, "f1_score": f1_score}            
            path = os.path.join(self.configs.artifacts_path, "metrics.json")
            save_artifact(path, metrics)
            
            classification_rep = classification_report(y_test, y_pred)
            path = os.path.join(self.configs.artifacts_path, "classification_report.txt")
            save_artifact(path, classification_rep)

            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            # Save curves data as json files
            roc_data = { "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                         "thresholds": thresholds.tolist(), "roc_auc": auc}

            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

            pr_data = {"precision": precision.tolist()[1:], "recall": recall.tolist()[1:],
                        "thresholds": thresholds.tolist()}
                
            path = os.path.join(self.configs.artifacts_path, "roc_data.csv")
            save_artifact(path, pd.DataFrame(roc_data))

            path = os.path.join(self.configs.artifacts_path, "pr_data.csv")
            save_artifact(path, pd.DataFrame(pr_data))

            # Call plotting functions from utils.py
            plot_confusion_matrix(cm, self.configs.artifacts_path)
            plot_roc_curve(fpr, tpr, auc, self.configs.artifacts_path)

            plot_precision_recall_curve(precision, recall, thresholds, self.configs.artifacts_path)
            cm_data= {'actual': y_test.tolist(), 'predicted': y_pred.tolist()}
            path = os.path.join(self.configs.artifacts_path, 'cm_data.csv')
            save_artifact(path, pd.DataFrame(cm_data))

            logging.info(f"y_test and y_pred saved as CSV in Artifacts")

        except Exception as e:
            raise CustomException(f"Error during evaluating the model: {e}")

        
        return metrics, cm, classification_rep

if __name__=='__main__':

    logging.info(f"Training Model")
    configs, model_params = read_yaml('params.yaml')
    X_train = np.load(os.path.join(configs.processed_data_dir,'X_train.npy'))
    X_test = np.load(os.path.join(configs.processed_data_dir,'X_test.npy'))
    y_train = np.load(os.path.join(configs.processed_data_dir,'y_train.npy'))
    y_test = np.load(os.path.join(configs.processed_data_dir,'y_test.npy'))
    logging.info(f"Loading data in four numpy files completed")

    exp = ModelTrainer(configs, model_params)   
    exp.train(X_train, y_train) 
    metrics, cm, classification_rep = exp.evaluate(X_train, y_train) 
    metrics, cm, classification_rep = exp.evaluate(X_test, y_test) 