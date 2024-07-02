"""
This module provides a class for training and evaluating machine learning models. 
It utilizes sklearn for model training, evaluation, and metrics calculation.
Classes:
    ModelTrainer: A class for training and evaluating machine learning models.
"""
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, roc_auc_score, roc_curve,
                            precision_recall_fscore_support, precision_recall_curve)

from src.logger import logging
from src.exception import CustomException
from src.utils import (save_artifact, read_yaml, plot_precision_recall_curve,
                         plot_roc_curve, plot_confusion_matrix)
from dvclive import Live 

class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    Attributes:
        configs (SimpleNamespace): Configuration parameters for the model training process.
        model_params (SimpleNamespace): Hyperparameters for the specific model to be trained.
        model: The trained machine learning model.
    """

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
        """
        Trains the specified model using the provided training data.
        Args:
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training target labels.
        """
        try:
            self.model = self.models_map[self.configs.model_name]
            self.model.set_params(**self.model_params)  # Set model hyperparameters
            self.model.fit(X_train, y_train)
            logging.info(f"Training model: {self.configs.model_name} \
                            with parameters: {self.model_params}")

            path = os.path.join(self.configs.checkpoints, f'{self.configs.model_name}.pkl')
            save_artifact(path, self.model)
            logging.info("Model saved as pickle file in Artifacts")
        
        except Exception as e:
            raise CustomException(f"Error during training the model: {e}")


    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model on the provided test data.
        Args:
            X_test (numpy.ndarray): Testing features.
            y_test (numpy.ndarray): Testing target labels.
        Returns:
            tuple: Evaluation metrics, confusion matrix, and classification report.
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # For AUC calculation

        with Live(save_dvc_exp=True) as live:
            # Calculate classification metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
            # Make metrics in dict style
            metrics = { "accuracy": acc,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score}

            live.log_metric("accuracy", acc)
            live.log_metric("auc", auc)
            live.log_metric("f1_score", f1_score)
            live.log_metric("precision", precision)
            live.log_metric("recall", recall)
            # Save metrics as json file in local storage     
            path = os.path.join(self.configs.artifacts_path, "metrics.json")
            save_artifact(path, metrics)
            
            ################################## Basic Metrics ######################################

            # Calculate data for ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            # Make ROC curve data in dict style
            roc_data = { "fpr": fpr.tolist(), 
                         "tpr": tpr.tolist(),
                         "thresholds": thresholds.tolist(), 
                         "roc_auc": auc}
            # Save ROC curve data as json in local storage  
            path = os.path.join(self.configs.artifacts_path, "roc_data.csv")
            save_artifact(path, pd.DataFrame(roc_data))
            # Plotting ROC curve
            plot_roc_curve(fpr, tpr, auc, self.configs.artifacts_path)
            # Plot in DVCLive 
            live.log_sklearn_plot("roc", y_test, y_pred, name="ROC Curve")

            #################################### Precision Recall ##################################
            
            # Calculate data for precision recall curve
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            # Make precision recall curves data in dict style
            pr_data = {"precision": precision.tolist()[1:],
                        "recall": recall.tolist()[1:],
                        "thresholds": thresholds.tolist()}
            # Save precision recall curves data as json in local storage
            path = os.path.join(self.configs.artifacts_path, "pr_data.csv")
            save_artifact(path, pd.DataFrame(pr_data))
            # Plot precision recall curves 
            plot_precision_recall_curve(precision, recall, thresholds, self.configs.artifacts_path)
            # Plot Using DVCLive
            live.log_sklearn_plot("precision_recall", y_test, y_pred, 
                                    name="Precision Recall Curve",
                                    drop_intermediate=True)

            ################################# Confusion Matrix #####################################

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # Make confusion matrix in dict style
            cm_data= {'actual': y_test.tolist(), 'predicted': y_pred.tolist()}
            # Save confusion matrix data as json in local storage
            path = os.path.join(self.configs.artifacts_path, 'cm_data.csv')
            save_artifact(path, pd.DataFrame(cm_data))
            # Plot confusion matrix
            plot_confusion_matrix(cm, self.configs.artifacts_path)
             # Plot in DVCLIve 
            live.log_sklearn_plot("confusion_matrix", y_test, y_pred, name="Confusion Matrix")

            ################################# Classification Report ##################################

            # Calculate classification report
            classification_rep = classification_report(y_test, y_pred)
            # Save classification report as text file in local storage  
            path = os.path.join(self.configs.artifacts_path, "classification_report.txt")
            save_artifact(path, classification_rep)
            
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
