import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import logging
from src.exception import CustomException
from src.utils import save_artifact

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support

class ModelTrainer:
    def __init__(self, configs , model_params):

        self.configs = configs 
        self.model_params = model_params 
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

            save_artifact(f'{self.configs.model_name}.pkl', self.model)
            model_path = os.path.join(self.configs.artifacts_path, f"{self.configs.model_name}.pkl")
            logging.info(f"Model saved as pickle file in {model_path}")
        
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self, X_test, y_test):
        """Evaluates the trained model on the test set."""
        try:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # For AUC calculation
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)

            metrics = {'accuracy': acc, 'auc': auc, 'precision': precision, 
                        'recall': recall, 'f1_score': f1_score}

            logging.info(f"Classification Report:\n{classification_rep}")
            logging.info(f"confusion matrix:\n{cm}")
            
            save_artifact("classification_report.txt", classification_rep)
            save_artifact("metrics.json", metrics)
            logging.info(f"Metrics saved to directory: {self.configs.artifacts_path}")

            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',ax=fig.add_subplot(111))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            save_artifact('confusion_matrix.png', fig)
            logging.info(f"Confusion Matrix saved to directory: {self.configs.artifacts_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
        
        return metrics, cm, classification_rep

# if __name__=='__main__':

#     X_train = np.load(os.path.join('./artifacts', f'X_train.npy'))
#     X_test  = np.load(os.path.join('./artifacts', f'X_test.npy'))
#     y_train = np.load(os.path.join('./artifacts', f'y_train.npy'))
#     y_test  = np.load(os.path.join('./artifacts', f'y_test.npy'))
#     logging.info("Data loaded successfully")

#     exp = ModelTrainer()   
#     exp.train(X_train, y_train) 
#     exp.test(X_test, y_test) 