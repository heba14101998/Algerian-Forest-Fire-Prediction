import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_artifact, read_yaml, model_evaluate

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

class ModelTrainer:
    def __init__(self, configs, model_params):

        self.configs = configs
        self.model_params = model_params
        self.models_map = {
            'RandomForestClassifier': RandomForestClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'LogisticRegression': LogisticRegression(),
            'XGBClassifier': XGBClassifier(),
            'CatBoostClassifier': CatBoostClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),        
            }

        # Load data as numpy
        try:
            self.X_train = np.load(os.path.join(self.configs.artifacts_path, f'X_train.npy'))
            self.X_eval  = np.load(os.path.join(self.configs.artifacts_path, f'X_eval.npy'))
            self.y_train = np.load(os.path.join(self.configs.artifacts_path, f'y_train.npy'))
            self.y_eval  = np.load(os.path.join(self.configs.artifacts_path, f'y_eval.npy'))
            logging.info("Data loaded successfully")
        
        except Exception as e:
            raise CustomException("Can't load the data from numpy files", e)

    def train(self, model_config):
        
        # Train and Evaluate Models types
        best_model = None
        best_performance = float('-inf')
        
        model_name = model_config.name
        model = self.models_map[model_name]
        params = model_config.params
        logging.info(f"Training model: {model_name} with parameters: {params}")
        
        # train vanilla model
        model.fit(self.X_train, self.y_train)

        if eval_performance > best_score:
            best_model = best_estimator
            best_performance = eval_performance
        
        return model
        
    def run(self):
        
        # Train and Evaluate Models types
        best_model = None
        best_performance = float('-inf')

        for model_config in self.hyperparams['models']:
            search = self.train(model_config)

            if self.configs.search_type in ['grid', 'random']:
                best_params = search.best_params_
                best_estimator = search.best_estimator_

                try:
                    y_pred = best_estimator.predict(self.X_eval)
                    y_pred_proba = best_estimator.predict_proba(self.X_eval)
                except Exception as e:
                        y_pred = best_estimator.predict(self.X_eval)
                        y_pred_proba = None
                        logging.warning(f"Error during prediction: {e}")

                eval_performance = model_evaluate(self.X_eval, y_pred, y_pred_proba, self.configs.task)
               

            if self.configs.task=='regression':
                if eval_performance < best_score:
                    best_model = best_estimator
                    best_performance = eval_performance

            elif self.configs.task=='classification':
                if eval_performance > best_score:
                    best_model = best_estimator
                    best_performance = eval_performance
            

if __name__=='__main__':
    try:
        configs, models = read_yaml('params.yaml')
        exp = ModelTrainer(configs, models )
        exp.run()
    
    except Exception as e:
<<<<<<< HEAD
        raise CustomException("Error during data transformation", e)

=======
        raise CustomException("Error during data transformation", e)
>>>>>>> working_on_notebook
