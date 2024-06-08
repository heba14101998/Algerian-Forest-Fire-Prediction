import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_artifact, read_yaml

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

class DataPreprocessor:

    def __init__(self, configs,):

        self.configs = configs
        self.cat_cols = []
        self.num_cols = []
        self.X_train = None
        self.X_eval = None
        self.y_train = None
        self.y_eval = None
        self.preprocessor = None
        # self.label_encoder = LabelEncoder()

    def seperate_data(self):

        train_df = pd.read_csv(self.configs.train_data_path)
        eval_df = pd.read_csv(self.configs.eval_data_path)
        logging.info("Read train and evaluation sets completed")

        self.X_train = train_df.drop(columns = [self.configs.target_column], index=1) 
        self.y_train = train_df[self.configs.target_column]

        self.X_eval = train_df.drop(columns = [self.configs.target_column], index=1) 
        self.y_eval = train_df[self.configs.target_column]

        # Identify categorical and numerical columns
        self.cat_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        logging.info(f"Categorical columns: {self.cat_cols}")
        logging.info(f"Numerical columns: {self.num_cols}")

    def create_pipline(self):
        # Preprocessing for numerical data: Impute missing values and scale
        numerical_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                                         ('scalar', StandardScaler())])
        # Preprocessing for categorical data: Impute missing values and one-hot encode
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                 ('encode', OneHotEncoder(handle_unknown='ignore'))])
        
        self.preprocessor = ColumnTransformer(transformers= [('num', numerical_transformer, self.num_cols), 
                                                            ('cat', categorical_transformer, self.cat_cols )])

    def preprocess(self,):
        
        self.seperate_data()
        self.create_pipline()
        
        # Encode the target variable
        # self.y_train = self.label_encoder.fit_transform(self.y_train)
        # self.y_eval = self.label_encoder.transform(self.y_eval)

        X_train_arr= self.preprocessor.fit_transform(self.X_train)
        X_eval_arr = self.preprocessor.transform(self.X_eval)
        
        # Save the model to a pickle file
        save_artifact('preprocessor.pkl', self.preprocessor)
        
        # Save processed data as numpy
        np.save(os.path.join(self.configs.artifacts_path,'X_train.npy'), X_train_arr)
        np.save(os.path.join(self.configs.artifacts_path,'X_eval.npy'), X_eval_arr)
        np.save(os.path.join(self.configs.artifacts_path,'y_train.npy'), np.array(self.y_train))
        np.save(os.path.join(self.configs.artifacts_path,'y_test.npy'), np.array(self.y_eval))
        logging.info(f"saving data in four numpy files in artifacts directory")
        
        return X_train_arr, X_eval_arr, np.array(self.y_train), np.array(self.y_eval)

if __name__=='__main__':
    try:
        args = read_yaml('params.yaml')
        run = DataPreprocessor(args)
        X_train, X_eval, y_train, y_eval = run.preprocess()
    
    except Exception as e:
        raise CustomException("Error during data transformation", e)