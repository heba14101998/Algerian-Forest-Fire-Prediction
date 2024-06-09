import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_artifact, read_yaml
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFECV

load_dotenv()
SEED = int(os.environ.get("SEED"))

class DataPreprocessor:

    def __init__(self):
        self.configs, _ = read_yaml('params.yaml')

    def clean_data(self, df):

        """Cleans the data by handling missing values and inconsistencies."""
        df.columns = df.columns.str.strip()
        df.iloc[168,-2]=np.NaN
        df.iloc[168,-1]='fire'
        df['Region'] = np.NaN
        df.loc[:122, "Region"]= 0
        df.loc[125:, "Region"]= 1
        df = df.dropna()
        df[self.configs.target_column] = df[self.configs.target_column].str.strip()
        df[self.configs.target_column] = df[self.configs.target_column].map({'not fire': 0, 'fire': 1})

        df.to_csv(f"cleaned_{self.configs.data_file_name}", index=False, header=True)
        
        return df

    def select_features(self, X, y):
        """Selects features using Recursive Feature Elimination (RFE)."""
        rfecv = RFECV(
            estimator=LogisticRegression(),
            step=1,
            cv=self.configs.cv,
            scoring=self.configs.scoring,
            n_jobs=2,
        )
        rfecv.fit(X, y)
        selected_features = X.columns[rfecv.support_]
        return selected_features

    def create_pipeline(self):

        if self.configs.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.configs.scaling_method == 'standard':
            scaler = StandardScaler()
        else:
            raise CustomException(f"Invalid scaling method: {self.scaling_method}")

        pipeline = Pipeline([
            ('scaling', scaler),
        ])

        return pipeline

    
    def preprocess(self, df):

        try:
            df = self.clean_data(df)

            X = df.drop(self.configs.target_column, axis=1)
            y = df[self.configs.target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=self.configs.test_size, 
                                                                random_state=SEED)
            logging.info("Split dataset to train and evaluation completed.")
            
            selected_features = self.select_features(X_train, y_train)
            logging.info(f"The selected features in feature selection process are: {list(selected_features)}")
            save_artifact("selected_features.json", {'selected_features':list(selected_features)})

            pipeline = self.create_pipeline()
            X_train_arr= pipeline.fit_transform(X_train[selected_features])
            X_test_arr = pipeline.transform(X_test[selected_features])
            save_artifact('preprocessor_pipline.pkl', pipeline)

            # Save processed data as numpy
            np.save(os.path.join(self.configs.artifacts_path,'X_train.npy'), X_train_arr)
            np.save(os.path.join(self.configs.artifacts_path,'X_test.npy'), X_test_arr)
            np.save(os.path.join(self.configs.artifacts_path,'y_train.npy'), np.array(y_train))
            np.save(os.path.join(self.configs.artifacts_path,'y_test.npy'), np.array(y_test))
            logging.info(f"saving data in four numpy files in artifacts directory")

        except Exception as e:
            raise CustomException(e, sys)
        
        return X_train_arr, X_test_arr, np.array(y_train), np.array(y_test)

# if __name__=='__main__':
#     run = DataPreprocessor()
#     df = pd.read_csv("data/raw/Algerian_forest_fires_dataset.csv")
#     X_train, X_test, y_train, y_test = run.preprocess(df)
    