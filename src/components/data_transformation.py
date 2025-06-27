import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object
@dataclass
class DataTransformationConfig:
   preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_ingestion_config=DataTransformationConfig()
    def get_data_transformation(self):
        try:
            logging.info("Data Transformation initiated")
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as a:
            raise CustomException(a,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data Transformation started")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Data loaded successfully")
            logging.info(f"Train DataFrame shape: {train_df.shape}")
            logging.info(f"Test DataFrame shape: {test_df.shape}")
            preprocessing_obj=self.get_data_transformation()
            target_column_name="math_score"
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Data Transformation completed successfully")
            save_object(
                file_path=self.data_ingestion_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_ingestion_config.preprocessor_obj_file_path
            )
        except Exception as a:
            raise CustomException(a,sys)
# Dummy input features
# input_feature_train_arr = np.array([[1, 2], [3, 4], [5, 6]])
# Dummy target feature
# target_feature_train_df = pd.Series([0, 1, 0])

# train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

# print(train_arr)
# Output:

# [[1 2 0]
#  [3 4 1]
#  [5 6 0]]
# OneHotEncoder() outputs a sparse matrix by default.
# StandardScaler() tries to center the data (i.e., subtract the mean).
# You cannot center sparse matrices without converting them to dense first (which is memory inefficient).
# cat_pipeline = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("one_hot_encoder", OneHotEncoder()),
#         ("scaler", StandardScaler(with_mean=False))  # <-- Fix here
#     ]
# )
