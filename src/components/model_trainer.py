import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error,r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import warnings
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from dataclasses import dataclass
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=ModelTrainerConfig()
#     def initiate_model_trainer(self,train_arr,test_arr):
#         try:
#             logging.info("Splitting Dependent and Independent variables from train and test data")
#             x_train,y_train,x_test,y_test=(
#                 train_arr[:,:-1],
#                 train_arr[:,-1],
#                 test_arr[:,:-1],
#                 test_arr[:,-1]
#             )
#             models={
#                 "Linear Regression":LinearRegression(),
#                 "Ridge":Ridge(),
#                 "Lasso":Lasso(),
#                 "K-Nearest Neighbors":KNeighborsRegressor(),
#                 "Decision Tree":DecisionTreeRegressor(),
#                 "Random Forest":RandomForestRegressor(),                
#                 "XGBRegressor":XGBRegressor(),
#             }
#             model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
#             # best_model_score=max(model_report.values())
#             # best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
#             # best_model_score = max(model_report.values())
#             # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
#             # best_model=models[best_model_name]
#             # Extract R² values from the report
#             model_report = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
#             print("\n===== Model Performance Report (on Test Data) =====")
#             for model_name, scores in model_report.items():
#                 print(f"{model_name}:")
#                 print(f"  R2 Score  : {scores[0]:.4f}")
#                 print(f"  MAE       : {scores[1]:.4f}")
#                 print(f"  MSE       : {scores[2]:.4f}")
#                 print("-" * 40)

#             # You can still select and save the best model
#             model_scores = {model: scores[7] for model, scores in model_report.items()}
#             best_model_score = max(model_scores.values())
#             best_model_name = list(model_scores.keys())[list(model_scores.values()).index(best_model_score)]
#             best_model = models[best_model_name]

#             # model_scores = {model: scores[0] for model, scores in model_report.items()}
#             # best_model_score = max(model_scores.values())
#             # best_model_name = list(model_scores.keys())[list(model_scores.values()).index(best_model_score)]
#             # best_model = models[best_model_name]
#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )
#             predicted=best_model.predict(x_test)
#             print("best model",best_model)
#             r2=r2_score(y_test,predicted)
#             return r2
#         except Exception as a:
#             raise CustomException(a,sys)
# model_trainer.py
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import os
import sys
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
            }

            model_report = evaluate_models(x_train, y_train, x_test, y_test, models)

            print("\n===== Model Performance Report =====")
            for model_name, scores in model_report.items():
                print(f"{model_name}:")
                print(f"  Train R²     : {scores['train_r2']:.4f}")
                print(f"  Test R²      : {scores['test_r2']:.4f}")
                print(f"  Test MAE     : {scores['test_mae']:.4f}")
                print(f"  Test RMSE    : {scores['test_rmse']:.4f}")
                print("-" * 40)

            best_model_name = max(model_report, key=lambda k: model_report[k]['test_r2'])
            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["test_r2"]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            print(f"Best Model: {best_model_name} with R²: {best_model_score:.4f}")
            return r2_score(y_test, predicted)

        except Exception as a:
            raise CustomException(a, sys)
