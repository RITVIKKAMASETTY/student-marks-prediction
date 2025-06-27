import os
import sys
import numpy as np
import dill
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as a:
        raise CustomException(a, sys)


# def evaluate_models(x_train, y_train, x_test, y_test, models):
#     try:
#         report = {}
#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             model.fit(x_train, y_train)
#             y_train_pred = model.predict(x_train)
#             y_test_pred = model.predict(x_test)
#             model_train_mae = mean_absolute_error(y_train, y_train_pred)
#             model_train_mse = mean_squared_error(y_train, y_train_pred)
#             model_train_rmse = np.sqrt(model_train_mse)
#             model_train_r2 = r2_score(y_train, y_train_pred)
#             model_test_mae = mean_absolute_error(y_test, y_test_pred)
#             model_test_mse = mean_squared_error(y_test, y_test_pred)
#             model_test_rmse = np.sqrt(model_test_mse)
#             model_test_r2 = r2_score(y_test, y_test_pred)
#             report[list(models.keys())[i]] = [
#                 model_train_mae,
#                 model_train_mse,
#                 model_train_rmse,
#                 model_train_r2,
#                 model_test_mae,
#                 model_test_mse,
#                 model_test_rmse,
#                 model_test_r2,
#             ]
#         return report
#     except Exception as a:
#         raise CustomException(a, sys)
# utils.py
import numpy as np
import os
import sys
import dill
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as a:
        raise CustomException(a, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        # Define hyperparameter grid for each model
        params = {
            "Linear Regression": {},
            "Ridge": {"alpha": [0.1, 1.0, 10.0]},
            "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
            "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7]},
            "Decision Tree": {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
            "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 5]},
            "XGBRegressor": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        }

        for name, model in models.items():
            print(f"Training and tuning {name}...")
            param_grid = params.get(name, {})

            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
                grid_search.fit(x_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                model.fit(x_train, y_train)
                best_model = model

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            model_train_mae = mean_absolute_error(y_train, y_train_pred)
            model_train_mse = mean_squared_error(y_train, y_train_pred)
            model_train_rmse = np.sqrt(model_train_mse)
            model_train_r2 = r2_score(y_train, y_train_pred)

            model_test_mae = mean_absolute_error(y_test, y_test_pred)
            model_test_mse = mean_squared_error(y_test, y_test_pred)
            model_test_rmse = np.sqrt(model_test_mse)
            model_test_r2 = r2_score(y_test, y_test_pred)

            report[name] = {
                "model": best_model,
                "train_mae": model_train_mae,
                "train_mse": model_train_mse,
                "train_rmse": model_train_rmse,
                "train_r2": model_train_r2,
                "test_mae": model_test_mae,
                "test_mse": model_test_mse,
                "test_rmse": model_test_rmse,
                "test_r2": model_test_r2,
            }

        return report
    except Exception as a:
        raise CustomException(a, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as a:
        raise CustomException(a, sys)
