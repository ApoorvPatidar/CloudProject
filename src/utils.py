import os
import sys
import numpy as np
import pandas as pd
import dill
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)


    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param_grids):
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Applying the GridSearch on {model_name}")
            if model_name in param_grids:
                grid_search = GridSearchCV(model,param_grids[model_name], scoring = 'r2', cv=2, n_jobs=-1)
                grid_search.fit(x_train, y_train)
                best_model = grid_search.best_estimator_

                logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
                logging.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

                train_pred = best_model.predict(x_train) # Prediction of the training data
                test_pred = best_model.predict(x_test)   # Prediction of the testing data

                logging.info(f"train_pred shape: {train_pred.shape}")
                logging.info(f"test_pred shape: {test_pred.shape}")

                train_model_score = r2_score(y_train, train_pred)  # R2 score of the training data
                test_model_score = r2_score(y_test, test_pred)     # R2 score of the testing data

                report[model_name] = test_model_score
        
        return report 


    except Exception as e:
        raise CustomException(e, sys)
        


