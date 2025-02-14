import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.utils import evaluate_models
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting train and test data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostClassifier()
            }
            param_grids = {
                "Random Forest": {
                    'n_estimators': [50, 100, 150, 200],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Decision Tree": {
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 10]
                },
                "Linear Regression": {
                    'fit_intercept': [True, False]
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski', 'euclidean']
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                },
                "CatBoost": {
                    'iterations': [50, 100, 150],
                    'learning_rate': [0.01, 0.1],
                    'depth': [3, 5],
                    'l2_leaf_reg': [1, 3]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                }
            }

            model_report:dict = evaluate_models(x_train, y_train, x_test, y_test, models, param_grids)
            
            best_r2_score = max(sorted(model_report.values()))
            best_model_name = max(model_report, key=model_report.get)

            best_model =  models[best_model_name]

            if best_r2_score < 0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best Model Found on training and testing data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path, # File path of where to save the pkl file
                obj = best_model  # Pickle file of the best model
            )   

            # predicted = best_model.predict(x_test)
            # r2_square = r2_score(y_test, predicted)
            

            return best_r2_score


        except Exception as e:
            raise CustomException(e, sys)
             