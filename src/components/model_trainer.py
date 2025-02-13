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
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Spliting train and test data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, :-1]
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

            best_model = {}

            for model_name, model in models.items():
                logging.info(f"Applying the GridSearch on {model_name}")
                if model_name in param_grids:
                    grid_search = GridSearchCV(model, param_grids[model_name], scoring = 'r2', cv=3, n_jobs=-1)
                    grid_search.fit(x_train, y_train)
                    best_param = grid_search.best_params_
                    best_param.predict(x_train)



        except Exception as e:
            raise CustomException(e, sys)
             