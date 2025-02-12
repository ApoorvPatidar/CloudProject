import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # ColumnTransfomer is basically used to make the pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get__data_transfomer_object(self):
        try:
            numerical_cols = ["writing_score", "reading_score"]
            categorical_cols = ['gender' ,'race_ethnicity' ,'parental_level_of_education' ,'lunch' ,'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median'))  # Replacing the missing value with median
                    ('scaler', StandardScaler())
                    ]
                )
            
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')) # Replacing the missing values with the mode i.e most frequent values
                    ('one_hot_encoder', OneHotEncoder())
                    ('scaler', StandardScaler())
                    ]
                )
            
            logging.info(f"Catergorical Columns: {categorical_cols}")
            logging.info(f"Numerical Columns: {numerical_cols} ")

            preprocessor = ColumnTransformer(
                transformers = [
                    ('num_pipeline', num_pipeline, numerical_cols),# (pipeline_name, preprocessing_pipeline / transformer, target_columns)
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                    ]
                )
            
            return preprocessor
        except:
            pass

            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_df)
            test_df = pd.read_csv(test_df)

            logging.info("Read the Training and Testing Data")

            logging.info("Opening preprocessing object")

            preprocessing_obj = self.get_data_transfomer_object()

            target_col_name = 'math_score'
            numerical_cols = ["writing_score", "reading_score"]


            def separate_features_and_target(df):
                input_features = df.drop(target_col_name, axis=1)
                target_feature = df[target_col_name]
                return input_features, target_feature


            input_feature_train_df, target_feature_train_df = separate_features_and_target(train_df)
            input_feature_test_df, target_feature_test_df = separate_features_and_target(test_df)

            logging.info("Applying the preprocessing object on the training and testing data")

            # fit_transform is used to learn the patterns form the data and then implement those learnings on the data to handle sacaling, encoding, missignValues etc
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # np.c_ concatinates the arrays column wise and converts the pandas dataframe into a column array if it not before concatinating 
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Saved the processing object. ")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path) # Returns train, test, pkl file
        
        except:
            pass

