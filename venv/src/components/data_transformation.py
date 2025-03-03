import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
import numpy as np

class DataTrans:
    def __init__(self):
        pass

    def get_data_trans_obj(self):
        '''
        This function returns a ColumnTransformer object which performs:
        - Imputation for missing values
        - OneHotEncoding for categorical columns
        - StandardScaling for numerical columns
        '''

        # Define numerical and categorical columns
        num_col = ["reading_score", "writing_score"]
        cat_col = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

        # Numerical pipeline
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical pipeline
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
        ])

        # Combine both pipelines into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, num_col),
                ("cat_pipeline", cat_pipeline, cat_col)
            ]
        )
        return preprocessor

    def initiate_data_trans(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = "math_score"

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Get preprocessing object (fixing function call)
            preprocessing_obj = self.get_data_trans_obj()

            # Transform data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target into arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            return train_arr, test_arr, preprocessing_obj

        except Exception as e:
            raise CustomException(e, sys)

  


