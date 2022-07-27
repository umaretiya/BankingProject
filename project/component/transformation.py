from importlib import import_module
from project.exception import ProjectException 
from project.logger import logging 
from project.entity.config_entity import DataTransformationConfig
from project.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from project.utils import load_object, read_yaml,save_numpy_array,save_object,load_data 
import os, sys 

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline 
from project.constant import *
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer  
import numpy as np 


class DataTransformation:
    def __init__(self,
                 data_transformation_config:DataTransformationConfig,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def get_data_transformer_object(self):
        try:
            # schema_file_path = self.data_validation_artifact.schema_file_path
            # dataset_schema = read_yaml(file_path=schema_file_path)
            
            # numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            # categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            
            # cat_pipeline= Pipeline(steps=[
            #     ("impute", SimpleImputer(strategy="most_frequent")),
            #     ("one_hot_encoder", OneHotEncoder()),
            #     ("scaler", StandardScaler(with_mean=False)),
            # ])
             
            # preprocessing_obj = ColumnTransformer([
            #     ("num_pipeline", num_pipeline, numerical_columns),
            #     ("cat_pipeline", cat_pipeline, categorical_columns),
            #     ]) 
            return num_pipeline
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def initiate_data_transformation(self):
        try:
            preprocessing_obj = self.get_data_transformer_object()
            
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            
            
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            
            schema = read_yaml(file_path=schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]
           
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns =[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
           
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            
            train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name= os.path.basename(test_file_path).replace(".csv", ".npz")
            
            transformed_train_file_path=os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path=os.path.join(transformed_test_dir, test_file_name)
            
            save_numpy_array(file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array(file_path=transformed_test_file_path, array=test_arr)
            
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path
            
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)
            
            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=True,
                message="Data transformation successfull.",
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            raise ProjectException(e, sys) from e 
        

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")

        
