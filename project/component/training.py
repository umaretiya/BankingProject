from project.exception import ProjectException 
from project.logger import logging 
from project.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from project.entity.config_entity import ModelTrainerConfig 
from project.utils import load_numpy_array,save_object,load_object

from project.model.estimator_model import ProjectEstimatorModel
from project.model.model_factory import evaluate_regression_model, ModelFactory
from project.model.model_factory_config import MetricInfoArtifact, GridSearchedBestModel

import sys, os 
from typing import List 



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact= data_transformation_artifact
        except Exception as e:
            raise ProjectException(e,sys) from e 
        
    
    def initiate_model_trainer(self):
        try:
            transfomed_train_file_path= self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array(file_path=transfomed_train_file_path)
            
            transformed_test_file_path= self.data_transformation_artifact.transformed_test_file_path
            test_array= load_numpy_array(file_path=transformed_test_file_path)
            
            x_train,y_train,x_test,y_test= train_array[:,:-1],train_array[:,-1], test_array[:,:-1], test_array[:,-1]
            
            model_config_file_path = self.model_trainer_config.model_config_file_path
            
            model_factory = ModelFactory(model_config_path=model_config_file_path)
            
            base_accuracy = self.model_trainer_config.base_accuracy 
            
            best_model = model_factory.get_best_model(X=x_train,y=y_train, base_accuracy=base_accuracy)
            logging.info(f"Best model found on training data: {best_model}")
            
            grid_searched_best_model_list:List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list
            
            model_list = [model.best_model for model in grid_searched_best_model_list]
            # print(f"----model_list--------{model_list}<<<<_____--inititate model trainer----")
            metric_info:MetricInfoArtifact = evaluate_regression_model(model_list=model_list, X_train=x_train,y_train=y_train, X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)
            # print(f"----------model_list--------{metric_info}<<<<_____--inititate model trainer----------")
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            
            model_object = metric_info.model_object
            
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            project_model = ProjectEstimatorModel(preprocessing_object=preprocessing_obj, trained_model_object=model_object)
            
            save_object(file_path=trained_model_file_path, obj=project_model)
            
            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message="Model Trained successfully",
                trained_model_file_path=trained_model_file_path,
                train_rmse=metric_info.train_rmse,
                test_rmse=metric_info.test_rmse,
                train_accuracy=metric_info.train_accuracy,
                test_accuracy=metric_info.test_accuracy,
                model_accuracy=metric_info.model_accuracy
            )
            return model_trainer_artifact 
        except Exception as e:
            raise ProjectException(e, sys) from e 
        

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")

        
