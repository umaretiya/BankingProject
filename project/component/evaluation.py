from project.logger import logging
from project.exception import ProjectException 
from project.constant import * 
from project.model.model_factory import evaluate_regression_model
from project.entity.config_entity import ModelEvaluationConfig
from project.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from project.utils import write_yaml,read_yaml,load_object,load_data

import os, sys,numpy as np


class ModelEvaluation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact,
                 model_evaluation_config:ModelEvaluationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact= data_validation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_evaluation_config=model_evaluation_config
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            # dx = {"best_model":{"model_path":self.model_trainer_artifact.trained_model_file_path}}
            if not os.path.exists(model_evaluation_file_path):
                write_yaml(file_path= model_evaluation_file_path, data= dict())
                return model 
            
            model_eval_file_content = read_yaml(file_path=model_evaluation_file_path)
            
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content
            
            if BEST_MODEL_KEY not in model_eval_file_content:
                return model
            # print(f"---model evaluation----getbestmodel---->>>>>>{model}<<<<<<<<_-----------best-model-----------")
            
            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model 
            
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def update_evaluation_report(self, model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content= read_yaml(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]
                
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }
            
            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)
                    
            model_eval_content.update(eval_result)
            write_yaml(file_path=eval_file_path, data=model_eval_content)
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
        
    def initiate_model_evaluation(self):
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)
            print(f"---model evaluation----trained_model_object---->>>>>>{trained_model_object}<<<<<<<<_----------trained_model_object-----------")
            
            train_file_path= self.data_ingestion_artifact.train_file_path
            test_file_path= self.data_ingestion_artifact.test_file_path
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            
            train_dataframe = load_data(file_path=train_file_path,
                                        schema_file_path=schema_file_path)
            
            test_dataframe = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            
            schema_content = read_yaml(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]
            
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])

            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)            
            
            model = self.get_best_model()
            print(f"---model evaluation----trained_model_object---->>>>>>{model}<<<<<<<<_----------model-----------")
            
            if model is None:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                return model_evaluation_artifact
            
            model_list = [model, trained_model_object]
            
            metric_info_artifact = evaluate_regression_model(model_list=model_list,
                                                             X_train=train_dataframe,
                                                             y_train=train_target_arr,
                                                             X_test=test_dataframe,
                                                             y_test=test_target_arr,
                                                             base_accuracy=self.model_trainer_artifact.model_accuracy)
            
            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path)
                return response 
            
            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted= True)
                self.update_evaluation_report(model_evaluation_artifact)
            else:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)

            return model_evaluation_artifact 
        except Exception as e:
            raise ProjectException(e, sys) from e 
        

    def __del__(self):
            logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")
            