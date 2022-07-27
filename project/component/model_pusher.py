from project.logger import logging 
from project.exception import ProjectException 
from project.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from project.entity.config_entity import ModelPusherConfig

import os, sys, shutil 


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config=model_pusher_config
            self.model_evaluation_artifact= model_evaluation_artifact
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def export_model(self):
        try:
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            export_dir = self.model_pusher_config.export_dir_path 
            
            model_file_name = os.path.basename(evaluated_model_file_path)
            export_model_file_path = os.path.join(export_dir, model_file_name)
            os.makedirs(export_dir, exist_ok=True)
            # print(f"---model pusher----export_model---->>>>>>{evaluated_model_file_path}<<<<<<<<_----------evaluated_model_file_path----------")
            print(f"---model pusher----export_model---->>>>>>{export_model_file_path}<<<<<<<<_---------export_model_file_path----------")
    
            shutil.copy(src=evaluated_model_file_path, dst=export_model_file_path)
            
            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,
                                                        export_model_file_path=export_model_file_path)
            return model_pusher_artifact             
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def initiate_model_pusher(self):
        try:
            return self.export_model()
        except Exception as e:
            raise ProjectException(e, sys) from e 
            
            
    def __del__(self):
        logging.info(f"{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")