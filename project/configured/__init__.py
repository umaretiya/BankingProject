import imp
from project.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from project.entity.config_entity import ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig,TrainingPipelineConfig
import os, sys
from project.exception import ProjectException
from project.logger import logging 
from project.constant import * 
from project.utils import read_yaml 


class Configuration:
    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 current_time_stamp=CURRENT_TIME_STAMP):
        try:
            self.config_info = read_yaml(file_path=config_file_path)
            self.training_pipeline_config = self.get_trainig_pipeline_config()
            self.time_stamp= current_time_stamp
            logging.info("Configuaraion initiated")
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
        
    def get_data_ingestion_config(self):
        try:
            artifact_dir=self.training_pipeline_config.artifact_dir
            
            data_ingestion_artifact_dir = os.path.join(
                artifact_dir, 
                DATA_INGESTION_ARTIFACT_DIR, 
                self.time_stamp
            )
            
            data_ingestion_info= self.config_info[DATA_INGESTION_CONFIG_KEY]
            
            dataset_download_url=data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]
            
            tgz_file_name=data_ingestion_info[DATA_INGESTION_TGZ_DOWNLOAD_FILE_NAME_KEY]
            
            tgz_download_dir= os.path.join(data_ingestion_artifact_dir,
                                           data_ingestion_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY])
            
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
                                        data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])
            
            ingested_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY]
            )
            
            ingested_train_dir= os.path.join(
                ingested_data_dir,
                data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY]
            )
            
            ingested_test_dir = os.path.join(
                ingested_data_dir,
                data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY]
            )
            logging.info("Configuaraion: Data ingestion_configured")
            data_ingestion_config = DataIngestionConfig(dataset_download_url=dataset_download_url,
                                                        tgz_file_name=tgz_file_name,
                                                        tgz_download_dir=tgz_download_dir,
                                                        raw_data_dir=raw_data_dir,
                                                        ingested_train_dir=ingested_train_dir,
                                                        ingested_test_dir=ingested_test_dir)
            return data_ingestion_config
        except Exception as e:
            raise ProjectException(e, sys) from e 


    def get_data_validation_config(self):
        try:
            artifact_dir=self.training_pipeline_config.artifact_dir
            
            data_validation_artifact_dir=os.path.join(
                artifact_dir,
                DATA_VALIDATION_ARTIFACT_DIR_NAME,
                self.time_stamp
            )
            
            data_validation_config= self.config_info[DATA_VALIDATION_CONFIG_KEY]
            
            schema_file_path= os.path.join(ROOT_DIR,
                                           data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                                           data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])
            
            report_file_path=os.path.join(data_validation_artifact_dir,
                                          data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY])
            
            report_page_file_path= os.path.join(data_validation_artifact_dir,
                                                data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY])
            
            logging.info("Configuaraion: Data validation configured")
            data_validation_configs= DataValidationConfig(
                schema_file_path=schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path,
            )
            return data_validation_configs
            
        except Exception as e:
            raise ProjectException(e, sys) from e 


    def get_data_transformation_config(self):
        try:
            artifact_dir=self.training_pipeline_config.artifact_dir
            
            data_transformation_artifact_dir=os.path.join(
                artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR,
                self.time_stamp
            )
            
            data_transformation_config_info=self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            
            add_bedroom_per_room=data_transformation_config_info[DATA_TRANSFORMATION_ADD_BEDROOM]
            
            preprocessed_object_file_path=os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY]
            )
            
            transformed_train_dir=os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
            )
            
            transformed_test_dir=os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_TEST_DIR_NAME]
            )
            
            logging.info("Configuaraion: Transformation configured")
            data_transformation_config=DataTransformationConfig(
                add_bedroom_per_room=add_bedroom_per_room,
                preprocessed_object_file_path=preprocessed_object_file_path,
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir
            )
            
            return data_transformation_config

        except Exception as e:
            raise ProjectException(e, sys) from e 


    def get_model_trainer_config(self):
        try:
            artifact_dir= self.training_pipeline_config.artifact_dir
            
            model_trainer_artifact_dir=os.path.join(
                artifact_dir,
                MODEL_TRAINER_ARTIFACT_DIR,
                self.time_stamp
            )
            
            model_trainer_config_info= self.config_info[MODEL_TRAINER_CONFIG_KEY]
            
            trained_model_file_path=os.path.join(model_trainer_artifact_dir,
                                                 model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],
                                                 model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY])
            
            model_config_file_path= os.path.join(
                model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],
                model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY],
            )
            
            base_accuracy=model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]
            
            logging.info("Configuaraion: Modeltrainign configured")
            model_trainer_config=ModelTrainerConfig(trained_model_file_path=trained_model_file_path,
                                                    base_accuracy=base_accuracy,
                                                    model_config_file_path=model_config_file_path)
            
            return model_trainer_config
            
        except Exception as e:
            raise ProjectException(e, sys) from e 
        

    def get_model_evaluation_config(self):
        try:
            model_evaluation_config= self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            
            artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                        MODEL_EVALUATION_ARTIFACT_DIR)
            
            model_evaluation_file_path= os.path.join(artifact_dir,
                                                     model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY])
            
            logging.info("Configuaraion: Model Evaluated configured")
            model_evaluated = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                                    time_stamp=self.time_stamp)
            
            return model_evaluated
        except Exception as e:
            raise ProjectException(e, sys) from e 



    def get_model_pusher_config(self):
        try:
            time_stamp=f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            model_pusher_config_info=self.config_info[MODEL_PUSHER_CONFIG_KEY]
            
            export_dir_path=os.path.join(ROOT_DIR, 
                                         model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                         time_stamp)
            
            logging.info("Configuaraion: Model pusher configured")
            model_pusher_config= ModelPusherConfig(export_dir_path=export_dir_path)
            
            return model_pusher_config
        except Exception as e:
            raise ProjectException(e, sys) from e 
        

    def get_trainig_pipeline_config(self):
        try:
            training_pipeline_config=self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR,
                                        training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            training_pipeline_config= TrainingPipelineConfig(artifact_dir=artifact_dir)
            
            logging.info("Configuaraion: Training pipeline configured")
            return training_pipeline_config
        except Exception as e:
            raise ProjectException(e, sys) from e 
        