
import numpy as np, pandas as pd
from typing import List

from project.constant import CONFIG_FILE_PATH,CURRENT_TIME_STAMP
from project.configured import Configuration
from project.component.ingestion import DataIngestion
from project.component.validation import DataValidation
from project.component.transformation import DataTransformation
from project.component.training import ModelTrainer


from project.utils import read_yaml,load_numpy_array, load_object

from project.model.model_factory_config import GridSearchedBestModel, InitializedModelDetail
from project.model.estimator_model import ProjectEstimatorModel
from project.model.model_factory import evaluate_regression_model, ModelFactory
from project.model.model_predictor import ProjectData,ProjectPredictor
from project.constant import * 



conf = Configuration(CONFIG_FILE_PATH, CURRENT_TIME_STAMP)
ingest = conf.get_data_ingestion_config()
validate = conf.get_data_validation_config()
transform = conf.get_data_transformation_config()
model_train= conf.get_model_trainer_config() 
model_eval= conf.get_model_evaluation_config()
model_push= conf.get_model_pusher_config()

data_ingested = DataIngestion(data_ingestion_config=ingest)
data_ingested_start = data_ingested.initiate_data_ingestion()
print(f"------trainfile path----<><><>{data_ingested_start.train_file_path}--------data_ingested_start.train_file_path------")
trained_file = pd.read_csv(data_ingested_start.train_file_path)
X = trained_file.drop(labels=['default'], axis=1)
data_validate = DataValidation(data_ingestion_artifact=data_ingested_start, data_validation_config=validate)
data_validate_start = data_validate.initiate_data_validation()
print(X.columns.to_list())
data_transformation = DataTransformation(data_transformation_config=transform, data_validation_artifact=data_validate_start,data_ingestion_artifact=data_ingested_start)
data_transformation_start = data_transformation.initiate_data_transformation()

model_trainer = ModelTrainer(model_trainer_config=model_train, data_transformation_artifact=data_transformation_start)
model_trainer_start = model_trainer.initiate_model_trainer()

prepro_obj = data_transformation.get_data_transformer_object() #call it
    
train_arr = data_transformation_start.transformed_train_file_path
train_array = load_numpy_array(train_arr)
test_arr = data_transformation_start.transformed_test_file_path
test_array = load_numpy_array(test_arr)

X_train,y_train = train_array[:,:-1],train_array[:,-1]
X_test,y_test = test_array[:,:-1],test_array[:,-1]
    
model_config_path = model_train.model_config_file_path
model_factory = ModelFactory(model_config_path=model_config_path)
initialized_model_list = model_factory.get_initialized_model_list()
lists_of_model = [initialized_model_list[0]._asdict()['model'].fit(X_train,y_train), initialized_model_list[0]._asdict()['model'].fit(X_train,y_train)]
metric_info = evaluate_regression_model(model_list=lists_of_model,X_train=X_train,y_train=y_train, X_test=X_test,y_test=y_test,base_accuracy=0.4)

prepro_obj.fit_transform(X)
project_model = ProjectEstimatorModel(preprocessing_object=prepro_obj, trained_model_object=metric_info.model_object)

inputs = [28466.0,240000.0,2.0,1.0,1.0,40.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
df_np = np.array(inputs).reshape(1,len(inputs))

predict = project_model.predict(df_np)

print(f"----{predict}-------<<<>>> --------------------------predict------------")