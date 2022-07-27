
import numpy as np
from typing import List

from project.constant import CONFIG_FILE_PATH,CURRENT_TIME_STAMP
from project.configured import Configuration
from project.component.ingestion import DataIngestion
from project.component.validation import DataValidation
from project.component.transformation import DataTransformation
from project.component.training import ModelTrainer
from project.component.evaluation import ModelEvaluation
from project.component.model_pusher import ModelPusher

from project.utils import read_yaml,load_numpy_array, load_object

from project.model.model_factory_config import GridSearchedBestModel, InitializedModelDetail
from project.model.estimator_model import ProjectEstimatorModel
from project.model.model_factory import evaluate_regression_model, ModelFactory
from project.model.model_predictor import ProjectData,ProjectPredictor
from project.constant import * 


def home():
    # file_path = os.getcwd()    
    # files = get_sample_model_config_yaml_file(file_path)
    # print(files)
    conf = Configuration(CONFIG_FILE_PATH, CURRENT_TIME_STAMP)
    ingest = conf.get_data_ingestion_config()
    validate = conf.get_data_validation_config()
    transform = conf.get_data_transformation_config()
    model_train= conf.get_model_trainer_config() 
    model_eval= conf.get_model_evaluation_config()
    model_push= conf.get_model_pusher_config()
    
    data_ingested = DataIngestion(data_ingestion_config=ingest)
    data_ingested_start = data_ingested.initiate_data_ingestion()
 
    data_validate = DataValidation(data_ingestion_artifact=data_ingested_start, data_validation_config=validate)
    data_validate_start = data_validate.initiate_data_validation()
    
    data_transformation = DataTransformation(data_transformation_config=transform, data_validation_artifact=data_validate_start,data_ingestion_artifact=data_ingested_start)
    data_transformation_start = data_transformation.initiate_data_transformation()
    
    model_trainer = ModelTrainer(model_trainer_config=model_train, data_transformation_artifact=data_transformation_start)
    model_trainer_start = model_trainer.initiate_model_trainer()
    
    model_evaluation = ModelEvaluation(data_ingestion_artifact=data_ingested_start, data_validation_artifact=data_validate_start, model_trainer_artifact=model_trainer_start, model_evaluation_config=model_eval)
    model_evaluation_start = model_evaluation.initiate_model_evaluation()
    
    
    model_pusher = ModelPusher(model_pusher_config=model_push, model_evaluation_artifact=model_evaluation_start)
    model_pusher_start = model_pusher.initiate_model_pusher()
    
    
    # print(model_pusher_start.is_model_pusher)
    # print(model_evaluation.get_best_model())
    # preprocessed_obj_path = data_transformation_start.preprocessed_object_file_path
    # preprocessed_obj = load_object(file_path=preprocessed_obj_path)
    
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
    
    # best_model = model_factory.get_best_model(X=X_train,y=y_train, base_accuracy=0.4)
    # lists_0 = model_factory.execute_grid_search_operation(initialized_model=initialized_model_list[0], input_feature=X_train, output_feature=y_train)
    # lists_1 = model_factory.execute_grid_search_operation(initialized_model=initialized_model_list[1], input_feature=X_train, output_feature=y_train)
    
    # model_inititalized = model_factory.get_initialized_model_list()
    
    # model_factory.get_best_model_from_grid_searched_best_model_list()
    print(f"-----demopy----initializedmodellist-<><><><>{initialized_model_list}---------initialized_model_list-------<><><><>>")
    print(f"-----demopy----GridSearchdBestmodel-<><><><>{List[GridSearchedBestModel]}---------gridserachmodel list-------<><><><>>")
    # best_model = model_factory.get_best_model(X_train, y_train, base_accuracy=0.4)
    
    gridsearched_model_list = model_factory.grid_searched_best_model_list
    grid_searched_best_model_list:List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list
    # model_list = [model.best_model for model in gridsearched_model_list]
    
    
    # print(f"----..demopy..---->>>{model_list}<<<<---modellist--------")
    # print(f"----.demopy...-,.,.<><><><><>-->>>{best_model}<<<<---bestmodel---------")
    # print(f"----.demopy..gridsearched_model_list.---->>>{grid_searched_best_model_list}<<<<---gridsearched_model_list---------")
    # # print(f"---model config--paht --{model_config_path}-----<<>>>---")
    
    # print(f"-----demopy----executegrdsearch ope-<><><><>{lists_0}---------lists---0----<><><><>>")
    # print(f"-----demopy----executegrdsearch ope-<><><><>{lists_1}---------lists---1----<><><><>>")
    prepro_obj.fit_transform(X_train)
    print(f"----demopy----metricinfo--<<<{metric_info}>>>>-----metricinfo----------")
    # model_path = model_pusher_start.export_model_file_path
    # # model_obj = load_object(model_path)
    # with open(model_path, 'rb') as fiels:
    #     model_obj = dill.load(fiels)
        
    # eval_model_path = model_evaluation_start.evaluated_model_path
    # # eval_model_obj = load_object(eval_model_path)
    # with open(eval_model_path, 'rb') as fiels01:
    #     eval_model_obj = dill.load(fiels01)
        
    # print(f"---demopy--->>>>>>>>>{model_obj}<<<<----------model_obj--------------")
    # print(f"---demopy--->>>>>>>>>{eval_model_obj}<<<<----------model_obj--------------")
    
    project_model = ProjectEstimatorModel(preprocessing_object=prepro_obj, trained_model_object=metric_info.model_object)
    # print(f"---demopy--->>>>>>>>>{model_path}<<<<-----model_path-----")
    # print(f"---demopy--none->>>>>>>>>{preprocessed_obj}<<<<-----preprocessed_obj--its shos none---")
    # print(f"---demopy---prepro_obj>>>>>>>>>{prepro_obj}<<<<-------prepro_obj--------")
    
    # pred_models = ProjectPredictor(model_dir=MODEL_DIR)
    # print(f"---demopy---ProjectPredictor>>>>>>>>>{model_push.export_dir_path}<<<<-------model_push.export_dir_path)-------")
    # print(f"---demopy---ProjectPredictor>>>>>>>>>{pred_models}<<<<-------pred_model--------")
    

    # print(f"---demopy---ProjectData>>>>>>>>>{pred_data}<<<<-------pred_data-------")
    # print(f"---demopy---ProjectData get df>>>>>>>>>{user_df}<<<<-------user_df------")
    # target = pred_models.predict(user_df)
    # pred_models.get_latest_model_path
    # print(f"---demopy---ProjectData get df>>>>>>>>>{pred_models.get_latest_model_path()}<<<<-------target------")
    # model = load_object(pred_models.get_latest_model_path())
    # print(f"---demopy---ProjectData get df>>>>>>>>>{model}<<<<-------target------")
    
    # print("ingestion_config", ingest ,"\n")
    # print("transformation_config", transform ,"\n")
    # print("validataion_config", validate ,"\n")
    # print("model_trainer_config", model_train ,"\n")
    # print("model_evaluation_config", model_eval ,"\n")
    # print("model_pusher_config", model_push ,"\n")

    inputs = [28466.0,240000.0,2.0,1.0,1.0,40.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    df_np = np.array(inputs).reshape(1,len(inputs))
    # print(f"---demopy--->>>>>>>>>{df_np}<<<<-----preprocessed_obj-----")
    print(f"--demopy ---project estimator----{project_model}---<<<>>>---<><><----------")
    predict = project_model.predict(df_np)
    # predict = model_obj.predict(df_np)
    # eval_model_obj = eval_model_obj.fit(X_train,y_train)
    # push_predict = model_obj.predict(df_np)
    # predict = eval_model_obj.predict(df_np)
    print(f"---demopy---predict>>>>>>>>>{predict}<<<<--------predict--eval object---")
    # print(f"---demopy---predict>>>>>>>>>{project_model}<<<<--------predict-----")
    
    
    data={
        "data_ingested":data_ingested,
        "data_ingested_start": data_ingested_start,
        "data_validate":data_validate,
        "data_validate_start":data_validate_start,
        "data_transformation":data_transformation,
        "data_transformation_start":data_transformation_start,
        "model_trainer":model_trainer,
        "model_trainer_start":model_trainer_start,
        "model_evaluation":model_evaluation,
        "model_evaluation_start":model_evaluation_start,
        "model_pusher":model_pusher,
        "model_pusher_start":model_pusher_start,
    }
    
    title = {
        "model_pushed":model_pusher_start.is_model_pusher,
        "model_accepted":model_evaluation_start.is_model_accepted,
        "model_best":model_evaluation.get_best_model(),
        "traine_model":model_trainer_start._asdict(),
        # "metric_info":metric_info,
        # "predict":predict,
    }
    
    context={
        "ingest":ingest,
        "validate":validate,
        "transform":transform,
        "model_train":model_train,
        "model_eval":model_eval,
        "model_push":model_push
            }
    return initialized_model_list
    # return render_template("test.html", context=context, data=data, title=title)
inputs = [28466.0,240000.0,2.0,1.0,1.0,40.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
df_np = np.array(inputs).reshape(1,len(inputs))

obj = home()
# x = obj.predict(df_np)
print(f"-----c--->>>>>>>{obj[0]._asdict()['model']}<<<<<<<<<<<<---as dict----module1----home()--")
print(" \n\n")
print(f"-----c--->>>>>>>{obj[1]._asdict()['model']}<<<<<<<<<<<<--ad_dict----module2-----home()--")