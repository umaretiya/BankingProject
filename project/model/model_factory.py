import importlib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,f1_score
import yaml, os, sys 
from project.exception import ProjectException 
from project.logger import logging
from typing import List 
from project.model.model_factory_config import InitializedModelDetail,GridSearchedBestModel,BestModel,MetricInfoArtifact

GRID_SEARCH_KEY = "grid_search"
MODULE_KEY="module"
CLASS_KEY="class"
PARAM_KEY="params"
MODEL_SELECTION_KEY="model_selection"
SEARCH_PARAM_GRID_KEY="search_param_grid"

def get_sample_model_config_yaml_file(export_dir):
    try:
        model_config={
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY:"GridSearchCV",
                PARAM_KEY:{
                    "cv":3,
                    "verbose": 1
                }
            },
            MODEL_SELECTION_KEY: {
                "module_0":{
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY:
                        {
                            "param_name1": "value1",
                            "param_name1": "value2",
                        },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ["param_value_1", "param_value_2"]
                    }
                },
            }
        }
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        
        with open(export_file_path, "w") as file:
            yaml.dump(model_config, file)
        return export_file_path
    
    except Exception as e:
        raise ProcessLookupError(e, sys) from e 
    


def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6):
    try:
        index_number = 0
        metric_info_artifact= None 
        for model in model_list:
            model_name = str(model)
            print(f"-------modelfactory--model name->>>>>>>{model_name}<<<<<<<<<-------evaluateregression model---------")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # train_acc = r2_score(y_train,y_train_pred)
            # test_acc = r2_score(y_test,y_test_pred)
            
            train_acc = accuracy_score(y_train,y_train_pred)
            test_acc = accuracy_score(y_test,y_test_pred)
            
            # train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
            # test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
            
            train_rmse = f1_score(y_train,y_train_pred)
            test_rmse = f1_score(y_test,y_test_pred)
            
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)
            # print(f"-------modelfactory--rmse: {train_rmse}->>>>>>>{model_accuracy}<<<<<<<<<------modelaccuracy-accuracy---------")
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                print(f"-------modelfactory--modelaccuracy ---: {model}->>>>>>>{model_accuracy}<<<<<<<<<------model------------")
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                          model_object=model,
                                                          train_rmse=train_rmse,
                                                          test_rmse=test_rmse,
                                                          train_accuracy=train_acc,
                                                          test_accuracy=test_acc,
                                                          model_accuracy=model_accuracy,
                                                          index_number=index_number)
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy")
        print(f"---metric info artifact------>>>>>--{metric_info_artifact}<<<-----regression model return--<<")
        return metric_info_artifact
            
    except Exception as e:
        raise ProcessLookupError(e, sys) from e 
    
    

class ModelFactory:
    def __init__(self, model_config_path=None):
        try:
            self.config = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module =self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data= dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])
            
            self.models_initialization_config= dict(self.config[MODEL_SELECTION_KEY])
            
            self.initialized_model_list= None
            self.grid_searched_best_model_list= None 
            
        except Exception as e:
            raise ProjectException(e, sys) from e
    
    
    @staticmethod
    def update_propery_of_class(instance_ref:object, propery_data:dict):
        try:
            if not isinstance(propery_data, dict):
                raise Exception("Property_data parameter required to dictionary")
            print(propery_data)
            for key,value in propery_data.items():
                setattr(instance_ref, key,value)
            return instance_ref
        except Exception as e:
            raise ProjectException(e, sys)
        
        
        
    @staticmethod
    def read_params(config_path):
        try:
            print(f"-------modelfactory-read_params->>>>>>>{config_path}<<<<<<<<<-------config path---------")
            
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)
            return config 
        except Exception as e:
            raise ProjectException(e, sys) from e 
    
    
    @staticmethod    
    def class_for_name(module_name, class_name):
        try:
            print(f"-------modelfactory--class for name->>>>>>>{module_name}<<<<<<<<<-------module name---------")
            
            module = importlib.import_module(module_name)
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise ProjectException(e, sys) from e 
    
    
    def execute_grid_search_operation(self, initialized_model:InitializedModelDetail,
                                      input_feature, output_feature):
        try:
            # print(f"-------modelfactory--execute grid search->>>>>>>{initialized_model}<<<<<<<<<-------initialized_model---------")
            
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name)
            print(f"-------modelfactory--execute grid search->>>>>>>{initialized_model.model_name}<<<<<<<<<-------initialized_model.modelname---------")
            
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_propery_of_class(grid_search_cv, self.grid_search_property_data)
            
            message = f"{'>>'*30} f'Training {type(initialized_model.model).__name__} started.' {'<<'*30}"
            grid_search_cv.fit(input_feature, output_feature)
            message = f"{'>>'*30} f'Training {type(initialized_model.model).__name__} completed.' {'<<'*30}"
            logging.info(message)
            
            grid_searched_best_model = GridSearchedBestModel(
                model_serial_number=initialized_model.model_serial_number,
                model=initialized_model.model,
                best_model=grid_search_cv.best_estimator_,
                best_parameters=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_
            )
            print(f"-------modelfactory--execute grid search->>>>>>>{grid_searched_best_model}<<<<<<<<<-------grid_searched_best_model---------")
            
            return grid_searched_best_model
            
        except Exception as e:
            raise ProjectException(e, sys) from e 
            
            
    def get_initialized_model_list(self) ->List[InitializedModelDetail]:
        try:
            initialized_model_list= []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config =self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY])
                model = model_obj_ref()
                print(f"-------modelfactory--get initiallized model->>>>>>>{model}<<<<<<<<<------model---------")
                
                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data= dict(model_initialization_config[PARAM_KEY])
                    model= ModelFactory.update_propery_of_class(instance_ref=model,
                                                                propery_data=model_obj_property_data)
                
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                
                model_initialization_config=InitializedModelDetail(model_serial_number=model_serial_number,
                                                                   model=model,
                                                                   param_grid_search=param_grid_search,
                                                                   model_name=model_name)
                
                initialized_model_list.append(model_initialization_config)
                
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
                
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    def initiate_best_parameter_search_for_initialized_model(self, initialized_model:InitializedModelDetail,
                                                            input_feature,output_feature):
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)   
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
        
    def initiate_best_parameter_search_for_initialized_models(self, initialized_model_list: List[InitializedModelDetail],
                                                              input_feature, output_feature):
        try:
            self.grid_searched_best_model_list = []
            for initialized_model_ls in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_ls,
                    input_feature=input_feature,
                    output_feature=output_feature,
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
        
    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number):
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data 
        except Exception as e:
            raise ProjectException(e, sys) from e 
                       
    
    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.6):
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    base_accuracy= grid_searched_best_model.best_score
                    
                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of the Model has base accuracy: {base_accuracy}")
            print(f"----model_factory---->>>>>>>>{best_model}<<<<<<<<<<<<-------get gridsearched--best model-------")
            return best_model
        
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def get_best_model(self, X,y, base_accuracy=0.6):
        try:
            initialized_model_list = self.get_initialized_model_list()
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list,
                base_accuracy=base_accuracy
            )
        
        except Exception as e:
            raise ProjectException(e, sys) from e 
                       
                       