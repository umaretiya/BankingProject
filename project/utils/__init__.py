import os,sys
import yaml
import numpy as np
import dill 
import pandas as pd
from project.exception import ProjectException
from project.constant import * 

def read_yaml(file_path):
    try:
        with open(file_path,"rb") as yaml_file:
            file_read = yaml.safe_load(yaml_file)
            return file_read
    except Exception as e:
        raise ProjectException(e, sys) from e
    
def write_yaml(file_path, data:dict):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as e:
        raise ProjectException(e, sys) from e
    
    
def load_numpy_array(file_path):
    try:
        with open(file_path,'rb') as numpy_array:
            return np.load(numpy_array)
    except Exception as e:
        raise ProjectException(e, sys) from e
    
def save_numpy_array(file_path, array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as np_obj:
            return np.save(np_obj, array)
    except Exception as e:
        raise ProjectException(e, sys) from e
    
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            # print(f"-------load_object----{file_path}------<<<<<<<<<<<---------")
            # print(f"-------load_object----{file_obj}------<<<<<<<<<<<---------")
            dill.load(file_obj)
    except Exception as e:
        raise ProjectException(e, sys) from e
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
                
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise ProjectException(e, sys) from e
    
    
def load_data(file_path, schema_file_path):
    try:
        schema_file = read_yaml(schema_file_path)
        dataframe = pd.read_csv(file_path)
        schema = schema_file[DATASET_SCHEMA_COLUMNS_KEY]
        
        for col in dataframe.columns:
            if col in list(schema.keys()):
                dataframe[col].astype(schema[col])
            else:
                raise Exception(f"{col} not in dataframe")
        return dataframe
        
    except Exception as e:
        raise ProjectException(e, sys) from e