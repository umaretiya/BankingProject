from tkinter.messagebox import NO
from project.constant import ROOT_DIR
from project.entity.config_entity import DataIngestionConfig 
from project.entity.artifact_entity import DataIngestionArtifact 
from project.logger import logging 
from project.exception import ProjectException 
import os, sys 
import numpy as np  
import pandas as pd 
import shutil,zipfile
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try: 
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def download_project_data(self):
        try:
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            os.makedirs(tgz_download_dir,exist_ok=True)
            
            tgz_file_name = self.data_ingestion_config.tgz_file_name
            
            source_data_dir = ROOT_DIR
            source_file = os.path.join(source_data_dir, tgz_file_name)
            
            shutil.copy(source_file, tgz_download_dir)
            
            tgz_file_path = os.path.join(tgz_download_dir, tgz_file_name)
            return tgz_file_path
        
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def exctracted_tgz_data(self, tgz_file_path):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            os.makedirs(raw_data_dir, exist_ok=True)
            
            with zipfile.ZipFile(tgz_file_path, 'r') as zipref:
                zipref.extractall(raw_data_dir)
                
            file_name = os.listdir(raw_data_dir)[0]
            banking_file_path = os.path.join(raw_data_dir,file_name)
            banking_data_frame = pd.read_csv(banking_file_path)
            banking_data_frame.rename(mapper={'default.payment.next.month':"default"},axis=1,inplace=True)
            banking_data_frame.to_csv(banking_file_path,index=False)
            
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def split_data_as_train_test(self):
        try:
            raw_data_dir=self.data_ingestion_config.raw_data_dir
            
            file_name=os.listdir(raw_data_dir)[0]
            
            project_file_path= os.path.join(raw_data_dir, file_name)
            
            project_data_frame=pd.read_csv(project_file_path)
            
            convert_dict = {i:float for i in project_data_frame.columns.to_list()}
            project_data_frame = project_data_frame.astype(convert_dict)
            
            strat_train_set=None
            strat_test_set=None 
            
            strat_train_set,strat_test_set= train_test_split(project_data_frame, test_size=0.3, random_state=42)
            
            train_file_path=os.path.join(self.data_ingestion_config.ingested_train_dir,file_name)
            test_file_path=os.path.join(self.data_ingestion_config.ingested_test_dir,file_name)
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                strat_train_set.to_csv(train_file_path, index=False)
                
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                strat_test_set.to_csv(test_file_path, index=False)
            
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_ingested=True,
                message=f"Data Ingestion completed successfully."
            )
            return data_ingestion_artifact
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
        
    def initiate_data_ingestion(self):
        try:
            tgz_file_path= self.download_project_data()
            self.exctracted_tgz_data(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise ProjectException(e, sys) from e 
        
    
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")

