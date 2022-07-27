from flask import Flask, render_template,request,abort, send_file
from project.utils import read_yaml, write_yaml, load_numpy_array
from project.logger import logging, get_log_dataframe
from project.configured import Configuration
from project.constant import *
from project.pipeline import Pipeline

from project.model.estimator_model import ProjectEstimatorModel
from project.model.model_factory import evaluate_regression_model, ModelFactory
from project.component.ingestion import DataIngestion
from project.component.validation import DataValidation
from project.component.transformation import DataTransformation
from project.model.model_predictor import  ProjectData

import os, json, pandas as pd

app = Flask(__name__)


LOG_FOLDER_NAME = "logs_project"
PIPELINE_FOLDER_NAME="project"
SAVED_MODELS_DIR_NAME="saved_models"

MODEL_CONFIG_FILE_PATH= os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR=os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR= os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR= os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


PROJECT_DATA_KEY="project_data"
DEFAULT = "default"



@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        raise str(e)

@app.route('/artifact', defaults={'req_path': 'project'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    print(f"req_path: {req_path}")
    abs_path=os.path.join(req_path)
    if not os.path.exists(abs_path):
        return abort(404)
    
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, 'r', encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if "artifact" in os.path.join(abs_path, file_name)}
    
    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
        }
    return render_template("files.html", result=result)

@app.route('/view_experiment_list', methods=['GET','POST'])
def view_experiment_history():
    config=Configuration(current_time_stamp=get_current_time_stamp())
    experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
    df = pd.read_csv(experiment_file_path)
    df = df[-10:].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
    
    context = {
        "experiment": df.to_html(classes='table table-striped col-12')
    }
    return render_template("experiment_history.html", context=context)

@app.route('/train',methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=CURRENT_TIME_STAMP))
    if not Pipeline.experiment.running_status:
        message = "Training started"
        pipeline.start()
    else:
        message = "Training is already in progress"
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message,
        }
    return render_template("train.html", context=context)


@app.route('/predict', methods=['GET','POST'])
def predict():
    context = {
        PROJECT_DATA_KEY: None,
        DEFAULT: None
    }
    conf = Configuration(config_file_path=CONFIG_FILE_PATH, current_time_stamp=CURRENT_TIME_STAMP)
    ingest = conf.get_data_ingestion_config()
    validate = conf.get_data_validation_config()
    transform = conf.get_data_transformation_config()
    model_train= conf.get_model_trainer_config() 

    
    data_ingested = DataIngestion(data_ingestion_config=ingest)
    data_ingested_start = data_ingested.initiate_data_ingestion()
 
    data_validate = DataValidation(data_ingestion_artifact=data_ingested_start, data_validation_config=validate)
    data_validate_start = data_validate.initiate_data_validation()
    
    data_transformation = DataTransformation(data_transformation_config=transform, data_validation_artifact=data_validate_start,data_ingestion_artifact=data_ingested_start)
    data_transformation_start = data_transformation.initiate_data_transformation()
    trained_file = pd.read_csv(data_ingested_start.train_file_path)
    X = trained_file.drop(labels=['default'], axis=1)
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

    if request.method == 'POST':
        ID = float(request.form['ID'])
        LIMIT_BAL = float(request.form['LIMIT_BAL'])
        SEX = float(request.form['SEX'])
        EDUCATION = float(request.form['EDUCATION'])
        MARRIAGE = float(request.form['MARRIAGE'])
        AGE = float(request.form['AGE'])
        PAY_0= float(request.form['PAY_0'])
        PAY_2= float(request.form['PAY_2'])
        PAY_3= float(request.form['PAY_3'])
        PAY_4= float(request.form['PAY_4'])
        PAY_5= float(request.form['PAY_5'])
        PAY_6= float(request.form['PAY_6'])
        BILL_AMT1= float(request.form['BILL_AMT1'])
        BILL_AMT2= float(request.form['BILL_AMT2'])
        BILL_AMT3= float(request.form['BILL_AMT3'])
        BILL_AMT4= float(request.form['BILL_AMT4'])
        BILL_AMT5= float(request.form['BILL_AMT5'])
        BILL_AMT6= float(request.form['BILL_AMT6'])
        PAY_AMT1= float(request.form['PAY_AMT1'])
        PAY_AMT2= float(request.form['PAY_AMT2'])
        PAY_AMT3= float(request.form['PAY_AMT3'])
        PAY_AMT4= float(request.form['PAY_AMT4'])
        PAY_AMT5= float(request.form['PAY_AMT5'])
        PAY_AMT6= float(request.form['PAY_AMT6'])
                
        project_data = ProjectData(
                ID =ID,
                LIMIT_BAL=LIMIT_BAL,
                SEX=SEX,
                EDUCATION=EDUCATION,
                MARRIAGE=MARRIAGE,
                AGE=AGE,
                PAY_0=PAY_0,
                PAY_2=PAY_2,
                PAY_3=PAY_3,
                PAY_4=PAY_4,
                PAY_5=PAY_5,
                PAY_6=PAY_6,
                BILL_AMT1=BILL_AMT1,
                BILL_AMT2=BILL_AMT2,
                BILL_AMT3=BILL_AMT3,
                BILL_AMT4=BILL_AMT4,
                BILL_AMT5=BILL_AMT5,
                BILL_AMT6=BILL_AMT6,
                PAY_AMT1=PAY_AMT1,
                PAY_AMT2=PAY_AMT2,
                PAY_AMT3=PAY_AMT3,
                PAY_AMT4=PAY_AMT4,
                PAY_AMT5=PAY_AMT5,
                PAY_AMT6=PAY_AMT6,
                )
        project_df = project_data.get_project_input_data_frame()
        
        # os.listdir()
        # [ID,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default]

        print(f"-----project_df------>>>>{project_df}<<<<<<<<--------project_df ----")
        
        project_df_sc = prepro_obj.transform(project_df)
        default = project_model.predict(project_df_sc)
        context = {
            PROJECT_DATA_KEY: project_data.get_project_data_as_dict(),
            DEFAULT:default,
            }
        return render_template("predict_credit.html", context=context)
    return render_template("predict_credit.html", context=context)


@app.route('/saved_models', defaults={'req_path':'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    
    if not os.path.exists(req_path):
        return abort(404)
    
    if os.path.isfile(abs_path):
        return send_file(abs_path)
    
    files = {os.path.join(abs_path, file):file for file in os.listdir(abs_path)}
    
    result = {
        "files": files,
        "parent_folder":os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)
    
    
@app.route("/update_model_config", methods=['GET','POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config= request.form['new_model_config']
            model_config = model_config.replace("'",'"')
            print(model_config)
            model_config = json.loads(model_config)
            
            write_yaml(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)
        model_config = read_yaml(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config":model_config})
    
    except Exception as e:
        logging.exception(e)
        return str(e)

@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    abs_path = os.path.join(req_path)
    
    if not os.path.exists(abs_path):
        return abort(404)
    
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log":log_df.to_html(classes="table table-striped", index=False)}
        return render_template("log.html", context=context)  
    
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
        }
    return render_template('log_files.html', result=result)  

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", debug=True,port=port)