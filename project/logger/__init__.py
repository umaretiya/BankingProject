import logging, os
from project.constant import get_current_time_stamp
import pandas as pd

def get_logfile_name():
    return f"log_{get_current_time_stamp()}.logs"

LOG_FILE_NAME=get_logfile_name()
LOG_DIR="logs_project"
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)


# logging.basicConfig(filename=LOG_FILE_PATH,
#                     filemode="w",
#                     format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcname)s()^;%(message)s^;' ,
#                     level=logging.INFO)

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode="w",
                    format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s',
                    level=logging.INFO
                    )



def get_log_dataframe(file_path):
    data = []
    with open(file_path) as log_file:
        for line in log_file.readline():
            data.append(line.split("^;"))
            
    log_df = pd.DataFrame(data)
    columns =["Time stamp","Log Level","line number","file name","function name","message"]
    log_df.columns = columns
    
    log_df["log_message"] = log_df["Time stamp"].astype(str) + ":$" + log_df["message"]
    
    return log_df[["log_message"]]