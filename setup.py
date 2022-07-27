from importlib import import_module
from pytz import VERSION
from setuptools import setup, find_packages
from typing import List

PROJECT_NAME = "Credit Dard Default-predictor"
VERSION="0.0.1"
AUTHOR="Keshav Umaretiya"
DESCRIPTION="This is a Machine Learning End to End Project"
REQUIREMENT_FILE_NAME= "requirements.txt"
HYPHEN_E_DOT="-e ."

def get_requirements_list():
    with open(REQUIREMENT_FILE_NAME) as requirementt_file:
        requirementt_list = requirementt_file.readlines()
        requirementt_list = [requirement_name.replace("\n", "") for requirement_name in requirementt_list]
        if HYPHEN_E_DOT in requirementt_list:
            requirementt_list.remove(HYPHEN_E_DOT)
        return requirementt_list

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements_list()
)