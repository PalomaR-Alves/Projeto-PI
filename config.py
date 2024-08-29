"""
Here all the configuration is stored
"""
from os import path, listdir

DEFAUL_DATA_FOLDER = "./data"
DEFAULT_OUTPUT_FOLDER="./outputs"

DATA= DEFAUL_DATA_FOLDER if path.exists(DEFAUL_DATA_FOLDER) else ""
OUTPUTS= DEFAULT_OUTPUT_FOLDER if path.exists(DEFAUL_DATA_FOLDER) else ""