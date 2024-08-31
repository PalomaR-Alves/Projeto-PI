"""
Here all the configuration is stored
"""
from os import path, makedirs

# Caminhos padrão para as pastas de dados e saídas
DEFAULT_DATA_FOLDER = "./data"
DEFAULT_OUTPUT_FOLDER = "./outputs"

# Verifica se a pasta de dados existe, se não existir, cria
if not path.exists(DEFAULT_DATA_FOLDER):
    makedirs(DEFAULT_DATA_FOLDER)

# Verifica se a pasta de saídas existe, se não existir, cria
if not path.exists(DEFAULT_OUTPUT_FOLDER):
    makedirs(DEFAULT_OUTPUT_FOLDER)

# Define as variáveis de configuração
DATA = DEFAULT_DATA_FOLDER
OUTPUTS = DEFAULT_OUTPUT_FOLDER
