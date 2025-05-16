import logging
import os
from datetime import datetime

# Crear nombre del archivo de log
LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.log"

# Ruta al directorio de logs
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Ruta completa al archivo de log
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configuración del logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Formato del log
formatter = logging.Formatter('[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s')

# Handler para archivo
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
