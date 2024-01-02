import os
import sys
import logging

# Constants should be in uppercase according to PEP 8
LOGGING_STR = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
LOG_DIR = "logs"
LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_STR,
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger instance
logger = logging.getLogger("cnnClassifierLogger")
