import os
import logging
from datetime import datetime


# Define the log file name and path
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), 'logs', LOG_FILE)

# Create the logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Configure basic logging setup
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] - %(levelname)s - %(lineno)d - %(message)s',
    level=logging.INFO,
)
