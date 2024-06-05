import sys
import os
from colorama import Fore, Style
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from src.logger import logging



def get_error_msg(error_custom_msg:str, error:Exception):
    """
    Returns a detailed error message string, including file name, line number, and custom message.
    """
    _, _, exc_tb = sys.exc_info()
    # file_path = exc_tb.tb_frame.f_code.co_filename
    file_name = os.path.basename(exc_tb.tb_frame.f_code.co_filename)  # Get only the file name
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in '{file_name}' on line {line_number}: {error_custom_msg}"
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class for more informative error handling with logging.
    """
    def __init__(self, error_custom_msg: str, error_detail: Exception = None):

        super().__init__()
        self.error_message = get_error_msg(error_custom_msg, error_detail)
        logging.error(self.error_message)

    def __str__(self):
        return Fore.MAGENTA + self.error_message + Style.RESET_ALL

# EXample for testing the code
if __name__ == "__main__":

    try:
        result = 10 / 0 # This will cause a ZeroDivisionError
    except Exception as e:
        raise CustomException("Error in division operation", e)
