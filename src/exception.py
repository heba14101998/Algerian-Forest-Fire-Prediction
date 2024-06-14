import sys
import os
from colorama import Fore, Style
from src.logger import logging 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def get_error_msg(error_custom_msg: str, exp: Exception) -> str:
    """
    Returns a detailed error message string, including file name, line number, and custom message.
    Args:
        error_custom_msg (str): The custom error message to be included.
        exp (Exception): The exception object that occurred.
    Returns:
        str: A formatted error message string.
    """
    _, _, exc_tb = sys.exc_info()
    file_name = os.path.basename(exc_tb.tb_frame.f_code.co_filename)  # Get only the file name
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in '{file_name}' on line {line_number}: {error_custom_msg}"
    return error_message


class CustomException(Exception):
    """
    Custom exception class for more informative error handling with logging.
    """
    def __init__(self, error_custom_msg: str, exp: Exception = None):
        """
        Initializes the CustomException with a custom message and optional exception detail.
        Args:
            error_custom_msg (str): The custom error message.
            exp (Exception, optional): The exception object that occurred. Defaults to None.
        """
        super().__init__()
        self.error_message = get_error_msg(error_custom_msg, exp)
        logging.error(self.error_message)

    def __str__(self) -> str:
        """
        Returns a string representation of the CustomException with colored output.

        Returns:
            str: The formatted error message with color.
        """
        return Fore.MAGENTA + self.error_message + Style.RESET_ALL

# Example for testing the code
# if __name__ == "__main__":
#     try:
#         result = 10 / 0  # This will cause a ZeroDivisionError
#     except Exception as e:
#         raise CustomException("Error in division operation", e)