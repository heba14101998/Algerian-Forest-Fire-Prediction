import os
from pathlib import Path

PAKAGE_NAME = ""

list_of_files = [
                ".github/workflows/ci.yaml",
                "Notebooks/Data_Expolration.ipynb", 
                "assets/"
                "data/raw/",
                "data/processed/",
                "checkpoints/",
                "artificts/",

                "src/__init__.py",
                "src/utils.py",
                "src/logger.py",
                "src/exception.py",
                "src/components/__init__.py", 
                "src/pipline/__init__.py",

                "tests/__init__.py",
                "tests/unit/__init__.py",
                "tests/unit/unit_test.py",
                "tests/integration/__init__.py",
                "tests/integration/init_test.py",
                
                "requirements.txt", 
                "setup.py",
                ".env",
                ".env.example",
                "README.md",
                ]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file
