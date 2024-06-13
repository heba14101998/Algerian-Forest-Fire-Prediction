import os

list_of_files = [
    "Notebooks/EDA_and_Feature_Engineering.ipynb",
    "assets/",
    "data/raw/",
    "data/processed/",
    "checkpoints/",
    "artifacts/",

    "src/__init__.py",
    "src/utils.py",
    "src/logger.py",
    "src/exception.py",

    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_factory.py",
    "src/components/model_training.py",

    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/inference_pipeline.py",

    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/unit/unit_test.py",
    "tests/integration/__init__.py",
    "tests/integration/integration_test.py",
    
    ".github/workflows/ci.yaml",
    ".github/workflows/publish.yaml",

    "assets/kaggle.json",

    "requirements.txt",
    "params.yaml",
    "dvc.yaml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "pyproject.toml",
    ".gitignore",
    ".env",
    ".env.example",
    "README.md",
]

# Use a set to keep track of created directories
created_directories = set()

for path in list_of_files:
    # Check if the path ends with a slash, indicating it's a directory
    if path.endswith('/'):
        # Create the directory if it doesn't exist and hasn't been created already
        if path not in created_directories:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Directory created: {path}")
            else:
                print(f"Directory already exists: {path}")
            created_directories.add(path)
    else:
        # Ensure the directory for the file exists
        directory = os.path.dirname(path)
        if directory and directory not in created_directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory created for file: {directory}")
            else:
                print(f"Directory already exists for file: {directory}")
            created_directories.add(directory)
        
        # Create the file if it doesn't exist
        if not os.path.exists(path):
            with open(path, 'w') as file:
                pass  # Just create an empty file
            print(f"File created: {path}")
        else:
            print(f"File already exists: {path}")