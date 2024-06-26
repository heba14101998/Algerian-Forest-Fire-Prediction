from setuptools import setup, find_packages

with open("README.md", 'r', encoding='utf-8') as file:
    long_description = file.read()

# Read requirements from requirements.txt, removing comments and newlines
with open('requirements.txt') as file:
    """
    from each line in the `requirements.txt` file.
        line.strip(): removes leading and trailing whitespace (spaces, tabs, or newlines)
        not line.startswith('#'): If the line doesn't start with the '#' character, which indicates a comment.
    """
    requirements = [line.strip() for line in file if not line.startswith('#') and line.strip()]

setup(
    name="forest-fire",
    version="0.1.5",
    description="Algerian Forest Fire Prediction Model",
    long_description=long_description, # Use read README.md
    long_description_content="text/markdown",
    author="hebamohamed1998",
    author_email="hebamohamed14101998@gmail.com" ,
    url="https://github.com/heba14101998/Algerian-Forest-Fire-Prediction.git",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,  # Use read requirements
    classifiers=[
        "Development Status :: 3 - Alpha", 
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        # "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        'console_scripts': [
            'train=src.pipeline.train_pipeline:main',
        ]
    },
)