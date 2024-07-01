# Algerian Forest Fire Prediction
This repository contains a machine learning project focused on predicting forest fire occurrences in Algeria using the "Algerian Forest Fires Dataset" from [Kaggle](https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset). The project aims to develop a robust model that can effectively identify potential fire risks, thereby supporting proactive measures for prevention, mitigation, and resource allocation.


## 🚩 Table of Contents

- [Project Overview](#-project-overview)
    - [Problem Statement](#-problem-statement)
    - [Project Goals](#-project-goals)
- [Getting Started](#-getting-started)
    - [Prerequisites](#-prerequisites)
    - [Installation](#-installation)
    - [Running the Project](#-running-the-project)
- [Dataset](#-dataset)
- [Methodology](#-dataset)
- [Tools](#-dataset)
- [Project Structure](#-project-structure)
- [Contributing](#-contributions)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## Project Overview

Forest fires pose a significant threat to the environment and human safety. This project aims to develop a  a machine learning model that can accurately predict whether a forest fire will occur based on input features based on environmental and weather data. This is a binary classification problem, where the model needs to learn the patterns that distinguish between instances where a fire occurred ("fire") and instances where no fire occurred ("not fire").

### Project Goals

1. **Data Acquisition and Preprocessing:** 
    - Download and prepare the "Algerian Forest Fires Dataset" for analysis.
    - Cleanse the data to handle missing values, inconsistencies, and outliers.
2. **Model Development:** 
    - Train a machine learning model capable of predicting whether a forest fire will occur based on environmental and weather factors.
    - Explore and compare different machine learning algorithms to identify the most suitable model.
    - Tune hyperparameters to optimize the model's performance.
3. **Model Evaluation:**
    - Evaluate the trained model using relevant metrics (e.g., accuracy, precision, recall, F1-score, ROC AUC).
    - Analyze the model's predictions and identify any potential areas for improvement.
4. **Pipeline Creation:**
    - Develop a streamlined pipeline to automate the entire process, from data ingestion to model training and evaluation, ensuring reproducibility.
5. **Deployment (Future Consideration):**
    -  Explore potential deployment options to make the model readily available for operational use, such as a web application, API, or integration with existing fire management systems.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.9 (or compatible)
- Conda (optional, but recommended)

### Installation

1. **Clone the repository:**
    ```bash
    $ git clone https://github.com/your-username/algerian-forest-fire-prediction.git
    ```
2. **Create a conda environment (optional):**
    ```bash
    $ conda create -n forest-fire-env python=3.9 
    ```
    This creates a Conda environment named `forest-fire-env` (you can choose your own name) with Python 3.9. Adjust the Python version as needed.

3. **Activate the environment:**
    ```bash
    $ conda activate forest-fire-env
    ```
4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Setup your command line interface for better readability (Optional):**
   ```bash
   export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
   ```
### Running the Project

#### 1) **Setup Kaggle API:**
    * **Download `kaggle.json`:**  Download your Kaggle API credentials (username and API key) from your Kaggle account. put the file in the project direactory.
    ```json
    {
      "username": "your_username",
      "key": "your_api_key"
    }
    ```

    * **Configure Kaggle:** Run these commands in your terminal to copy the `kaggle.json` file in a spacific directory.
        ```bash
        mkdir -p ~/.kaggle
        cp kaggle.json ~/.kaggle/kaggle.json
        ```
    * **Set Permissions:** Make sure the file is only accessible to you:
        ```bash
        chmod 600 ~/.kaggle/kaggle.json
        ```

#### 2) **Configure DVC with Remote Storage:**
    * **Create a Google Drive folder:** Go to your Google Drive and create a new folder in your Google Drive to store your project's data and model artifacts (e.g., "Algerian Forest Fire Project").
    * **Obtain Drive Key:**  Go to the newly created Google Drive folder and get the folder's unique key from the URL (the part after `id=` in the URL).
    * **Set up DVC remote:**
        ```bash
        dvc remote add -d gdrive dvc://?token=<your_drive_key>
        dvc remote default gdrive
        ```
        Replace `<your_drive_key>` with the Google Drive folder key you obtained in the previous step. 

#### 3) **Run the project:**
   ```bash
   $ python template.py
   $ dvc repro
   ```
   These commands will execute the project's pipeline, including:
   - **Data Ingestion:** Download the dataset directly from Kaggle using the Kaggle API you just configured.
   - **Preprocessing:** Clean, transform, and prepare the data for model training.
   - **Model Training:** Train a machine learning model based on the chosen algorithm and hyperparameters.
   - **Model Evaluation:** Evaluate the trained model's performance using various metrics.
   - **Artifact Saving:** Save the trained model, evaluation results, and other important artifacts for future use or analysis.

This will guide you through the entire workflow from setting up your Kaggle API to running the project and generating valuable results.

## Dataset

## Project Structure

## Methodology 


### Contributions

Contributions to this project are welcome! Feel free to:

* **Report issues:** If you encounter any bugs or have suggestions for improvement, please open an issue on the GitHub repository.
* **Submit pull requests:** If you'd like to contribute code, fork the repository, make your changes, and submit a pull request.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

<!-- - [Dataset Source] - For providing the Algerian forest fire dataset.
- [Library Name] - For providing the machine learning library used. -->
``` 
