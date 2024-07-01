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

#### Step 1: Setup Kaggle API

* **Download `kaggle.json`:** Download your Kaggle API credentials (username and API key) from your Kaggle account. Put the file in the project directory.
    ```json
    {
      "username": "your_username",
      "key": "your_api_key"
    }
    ```

* **Configure Kaggle:** Run these commands in your terminal to copy the `kaggle.json` file in a specific directory. 
    ```bash
    mkdir -p secrets
    ```
    Then put your Kaggle token in the `secrets` directory.
    ```bash
    mkdir -p ~/.kaggle
    cp secrets/kaggle.json ~/.kaggle/kaggle.json
    ```
* **Set Permissions:** Make sure the file is only accessible to you:
    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```

#### Step 2: Configure DVC with Remote Storage

##### Creating a Service Account

1. **Create a Google Cloud Platform Project:**
   - If you don't have one, create a Google Cloud Platform project (follow instructions [here](https://cloud.google.com/iam/docs/service-account-overview)), you can follow instraction from [DVC documentation](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-service-accounts).

2. **Create a Service Account:**
   - Go to the IAM & Admin section in your Google Cloud Platform project.
   - Click "Service Accounts" and then "Create Service Account."
   - Give your service account a name and description.
   - In the "Roles" section, select "Storage Object Viewer" or "Storage Object Admin" (depending on your needs).
   - Click "Create."

3. **Generate Key:**
   - On the service account's detail page, click "Keys" and then "Add Key."
   - Choose "JSON" as the key type. 
   - Click "Create." This will download a JSON file containing your service account's credentials. 

4. **Store Service Account Key:**
   - Place this JSON file (e.g., `secrets/service-account-token.json`) in the root of your project directory.

##### **Add the Google Drive Remote:**
* **Create a Google Drive folder:** Go to your Google Drive and create a new folder in your Google Drive to store your project's data and model artifacts (e.g., "Algerian Forest Fire Project").
* **Obtain Drive Key:**  Go to the newly created Google Drive folder and get the folder's unique key from the URL (the part after `id=` in the URL).
   ```bash
   dvc remote add -d gdrive gdrive://?token=<your_drive_key>
   ```
   Replace `<your_drive_key>` with the Google Drive folder key you obtained in the previous step (after `id=` in the URL of your Drive folder).

2. **Set the Default Remote:**
   ```bash
   dvc remote default gdrive
   ```

3. **Configure Service Account for DVC:**
   ```bash
   dvc remote modify gdrive gdrive_use_service_account true
   dvc remote modify gdrive --local gdrive_service_account_json_file_path secrets/service-account-token.json
   ```

#### Step 3: Run the Project

1. **Run the project's setup script:**
   ```bash
   $ python template.py
   ```

2. **Run the DVC pipeline:**
   ```bash
   $ dvc repro
   $ dvc push
   ```

   This will execute the project's pipeline:
   - **Data Ingestion:** Download the dataset directly from Kaggle using the Kaggle API you just configured.
   - **Preprocessing:** Clean, transform, and prepare the data for model training.
   - **Model Training:** Train a machine learning model based on the chosen algorithm and hyperparameters.
   - **Model Evaluation:** Evaluate the trained model's performance using various metrics.
   - **Artifact Saving:** Save the trained model, evaluation results, and other important artifacts for future use or analysis.

   The artifacts will be uploaded to your Google Drive using the service account.

## Dataset
for more explination read docs/dataset-description.md
## Methodology 
for more explination read docs/Methodology.md

## Contributions

Contributions to this project are welcome! Feel free to:

* **Report issues:** If you encounter any bugs or have suggestions for improvement, please open an issue on the GitHub repository.
* **Submit pull requests:** If you'd like to contribute code, fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

<!-- - [Dataset Source] - For providing the Algerian forest fire dataset.
- [Library Name] - For providing the machine learning library used. -->
``` 
