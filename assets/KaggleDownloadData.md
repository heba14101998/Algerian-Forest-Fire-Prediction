

**1. Generate Your Kaggle API Token**

* **Go to Kaggle:** Log in to your Kaggle account at [https://www.kaggle.com/](https://www.kaggle.com/).
* **Navigate to Account Settings:** Click on your profile picture, then select "My Account".
* **Create a New API Token:**  Click on "Create New API Token".
* **Download the JSON file:** This file will be downloaded as a `kaggle.json` file. I saved it in directory called `assests` in my project directory.

**2. Set Up Your Kaggle API Configuration**

**Make the file secure:**
   ```bash
   chmod 600 assets/kaggle.json 
   ```

**3. Download the Dataset**

* **Find the dataset's API URL:** Go to the dataset [https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset](page) Look for the "API" section. You'll see the dataset's unique URL (it will look something like `nitinchoudhary012/algerian-forest-fires-dataset`).
* **Use the `kaggle` command:**  In your terminal, run this command:
   ```bash

   kaggle datasets download -d nitinchoudhary012/algerian-forest-fires-dataset -p data/raw -f algerian_forest_fires_dataset.csv 
   ```