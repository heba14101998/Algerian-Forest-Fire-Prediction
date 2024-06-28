#!/bin/bash

# Function to clear local experiments
clear_local_experiments() {
  echo "Clearing local experiments..."
  dvc exp clean
  dvc exp remove --all-commits 
}

# Function to clear remote experiments
clear_remote_experiments() {
  local remote_name=$1
  echo "Clearing remote experiments from $remote_name..."
  
  # List all remote experiments
  exp_list=$(dvc exp list $remote_name)
  
  # Check if there are any experiments to remove
  if [[ -z "$exp_list" ]]; then
    echo "No experiments found in remote $remote_name."
  else
    # Remove all remote experiments
    dvc exp remove --remote $remote_name
    echo "All experiments removed from remote $remote_name."
  fi
}

# Clear local experiments
clear_local_experiments

# Clear remote experiments (replace 'myremote' with your actual remote name)
clear_remote_experiments "https://github.com/heba14101998/Algerian-Forest-Fire-Prediction.git"

echo "Cleanup complete."
