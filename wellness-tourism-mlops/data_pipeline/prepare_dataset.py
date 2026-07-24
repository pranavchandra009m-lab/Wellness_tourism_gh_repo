
# Import required libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Load source dataset from Hugging Face Dataset Repository
source_dataset_path = "hf://datasets/Jags99/wellness-tourism-dataset/tourism.csv"

wellness_tourism_df = pd.read_csv(source_dataset_path)

print("Source dataset loaded successfully from Hugging Face.\n")

# Remove columns that are not useful for model training
columns_to_remove = ['Unnamed: 0', 'CustomerID']

# Define target variable
target_column = "ProdTaken"

# Separate input features and target variable
X = wellness_tourism_df.drop(columns=columns_to_remove + [target_column])
y = wellness_tourism_df[target_column]

print("Feature and target datasets prepared successfully.")
print(f"Number of features used for training: {X.shape[1]}\n")

# Split dataset into training and testing sets
# Stratify maintains the target class distribution in both datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset split into training and testing sets successfully.")
print(f"Training dataset size: {X_train.shape[0]} rows")
print(f"Testing dataset size: {X_test.shape[0]} rows\n")

# Create directory to store processed train-test datasets
processed_data_directory = "wellness-tourism-mlops/processed_data"

os.makedirs(processed_data_directory, exist_ok=True)

print(f"Processed data directory created: {processed_data_directory}\n")

# Save processed datasets locally
processed_files = {
    "X_train.csv": X_train,
    "X_test.csv": X_test,
    "y_train.csv": y_train,
    "y_test.csv": y_test
}

for file_name, dataset in processed_files.items():
    dataset.to_csv(
        os.path.join(processed_data_directory, file_name),
        index=False
    )

print("Processed training and testing files saved successfully.")
print(f"Generated files: {', '.join(processed_files.keys())}\n")

# Upload processed datasets to Hugging Face Dataset Repository
api = HfApi(token=os.getenv("HF_TOKEN"))

hf_dataset_repo_id = "Jags99/wellness-tourism-dataset"

for file_name in processed_files.keys():
    api.upload_file(
        path_or_fileobj=os.path.join(processed_data_directory, file_name),
        path_in_repo=f"train_test_data/{file_name}",
        repo_id=hf_dataset_repo_id,
        repo_type="dataset"
    )

print("Training and testing datasets uploaded successfully to Hugging Face.")
print(f"Repository location: {hf_dataset_repo_id}/train_test_data")
