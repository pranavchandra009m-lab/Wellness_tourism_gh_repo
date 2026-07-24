
# Import required libraries for Hugging Face dataset management
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Authenticate with Hugging Face using the stored token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define Hugging Face dataset repository details
hf_dataset_repo_id = "Jags99/wellness-tourism-dataset"

# Define local folder containing the source dataset
dataset_folder_path = "wellness-tourism-mlops/data"

# Check if the dataset repository exists; create it if unavailable
try:
    api.repo_info(
        repo_id=hf_dataset_repo_id,
        repo_type="dataset"
    )
    print(f"Dataset repository '{hf_dataset_repo_id}' already exists in Hugging Face.")

except RepositoryNotFoundError:
    api.create_repo(
        repo_id=hf_dataset_repo_id,
        repo_type="dataset",
        private=False
    )
    print(f"Dataset repository '{hf_dataset_repo_id}' created successfully.")

# Upload source dataset files to Hugging Face dataset repository
api.upload_folder(
    folder_path=dataset_folder_path,
    repo_id=hf_dataset_repo_id,
    repo_type="dataset"
)

print(f"\nSource dataset uploaded successfully to Hugging Face: '{hf_dataset_repo_id}'")
