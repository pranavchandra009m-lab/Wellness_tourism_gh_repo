from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
hf_space_repo_id = "Jags99/SK_FrontEnd_HF"

api.upload_folder(
    folder_path = "wellness-tourism-mlops/prediction_app",
    repo_id = hf_space_repo_id,
    repo_type='space',
    path_in_repo=''
)
