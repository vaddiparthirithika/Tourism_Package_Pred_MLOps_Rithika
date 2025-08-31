import os
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import RepositoryNotFoundError
from dotenv import load_dotenv

# Hugging Face repo details
repo_id = "Vaddiritz/Tourism-Package-Prediction-rithika"
repo_type = "space"

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
api = HfApi()

# Check if repo exists, else create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f" Creating new Space '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="streamlit")
    print(f" Repo '{repo_id}' created.")

# Upload deployment files
base_path = os.getcwd()
files_to_upload = [
    os.path.join(base_path, "Dockerfile"),
    os.path.join(base_path, "app.py"),
    os.path.join(base_path, "requirements.txt")
]

for file in files_to_upload:
    upload_file(
        path_or_fileobj=file,
        path_in_repo=os.path.basename(file),
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f" Uploaded {file} to {repo_id}")
