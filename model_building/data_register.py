
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import os

# Repo Information
repo_id = "Vaddiritz/Tourism-Package-Prediction-rithika_new"
repo_type = "dataset"

# Hugging Face Token
#os.environ["HF_TOKEN"] = userdata.get("HF_token")
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
api = HfApi()

# Check if dataset repo exists, otherwise create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Path to your CSV
base_path = os.getcwd()
print(base_path)
csv_path = os.path.join(base_path,"data/tourism.csv")
print(csv_path)

# Upload the CSV file
api.upload_file(
    path_or_fileobj=csv_path,
    path_in_repo="tourism.csv",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Upload tourism dataset"
)

print("tourism.csv uploaded successfully to Hugging Face Hub!")
