from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
from google.colab import userdata

# Repo Information
repo_id = "Vaddiritz/Tourism-Package-Prediction-rithika"
repo_type = "dataset"

# Hugging Face Token
os.environ["HF_TOKEN"] = userdata.get("HF_token")
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if dataset repo exists, otherwise create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Path to your CSV
csv_path = "/content/drive/MyDrive/Colab_Notebooks/MLOps_TourismPackagePred/data/tourism.csv"

# Upload the CSV file
api.upload_file(
    path_or_fileobj=csv_path,
    path_in_repo="tourism.csv",   # how it will appear in the repo
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Upload tourism dataset"
)

print("tourism.csv uploaded successfully to Hugging Face Hub!")
