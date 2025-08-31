
# for data manipulation
import pandas as pd
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face hub API
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load Hugging Face Token
#os.environ["HF_TOKEN"] = userdata.get("HF_token")
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
api = HfApi()

# Download dataset file from HF repo
local_path = hf_hub_download(
    repo_id="Vaddiritz/Tourism-Package-Prediction-rithika",
    repo_type="dataset",
    filename="tourism.csv",
    token=os.environ["HF_TOKEN"]
)

# Load into pandas
df = pd.read_csv(local_path)
print("Dataset loaded. Shape:", df.shape)


# Basic Cleaning

# Drop unique identifier column
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)
    print("Removed CustomerID column.")

# Drop index-like column if present
if "Unnamed: 0" in df.columns:
  df = df.drop(columns=["Unnamed: 0"])
  print("Dropped 'Unnamed: 0'")

# Fix typos/inconsistent categories
if "Gender" in df.columns:
  df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Handle missing values
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])
    print(f"Encoded {col}")


# Split into features and target
target_col = "ProdTaken" 
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/test split done:", Xtrain.shape, Xtest.shape)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload back to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="Vaddiritz/Tourism-Package-Prediction-rithika",
        repo_type="dataset",
        token=os.environ["HF_TOKEN"]
    )

print("Data prep finished and uploaded to HF.")
