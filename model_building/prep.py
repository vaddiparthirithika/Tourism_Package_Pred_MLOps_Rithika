
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv

# -------------------------------
# Hugging Face setup
# -------------------------------
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
api = HfApi()

# Download dataset from Hugging Face
local_path = hf_hub_download(
    repo_id="Vaddiritz/Tourism-Package-Prediction-rithika_new",
    repo_type="dataset",
    filename="tourism.csv",
    token=hf_token
)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(local_path)
print("Dataset loaded. Shape:", df.shape)

# -------------------------------
# Basic cleaning
# -------------------------------
# Drop unique identifier column
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)
    print("Removed CustomerID column.")

# Drop unnamed index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
    print("Dropped 'Unnamed: 0' column.")

# Fix typos in categorical columns
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Fill missing values
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# Split into features and target
# -------------------------------
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/test split done:")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape, "ytest:", ytest.shape)

# -------------------------------
# Save locally
# -------------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# -------------------------------
# Upload datasets to Hugging Face
# -------------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Check shapes before upload
assert Xtrain.shape[0] == ytrain.shape[0], f"Xtrain rows ({Xtrain.shape[0]}) != ytrain rows ({ytrain.shape[0]})"
assert Xtest.shape[0] == ytest.shape[0], f"Xtest rows ({Xtest.shape[0]}) != ytest rows ({ytest.shape[0]})"

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="Vaddiritz/Tourism-Package-Prediction-rithika_new",
        repo_type="dataset",
        token=hf_token
    )

print("Data prep finished and uploaded to Hugging Face Hub.")
