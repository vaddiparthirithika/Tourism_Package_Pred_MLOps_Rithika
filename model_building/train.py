
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from dotenv import load_dotenv
import os
from datetime import datetime

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Experiment")

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
api = HfApi()

# -------------------------------
# Download datasets from Hugging Face
# -------------------------------
Xtrain_path = hf_hub_download(repo_id="Vaddiritz/Tourism-Package-Prediction-rithika_new", repo_type="dataset", filename="Xtrain.csv")
Xtest_path  = hf_hub_download(repo_id="Vaddiritz/Tourism-Package-Prediction-rithika_new", repo_type="dataset", filename="Xtest.csv")
ytrain_path = hf_hub_download(repo_id="Vaddiritz/Tourism-Package-Prediction-rithika_new", repo_type="dataset", filename="ytrain.csv")
ytest_path  = hf_hub_download(repo_id="Vaddiritz/Tourism-Package-Prediction-rithika_new", repo_type="dataset", filename="ytest.csv")

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest  = pd.read_csv(ytest_path).values.ravel()

print(f"Datasets loaded successfully. Shapes -> Xtrain: {Xtrain.shape}, ytrain: {ytrain.shape}, Xtest: {Xtest.shape}, ytest: {ytest.shape}")

# -------------------------------
# Feature groups
# -------------------------------
numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

# Since LabelEncoding is done in prep.py, assert no categorical object columns remain
#assert len(categorical_features) == 0, "All categorical features should be encoded already!"

# Handle class imbalance
class_weight = ytrain.tolist().count(0) / ytrain.tolist().count(1)

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Base model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")

# Pipeline
pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter distributions
param_distributions = {
    'xgbclassifier__n_estimators': [50, 100, 150, 200, 300],
    'xgbclassifier__max_depth': [3, 4, 5, 6, 8, 10],
    'xgbclassifier__colsample_bytree': np.linspace(0.3, 1.0, 8),
    'xgbclassifier__learning_rate': np.linspace(0.01, 0.3, 10),
    'xgbclassifier__reg_lambda': np.linspace(0.1, 2.0, 10),
}

# -------------------------------
# Model training with MLflow
# -------------------------------
with mlflow.start_run():
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(Xtrain, ytrain)

    # Log best params and metrics
    best_pipeline = random_search.best_estimator_

    y_pred_train = best_pipeline.predict(Xtrain)
    y_pred_test  = best_pipeline.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_params(random_search.best_params_)
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # -------------------------------
    # Save full pipeline with versioned filename
    # -------------------------------
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"tourism_pipeline.joblib"
    joblib.dump(best_pipeline, model_filename)
    mlflow.log_artifact(model_filename, artifact_path="model")
    print(f"Pipeline saved as: {model_filename}")

    # -------------------------------
    # Upload pipeline to Hugging Face
    # -------------------------------
    repo_id = "Vaddiritz/Tourism-Package-Prediction-rithika_new"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=model_filename,
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Pipeline uploaded to Hugging Face: {repo_id}")
