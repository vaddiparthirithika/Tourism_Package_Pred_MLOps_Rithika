
# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Experiment")

api = HfApi()

# Dataset paths from Hugging Face
Xtrain_path = "hf://datasets/Vaddiritz/Tourism-Package-Prediction-rithika/Xtrain.csv"
Xtest_path = "hf://datasets/Vaddiritz/Tourism-Package-Prediction-rithika/Xtest.csv"
ytrain_path = "hf://datasets/Vaddiritz/Tourism-Package-Prediction-rithika/ytrain.csv"
ytest_path = "hf://datasets/Vaddiritz/Tourism-Package-Prediction-rithika/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

print("Tourism dataset loaded successfully.")

# Feature groups
numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

# Handle class imbalance
class_weight = ytrain.tolist().count(0) / ytrain.tolist().count(1)

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Base model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter distributions (broader for Random Search)
param_distributions = {
    'xgbclassifier__n_estimators': [50, 100, 150, 200, 300],
    'xgbclassifier__max_depth': [3, 4, 5, 6, 8, 10],
    'xgbclassifier__colsample_bytree': np.linspace(0.3, 1.0, 8),
    'xgbclassifier__learning_rate': np.linspace(0.01, 0.3, 10),
    'xgbclassifier__reg_lambda': np.linspace(0.1, 2.0, 10),
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=20,# number of random combinations to try
        cv=5,
        n_jobs=-1,
        random_state=42,
    )
    random_search.fit(Xtrain, ytrain)

    # Log all param sets & scores
    results = random_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best params
    mlflow.log_params(random_search.best_params_)

    # Best model
    best_model = random_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
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

    # Save model
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved at: {model_path}")

    # Upload to Hugging Face Hub
    repo_id = "Vaddiritz/Tourism-Package-Prediction-rithika"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face Hub: {repo_id}")
