# ============================================
# Standard Library Imports
# ============================================
import os
import joblib

# ============================================
# Third-Party Libraries
# ============================================
import mlflow
import pandas as pd
import xgboost as xgb

from huggingface_hub import HfApi
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================
# MLflow Configuration
# ============================================

# Configure MLflow to log experiments to the remote tracking server.
mlflow.set_tracking_uri(mlflow_public_url)

# Create the experiment if it does not exist, otherwise use the existing one.
MLFLOW_EXPERIMENT_NAME = "Wellness_tourism_MLflow"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("MLflow tracking configured successfully.")
print(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")

# ============================================
# Project Configuration
# ============================================

# Hugging Face dataset repository
HF_DATASET_REPOSITORY = "Jags99/wellness-tourism-dataset/train_test_datasets"

# Hugging Face model repository
HF_MODEL_REPOSITORY = "Jags99/wellness-tourism-model"

# Local filename for the trained model
MODEL_FILENAME = "wellness_tourism_v1.joblib"

# Classification probability threshold
CLASSIFICATION_THRESHOLD = 0.45

# Random seed for reproducibility
RANDOM_STATE = 42

# ============================================
# Initialize Hugging Face API
# ============================================

huggingface_api = HfApi(token=os.getenv("HF_TOKEN"))

# ============================================
# Dataset File Paths
# ============================================

train_features_path = (
    f"hf://datasets/{HF_DATASET_REPOSITORY}/X_train.csv"
)

test_features_path = (
    f"hf://datasets/{HF_DATASET_REPOSITORY}/X_test.csv"
)

train_labels_path = (
    f"hf://datasets/{HF_DATASET_REPOSITORY}/y_train.csv"
)

test_labels_path = (
    f"hf://datasets/{HF_DATASET_REPOSITORY}/y_test.csv"
)

# ============================================
# Load Training and Test Data
# ============================================

training_features = pd.read_csv(train_features_path)
testing_features = pd.read_csv(test_features_path)

training_labels = pd.read_csv(train_labels_path)
testing_labels = pd.read_csv(test_labels_path)


# ============================================
# Feature Definitions
# ============================================

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

numerical_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]


# ============================================
# Compute Class Weight
# ============================================

# Calculate the imbalance ratio between the negative and positive classes.
positive_class_weight = (
    training_labels.value_counts()[0]
    / training_labels.value_counts()[1]
)

print(f"Positive class weight: {positive_class_weight:.2f}")


# ============================================
# Data Preprocessing Pipeline
# ============================================

# Standardize numerical features and one-hot encode categorical features.
# The preprocessing pipeline is automatically applied during both training
# and inference.
preprocessing_pipeline = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)


# ============================================
# XGBoost Model Configuration
# ============================================

# Initialize the XGBoost classifier.
# scale_pos_weight compensates for class imbalance by assigning
# a higher weight to the minority class.
xgboost_classifier = xgb.XGBClassifier(
    scale_pos_weight=positive_class_weight,
    random_state=RANDOM_STATE,
)


# ============================================
# Hyperparameter Grid
# ============================================

# Hyperparameter combinations to evaluate during GridSearchCV.
hyperparameter_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100, 125, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

# ============================================
# Machine Learning Pipeline
# ============================================

# Combine data preprocessing and the classifier into a single pipeline.
# This ensures that identical preprocessing steps are applied during
# cross-validation, training, and inference.
training_pipeline = make_pipeline(
    preprocessing_pipeline,
    xgboost_classifier,
)


# ============================================
# Start MLflow Experiment
# ============================================

with mlflow.start_run():

    # Perform exhaustive hyperparameter tuning using 5-fold
    # cross-validation.
    grid_search_cv = GridSearchCV(
        estimator=training_pipeline,
        param_grid=hyperparameter_grid,
        cv=5,
        n_jobs=-1,
    )

    # Train all candidate models.
    grid_search_cv.fit(training_features, training_labels)

# ============================================
# Log Cross-Validation Results
# ============================================

    cv_results = grid_search_cv.cv_results_

    for (
        hyperparameters,
        mean_validation_score,
        validation_score_std,
    ) in zip(
        cv_results["params"],
        cv_results["mean_test_score"],
        cv_results["std_test_score"],
    ):

        # Create a nested MLflow run for each hyperparameter combination.
        with mlflow.start_run(nested=True):

            mlflow.log_params(hyperparameters)

            mlflow.log_metric(
                "mean_test_score",
                mean_validation_score,
            )

            mlflow.log_metric(
                "std_test_score",
                validation_score_std,
            )

# ============================================
# Retrieve the Best Model
# ============================================

    # Log the best hyperparameters found during GridSearchCV.
    mlflow.log_params(grid_search_cv.best_params_)

    # Retrieve the best-performing pipeline.
    best_trained_model = grid_search_cv.best_estimator_

    # ============================================
    # Generate Predictions
    # ============================================

    # Generate prediction probabilities for the positive class.
    training_prediction_probabilities = (
        best_trained_model.predict_proba(training_features)[:, 1]
    )

    testing_prediction_probabilities = (
        best_trained_model.predict_proba(testing_features)[:, 1]
    )

    # Convert probabilities into binary predictions using the
    # predefined classification threshold.
    training_predictions = (
        training_prediction_probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    testing_predictions = (
        testing_prediction_probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)


    # ============================================
    # Model Evaluation
    # ============================================

    # Generate classification reports for both the training and
    # testing datasets.
    training_classification_report = classification_report(
        training_labels,
        training_predictions,
        output_dict=True,
    )

    testing_classification_report = classification_report(
        testing_labels,
        testing_predictions,
        output_dict=True,
    )

    # ============================================
    # Log Evaluation Metrics
    # ============================================

    # Log the most important performance metrics for both
    # the training and testing datasets.
    mlflow.log_metrics(
        {
            "train_accuracy": training_classification_report["accuracy"],
            "train_precision": training_classification_report["1"]["precision"],
            "train_recall": training_classification_report["1"]["recall"],
            "train_f1_score": training_classification_report["1"]["f1-score"],
            "test_accuracy": testing_classification_report["accuracy"],
            "test_precision": testing_classification_report["1"]["precision"],
            "test_recall": testing_classification_report["1"]["recall"],
            "test_f1_score": testing_classification_report["1"]["f1-score"],
        }
    )

    # ============================================
    # Save the Trained Model
    # ============================================

    # Save the best-performing pipeline locally.
    joblib.dump(best_trained_model, MODEL_FILENAME)

    # Store the trained model as an MLflow artifact.
    mlflow.log_artifact(
        MODEL_FILENAME,
        artifact_path="model",
    )

    print(f"Model saved successfully: {MODEL_FILENAME}")

    # ============================================
    # Upload Model to Hugging Face
    # ============================================

    # Create the model repository if it does not already exist.
    huggingface_api.create_repo(
        repo_id=HF_MODEL_REPOSITORY,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    # Upload the trained model to the repository.
    huggingface_api.upload_file(
        path_or_fileobj=MODEL_FILENAME,
        path_in_repo=MODEL_FILENAME,
        repo_id=HF_MODEL_REPOSITORY,
        repo_type="model",
    )

    print("Model uploaded successfully to Hugging Face.")
