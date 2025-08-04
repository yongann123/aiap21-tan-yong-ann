from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import json
import os

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = "data/featured_data.csv"
MODEL_DIR = "models/"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
    return pd.read_csv(input_path)

def train_and_save_model(model, model_name, X_train, y_train):
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    print(f"[✓] Saved {model_name} model.")
    return model

def encode_activity_level(y, model_dir="models"):
    """
    Manually encodes Activity Level as an ordinal variable:
    Low Activity (0), Moderate Activity (1), High Activity (2)
    Saves mapping dictionary for inverse transform.
    """
    mapping = {
        'Low Activity': 0,
        'Moderate Activity': 1,
        'High Activity': 2
    }

    # Check for unseen categories
    if not set(y.unique()).issubset(set(mapping.keys())):
        raise ValueError("Unexpected category found in 'Activity Level'. Check for typos or normalization issues.")

    y_encoded = y.map(mapping)

    # Save mapping
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(mapping, os.path.join(model_dir, "activity_level_mapping.pkl"))
    logging.info("Saved custom label mapping to models/activity_level_mapping.pkl")

    return y_encoded

def save_best_params(model_name, best_params, output_dir="models"):
    """Saves the best hyperparameters to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{model_name}_best_params.json")
    with open(filepath, "w") as f:
        json.dump(best_params, f, indent=4)
    logging.info(f"Saved best hyperparameters for {model_name} to {filepath}")

def get_best_model(model_name, X_train, y_train):
    """
    Returns the best estimator for a given model name using CV hyperparameter tuning.
    Uses GridSearchCV for logistic regression, and RandomizedSearchCV for tree-based models.
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'f1_weighted'

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear']  # Supports both L1 and L2
        }
        grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                            cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        print(f"Best hyperparameters for {model_name}: {grid.best_params_}")
        save_best_params(model_name, grid.best_params_)
        return grid.best_estimator_

    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': [100, 200],          
            'max_depth': [10, 20, 50, None],         
            'min_samples_split': [2, 5],         
            'min_samples_leaf': [1, 2],          
            'bootstrap': [True, False]                  
        }
        rand_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                         n_iter=15, cv=cv, scoring=scoring, 
                                         n_jobs=-1, verbose=1, random_state=42)
        rand_search.fit(X_train, y_train)
        print(f"Best hyperparameters for {model_name}: {rand_search.best_params_}")
        save_best_params(model_name, rand_search.best_params_)
        return rand_search.best_estimator_

    elif model_name == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        param_dist = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        rand_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                         n_iter=30, cv=cv, scoring=scoring,
                                         n_jobs=-1, verbose=1, random_state=42)
        rand_search.fit(X_train, y_train)
        print(f"Best hyperparameters for {model_name}: {rand_search.best_params_}")
        save_best_params(model_name, rand_search.best_params_)
        return rand_search.best_estimator_

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    print("Training models...")

    df = load_data(INPUT_PATH)
    
    X = df.drop(columns=["Activity Level"])
    y = df["Activity Level"]

    # Encode target
    y = encode_activity_level(y)    

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "logistic_regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(eval_metric='mlogloss')
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        best_model = get_best_model(name, X_train, y_train)

        # Save the tuned model directly — no re-fitting
        joblib.dump(best_model, os.path.join(MODEL_DIR, f"{name}.pkl"))

        # Predict on held-out test set
        preds = best_model.predict(X_test)

        # Decode predictions and true labels
        mapping = joblib.load("models/activity_level_mapping.pkl")
        inverse_mapping = {v: k for k, v in mapping.items()}
        y_labels = [inverse_mapping[pred] for pred in preds]
        y_test_labels = [inverse_mapping[true] for true in y_test]

        print(f"\n{name} classification report:")
        print(classification_report(y_test_labels, y_labels))

if __name__ == "__main__":
    main()