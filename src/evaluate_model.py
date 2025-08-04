from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import json
import os


DATA_PATH = "data/featured_data.csv"
MODEL_DIR = "models/"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)

def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model {name} not found at {path}")
    return joblib.load(path)

def save_metrics_json_csv(model_name, metrics_dict, output_dir="evaluation"):
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    with open(os.path.join(output_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # CSV
    pd.DataFrame([metrics_dict]).to_csv(
        os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False
    )

def plot_confusion_matrix(cm, labels, model_name, output_dir="evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

def evaluate(model, X_test, y_test_encoded, model_name, inverse_mapping):
    y_pred_encoded = model.predict(X_test)
    y_pred = [inverse_mapping[p] for p in y_pred_encoded]
    y_true = [inverse_mapping[t] for t in y_test_encoded]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n====== {model_name.upper()} ======")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Save to disk
    metrics = {
        "model": model_name,
        "accuracy": acc,
        "f1_weighted": f1,
        "confusion_matrix": cm.tolist()
    }
    save_metrics_json_csv(model_name, metrics)
    plot_confusion_matrix(cm, labels=inverse_mapping.values(), model_name=model_name)


def main():
    print("Evaluating models...")

    df = load_data(DATA_PATH)
    X = df.drop(columns=["Activity Level"])
    y = df["Activity Level"]

    mapping = joblib.load(os.path.join(MODEL_DIR, "activity_level_mapping.pkl"))
    y = y.map(mapping)
    inverse_mapping = {v: k for k, v in mapping.items()}

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #model_names = ["xgboost"]

    model_names = ["logistic_regression", "random_forest", "xgboost"]

    for name in model_names:
        model = load_model(name)
        evaluate(model, X_test, y_test, name, inverse_mapping)

if __name__ == "__main__":
    main()