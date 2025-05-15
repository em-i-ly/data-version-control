from joblib import dump
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yaml

def main(repo_path):
    # Load training params
    params = yaml.safe_load((repo_path / "params.yaml").read_text())
    model_type = params["train"]["model"]
    C = params["train"].get("C", 1.0)
    max_iter = params["train"].get("max_iter", 100)
    n_estimators = params["train"].get("n_estimators", 100)

    # Load and split data
    train_df = pd.read_csv(repo_path / "data/prepared/train.csv")
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model
    if model_type == "logistic":
        model = LogisticRegression(C=C, max_iter=max_iter)
    elif model_type == "forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(x_train, y_train)
    dump(model, repo_path / "model/model.joblib")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
