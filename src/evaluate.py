from joblib import load
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def main(repo_path):
    test_df = pd.read_csv(repo_path / "data/prepared/test.csv")
    labels = test_df["Survived"]
    test_df = test_df.drop("Survived", axis=1)

    model = load(repo_path / "model/model.joblib")
    predictions = model.predict(test_df)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

    summary_path = repo_path / "metrics/accuracy.json"
    summary_path.write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
