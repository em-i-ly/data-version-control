from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main(repo_path):
    data_path = repo_path / "data"
    raw_path = data_path / "raw"
    prepared_path = data_path / "prepared"
    prepared_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path / "train.csv")  # Kaggle Titanic train.csv

    # Drop irrelevant columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    # Encode categorical columns
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Split into train/test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save prepared datasets
    train_df.to_csv(prepared_path / "train.csv", index=False)
    test_df.to_csv(prepared_path / "test.csv", index=False)

if __name__ == "__main__":
    main(Path(__file__).parent.parent)
