
from pathlib import Path
import urllib.request

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


DATA_URL = (
    "https://raw.githubusercontent.com/agconti/kaggle-titanic/"
    "master/data/train.csv"
)

FEATURE_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]


def download_dataset(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()

    path.write_bytes(content)

    check = pd.read_csv(path)

    if len(check) < 800 or "Survived" not in check.columns:
        path.unlink(missing_ok=True)
        raise ValueError(
            "Downloaded file is not the original Titanic training dataset."
        )


def load_dataset(path):
    path = Path(path)

    if not path.exists():
        download_dataset(path)

    df = pd.read_csv(path)

    required = FEATURE_COLUMNS + ["Survived"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Dataset is missing columns: " + ", ".join(missing)
        )

    return df


def prepare_features(frame):
    x = frame[FEATURE_COLUMNS].copy()

    x["Sex"] = x["Sex"].map(
        lambda value: value.strip().lower()
        if pd.notna(value) else np.nan
    )

    x["Embarked"] = x["Embarked"].map(
        lambda value: value.strip().upper()
        if pd.notna(value) else np.nan
    )

    return x


def evaluate_model(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        )
    }


def build_model(df):
    x = prepare_features(df)
    y = df["Survived"].astype(int)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, predictions)

    return (
        model,
        X_test,
        y_test,
        metrics,
        predictions,
        probabilities
    )


def get_feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]

    names = preprocessor.get_feature_names_out()

    coefficients = model.named_steps["classifier"].coef_[0]

    result = pd.DataFrame({
        "Feature": names,
        "Coefficient": coefficients
    })

    result["Feature"] = (
        result["Feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )

    return result.sort_values(
        "Coefficient",
        ascending=True
    )
