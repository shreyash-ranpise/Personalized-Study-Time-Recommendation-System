from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

NUMERIC_FEATURES = [
    "sleep_hours",
    "class_hours",
    "travel_time",
    "screen_time",
    "subject_difficulty",
    "previous_marks",
    "backlogs",
]

@dataclass
class StudyModels:
    hours_pipeline: Pipeline
    slot_pipeline: Pipeline

def build_preprocessor() -> ColumnTransformer:
    """
    Builds a simple preprocessor for numerical features.
    We only have numeric inputs here, so just scale them.
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, NUMERIC_FEATURES)],
        remainder="drop",
    )
    return preprocessor

def train_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[StudyModels, dict]:
    """
    Trains:
      - LinearRegression to predict study_hours
      - DecisionTreeClassifier to predict study_time_slot

    Returns the fitted models and basic evaluation metrics.
    """
    features = df[NUMERIC_FEATURES]
    target_hours = df["study_hours"]
    target_slot = df["study_time_slot"]

    X_train, X_test, y_hours_train, y_hours_test, y_slot_train, y_slot_test = train_test_split(
        features,
        target_hours,
        target_slot,
        test_size=test_size,
        random_state=random_state,
        stratify=target_slot,
    )

    preprocessor = build_preprocessor()

    hours_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    slot_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=random_state, max_depth=4)),
        ]
    )

    hours_pipeline.fit(X_train, y_hours_train)
    slot_pipeline.fit(X_train, y_slot_train)

    # Evaluate
    hours_pred = hours_pipeline.predict(X_test)
    slot_pred = slot_pipeline.predict(X_test)

    # Some scikit-learn versions don't support the `squared` argument, so compute RMSE manually.
    mse = float(mean_squared_error(y_hours_test, hours_pred))
    rmse = float(np.sqrt(mse))

    metrics = {
        "hours_mae": float(mean_absolute_error(y_hours_test, hours_pred)),
        "hours_rmse": rmse,
        "slot_accuracy": float(accuracy_score(y_slot_test, slot_pred)),
        "test_samples": int(len(X_test)),
    }

    models = StudyModels(hours_pipeline=hours_pipeline, slot_pipeline=slot_pipeline)
    return models, metrics

def recommend_study_plan(
    models: StudyModels,
    inputs: dict,
    max_hours_per_day: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Given the trained models and a dict of input values, predict:
      - recommended study hours (float)
      - best study time slot (Morning/Evening/Night)
    """
    data = pd.DataFrame([inputs])
    hours = float(models.hours_pipeline.predict(data)[0])
    slot = str(models.slot_pipeline.predict(data)[0])

    # Clamp hours to a reasonable range
    upper_bound = 8.0
    if max_hours_per_day is not None:
        upper_bound = min(upper_bound, max_hours_per_day)

    # Ensure upper bound is not negative
    upper_bound = max(0.0, upper_bound)

    hours = max(0.5, min(hours, upper_bound)) if upper_bound >= 0.5 else upper_bound
    return round(hours, 2), slot

if __name__ == "__main__":
    # Simple manual test if run directly
    from data_loader import load_student_data

    df = load_student_data()
    models, metrics = train_models(df)
    print("Metrics:", metrics)
    sample_input = {
        "sleep_hours": 7,
        "class_hours": 5,
        "travel_time": 1,
        "screen_time": 3,
        "subject_difficulty": 3,
        "previous_marks": 70,
    }
    hours, slot = recommend_study_plan(models, sample_input)
    print("Recommendation:", hours, "hours,", slot)

