import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load the dataset
data = pd.read_csv(
    "german.data",
    sep=r"\s+",
    header=None,
    engine="python"
)

columns = [
    "checking_account", "duration", "credit_history",
    "purpose", "credit_amount", "savings_account",
    "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing",
    "existing_credits", "job", "dependents",
    "telephone", "foreign_worker", "target"
]

data.columns = columns

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# 2. Separate input features and target
X = data.drop(columns=["target"])

# Original labels: 1 = good credit, 2 = bad credit
# New labels: 0 = good credit, 1 = bad credit
y = data["target"].map({1: 0, 2: 1})

# 3. Identify column types
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

# 4. Prepare preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Define two models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

best_model = None
best_model_name = None
best_auc = -1

# 7. Train and evaluate each model
for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Good credit", "Bad credit"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if auc > best_auc:
        best_auc = auc
        best_model = pipeline
        best_model_name = name

# 8. Save the selected model
joblib.dump(best_model, "credit_scoring_model.pkl")

print("\n" + "=" * 40)
print("TRAINING COMPLETED!")
print("Selected model:", best_model_name)
print("Best ROC-AUC:", round(best_auc, 4))
print("Saved model: credit_scoring_model.pkl")
print("=" * 40)