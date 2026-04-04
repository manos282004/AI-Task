"""
Basic ML pipeline for fresher-level submission.
Dataset: Iris (built into scikit-learn)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 1) Load dataset
iris = load_iris(as_frame=True)
X = iris.data.copy()
y = iris.target.copy()

# Add a few missing values so preprocessing step is shown
X.loc[2, "sepal length (cm)"] = np.nan
X.loc[15, "petal width (cm)"] = np.nan

print("Dataset chosen: Iris dataset from scikit-learn")
print("Total rows:", len(X))
print("Missing values before preprocessing:")
print(X.isna().sum())

# 2) Basic preprocessing (handle missing values)
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 3) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4) Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5) Evaluation metric
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nModel used: DecisionTreeClassifier")
print(f"Accuracy: {acc:.4f}")

# 6) Show 2-3 sample predictions
sample_count = 3
sample_X = X_test.head(sample_count)
sample_true = y_test.head(sample_count)
sample_pred = model.predict(sample_X)

print("\nSample predictions:")
for i in range(sample_count):
    true_name = iris.target_names[sample_true.iloc[i]]
    pred_name = iris.target_names[sample_pred[i]]
    print(f"Sample {i + 1}: true={true_name}, predicted={pred_name}")
