# AI Engineer Task (Basic Version)

This is a simple beginner-friendly ML pipeline.

## Dataset chosen
- Iris dataset (already available in scikit-learn)

## Model used
- DecisionTreeClassifier

## Steps covered
1. Load dataset
2. Handle missing values (median imputation)
3. Train/test split
4. Train model
5. Print accuracy
6. Show 3 sample predictions

## Line-by-line explanation of `ai_engineer_task.py`
1. `import numpy as np` and `import pandas as pd`: load libraries for missing values (`np.nan`) and table handling.
2. `from sklearn...`: imports ML tools from scikit-learn.
3. `iris = load_iris(as_frame=True)`: loads Iris dataset as pandas DataFrames.
4. `X = iris.data.copy()` and `y = iris.target.copy()`: separates features (`X`) and labels (`y`).
5. Two `X.loc[...] = np.nan` lines: intentionally add missing values so preprocessing is demonstrated.
6. `print(...)` lines: show dataset info and missing-value counts.
7. `imputer = SimpleImputer(strategy="median")`: creates a median-based imputer.
8. `imputer.fit_transform(X)`: learns medians from each feature and fills missing values.
9. `train_test_split(...)`: splits data into train (80%) and test (20%) sets.
10. `model = DecisionTreeClassifier(random_state=42)`: creates a decision tree model.
11. `model.fit(X_train, y_train)`: trains the model on training data.
12. `y_pred = model.predict(X_test)`: predicts labels for test data.
13. `accuracy_score(y_test, y_pred)`: computes accuracy metric.
14. `head(3)` + prediction loop: prints 3 sample true vs predicted class names.

## What scikit-learn does in this project
- `load_iris`: provides a ready-to-use sample dataset.
- `SimpleImputer`: replaces missing values using a chosen rule (median here).
- `train_test_split`: creates fair train/test subsets.
- `DecisionTreeClassifier`: trains a classification model.
- `accuracy_score`: evaluates model performance.

In short, scikit-learn gives standard, reliable building blocks so you can create an ML pipeline with very little code.

## Run
```bash
pip install -r requirements.txt
python ai_engineer_task.py
```

## Push to GitHub (after adding your remote)
```bash
git remote add origin <your-github-repo-url>
git push -u origin HEAD
```
