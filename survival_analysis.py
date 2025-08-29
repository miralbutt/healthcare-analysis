
"""
Healthcare Survival (Logistic Regression)
----------------------------------------
Simple analysis combining categorical and numeric predictors.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("healthcare_data.csv")

X = df.drop(columns=["survived"])
y = df["survived"]

categorical = ["treatment"]
numeric = [c for c in X.columns if c not in categorical]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("prep", pre),
    ("clf", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model.fit(X_train, y_train)
preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

print("\n=== Logistic Regression ===")
print(classification_report(y_test, preds, digits=3))
print("ROC AUC:", roc_auc_score(y_test, proba))
