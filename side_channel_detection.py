import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

# CREATE DATASET
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

print("Dataset Generated Successfully")

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# SCALE DATA
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LOGISTIC REGRESSION
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))

# RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

# XGBOOST
xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("\nXGBoost Accuracy:")
print(accuracy_score(y_test, xgb_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix - XGBoost")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ROC CURVE
xgb_probs = xgb.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    xgb_probs
)

auc_score = roc_auc_score(
    y_test,
    xgb_probs
)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr)

plt.title(f"ROC Curve (AUC = {auc_score:.3f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()