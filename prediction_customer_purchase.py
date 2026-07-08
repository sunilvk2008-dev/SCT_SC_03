# ==========================================================
# Decision Tree Classifier - Bank Marketing Dataset
# ==========================================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# Load Dataset
# ==========================================================

bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features
y = bank_marketing.data.targets

# Convert target dataframe to series
y = y.squeeze()

# ==========================================================
# Dataset Information
# ==========================================================

print("First 5 Rows:")
print(X.head())

print("\nDataset Shape:")
print(X.shape)

print("\nMissing Values:")
print(X.isnull().sum())

print("\nTarget Distribution:")
print(y.value_counts())

# ==========================================================
# Handle Missing Values
# ==========================================================

X = X.fillna("Unknown")

# ==========================================================
# Convert Categorical Columns
# ==========================================================

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Convert target variable
y = y.map({
    "no": 0,
    "yes": 1
})

# ==========================================================
# Split Dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Train Decision Tree
# ==========================================================

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# Prediction
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Accuracy Score")
print("==============================")
print(f"{accuracy:.4f}")

# ==========================================================
# Confusion Matrix
# ==========================================================

print("\n==============================")
print("Confusion Matrix")
print("==============================")
print(confusion_matrix(y_test, y_pred))

# ==========================================================
# Classification Report
# ==========================================================

print("\n==============================")
print("Classification Report")
print("==============================")
print(classification_report(y_test, y_pred))

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print(importance.head(10))

# ==========================================================
# Plot Feature Importance
# ==========================================================

plt.figure(figsize=(12,6))

plt.bar(
    importance["Feature"][:10],
    importance["Importance"][:10]
)

plt.xticks(rotation=45, ha='right')

plt.title("Top 10 Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

# ==========================================================
# Plot Decision Tree
# ==========================================================

plt.figure(figsize=(25,12))

plot_tree(
    model,
    filled=True,
    rounded=True,
    fontsize=7,
    max_depth=3,
    feature_names=X.columns,
    class_names=["No", "Yes"]
)

plt.title("Decision Tree Classifier")

plt.savefig("decision_tree.png")

plt.show()

print("\nImages Saved Successfully!")
print("1. feature_importance.png")
print("2. decision_tree.png")