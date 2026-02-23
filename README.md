# CS5710-Homework-2
# CS5710 - Machine Learning
## Homework Assignment 2
Spring 2026
Student Name: Vemparala Nishitha
Student id : 700774602
University: University of Central Missouri

## 📌 Overview
This assignment covers:
- Decision Trees
- k-Nearest Neighbors
- Model Evaluation Metrics
- ROC Curve Analysis

---

## 🔹 Part A
Includes:
- Decision stump error calculation
- Entropy and Information Gain
- Confusion matrix metrics
- kNN distance computation
- Cross-validation analysis


# ==============================
# Q7 - Decision Tree (Iris)
# ==============================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

depths = [1, 2, 3]

print("Decision Tree Results:\n")

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Max Depth = {depth}")
    print(f"Training Accuracy = {train_acc:.4f}")
    print(f"Test Accuracy     = {test_acc:.4f}")
    print("-" * 30)

# ==============================
# Q8 - kNN Decision Boundaries
# ==============================

from sklearn.neighbors import KNeighborsClassifier

# Use only 2 features: sepal length & sepal width
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

k_values = [1, 3, 5, 10]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k')
    plt.title(f"kNN Decision Boundary (k={k})")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.show()

# ==============================
# Q9 - Performance Evaluation
# ==============================

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Use all features again
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# kNN with k=5
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ROC Curve (One-vs-Rest for multi-class)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

plt.figure()

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - kNN (k=5)")
plt.legend()
plt.show()

    
