"""
Crop Recommendation System using Random Forest
----------------------------------------------
FAI Case Study

Given soil nutrients (N, P, K), temperature, humidity, pH and rainfall,
the model recommends the most suitable crop out of 22 crops.

Run:  python model.py
All charts and results are saved in the outputs/ folder.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

DATA_PATH = os.path.join("data", "Crop_recommendation.csv")
OUT_DIR = "outputs"
RANDOM_STATE = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
print("\n[1] Loading dataset")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Duplicate rows: {df.duplicated().sum()}")
print(f"  Number of crops: {df['label'].nunique()}")

df.describe().round(2).to_csv(os.path.join(OUT_DIR, "dataset_summary.csv"))

# ---------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------
print("\n[2] Exploratory data analysis")

# Class distribution
plt.figure(figsize=(12, 5))
df["label"].value_counts().plot(kind="bar", color="seagreen")
plt.title("Number of Samples per Crop")
plt.xlabel("Crop")
plt.ylabel("Count")
save_fig("01_class_distribution.png")

# Feature distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.flat, FEATURES):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(col)
axes.flat[-1].axis("off")
save_fig("02_feature_distributions.png")

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
save_fig("03_correlation_heatmap.png")

# Average requirement of each crop (normalised so features are comparable)
crop_means = df.groupby("label")[FEATURES].mean()
crop_means_norm = (crop_means - crop_means.min()) / (crop_means.max() - crop_means.min())
plt.figure(figsize=(10, 10))
sns.heatmap(crop_means_norm, cmap="YlGn", annot=crop_means.round(0), fmt="g",
            cbar_kws={"label": "relative level"})
plt.title("Average Conditions per Crop (cell values = actual mean)")
save_fig("04_crop_requirements.png")
crop_means.round(2).to_csv(os.path.join(OUT_DIR, "crop_mean_conditions.csv"))

# Boxplots of key features by crop
for col in ["rainfall", "K"]:
    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df, x="label", y=col, color="lightgreen")
    plt.xticks(rotation=60)
    plt.title(f"{col} by Crop")
    save_fig(f"05_boxplot_{col}.png")

# ---------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------
print("\n[3] Splitting data (80% train / 20% test, stratified)")
X = df[FEATURES]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"  Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

# ---------------------------------------------------------------
# 4. Train Random Forest
# ---------------------------------------------------------------
print("\n[4] Training Random Forest")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}")

report = classification_report(y_test, y_pred, digits=4)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write("Random Forest - Classification Report (test set)\n\n")
    f.write(report)

# 5-fold cross-validation on the full data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------------
print("\n[5] Confusion matrix")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Random Forest Confusion Matrix (Accuracy = {acc:.2%})")
save_fig("06_confusion_matrix.png")

# ---------------------------------------------------------------
# 6. Feature importance
# ---------------------------------------------------------------
print("\n[6] Feature importance")
importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
print(importance.sort_values(ascending=False).round(4).to_string())
plt.figure(figsize=(8, 5))
importance.plot(kind="barh", color="darkgreen")
plt.title("Random Forest Feature Importance (Gini)")
plt.xlabel("Importance")
save_fig("07_feature_importance.png")

# ---------------------------------------------------------------
# 7. Effect of number of trees
# ---------------------------------------------------------------
print("\n[7] Accuracy vs number of trees")
tree_counts = [1, 5, 10, 25, 50, 100, 200]
tree_acc = []
for n in tree_counts:
    m = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train, y_train)
    tree_acc.append(accuracy_score(y_test, m.predict(X_test)))
    print(f"  {n:>3} trees -> {tree_acc[-1]:.4f}")
plt.figure(figsize=(8, 5))
plt.plot(tree_counts, tree_acc, marker="o", color="darkgreen")
plt.xscale("log")
plt.xticks(tree_counts, tree_counts)
plt.xlabel("Number of trees (n_estimators)")
plt.ylabel("Test accuracy")
plt.title("Random Forest: Accuracy vs Number of Trees")
save_fig("08_accuracy_vs_trees.png")

# ---------------------------------------------------------------
# 8. Comparison with other algorithms (baseline)
# ---------------------------------------------------------------
print("\n[8] Comparing with other algorithms")
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p, average="macro"),
        "Recall": recall_score(y_test, p, average="macro"),
        "F1": f1_score(y_test, p, average="macro"),
    })
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print(results_df.round(4).to_string(index=False))
results_df.round(4).to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

plt.figure(figsize=(9, 5))
ax = sns.barplot(data=results_df, x="Accuracy", y="Model", color="seagreen")
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", padding=3)
plt.xlim(0.8, 1.01)
plt.title("Test Accuracy of Different Algorithms")
save_fig("09_model_comparison.png")

# ---------------------------------------------------------------
# 9. Save model and demo prediction
# ---------------------------------------------------------------
print("\n[9] Saving model")
model_path = os.path.join(OUT_DIR, "crop_rf_model.pkl")
joblib.dump(rf, model_path)
print(f"  saved {model_path}")


def recommend_crop(N, P, K, temperature, humidity, ph, rainfall, top_k=3):
    """Return the top_k recommended crops with their probabilities."""
    sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
    probs = rf.predict_proba(sample)[0]
    top = np.argsort(probs)[::-1][:top_k]
    return [(rf.classes_[i], probs[i]) for i in top]


print("\n[10] Example recommendations")
examples = [
    dict(N=90, P=42, K=43, temperature=21, humidity=82, ph=6.5, rainfall=203),
    dict(N=20, P=130, K=200, temperature=22, humidity=92, ph=6.0, rainfall=110),
    dict(N=120, P=45, K=20, temperature=24, humidity=80, ph=6.8, rainfall=80),
]
for ex in examples:
    recs = recommend_crop(**ex)
    print(f"  {ex}")
    print("    -> " + ", ".join(f"{c} ({p:.0%})" for c, p in recs))

# Summary file for the report
with open(os.path.join(OUT_DIR, "results_summary.txt"), "w") as f:
    f.write("Crop Recommendation - Random Forest Results\n")
    f.write("=" * 45 + "\n")
    f.write(f"Dataset: {df.shape[0]} samples, {len(FEATURES)} features, {df['label'].nunique()} crops\n")
    f.write(f"Train/Test: {len(X_train)}/{len(X_test)}\n\n")
    f.write(f"Test Accuracy : {acc:.4f}\n")
    f.write(f"Precision     : {prec:.4f}\n")
    f.write(f"Recall        : {rec:.4f}\n")
    f.write(f"F1-score      : {f1:.4f}\n")
    f.write(f"5-fold CV     : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n")
    f.write("Feature importance:\n")
    f.write(importance.sort_values(ascending=False).round(4).to_string() + "\n\n")
    f.write("Model comparison:\n")
    f.write(results_df.round(4).to_string(index=False) + "\n")

print("\nDone. All outputs are in the 'outputs/' folder.")
