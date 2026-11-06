"""Rice Type Classification: compare 6 classifiers on grain-shape features.

Run:  python rice_classification.py
Outputs: accuracy table in the terminal + charts in the images/ folder.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

os.makedirs("images", exist_ok=True)

# 1. Load data
df = pd.read_csv("data/riceClassification.csv")
features = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea",
            "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation"]
print("Shape:", df.shape)
print(df["Class"].value_counts(), "\n")

# 2. Quick exploration charts
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("images/correlation_heatmap.png", dpi=120); plt.close()

pcs = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(df[features]))
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Class"], s=20)
plt.title("PCA Projection by Class"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig("images/pca_projection.png", dpi=120); plt.close()

# 3. Split and scale (scaler is fit on training data only, to avoid leakage)
X, y = df[features], df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("Training samples:", len(X_train), "| Testing samples:", len(X_test), "\n")

# 4. Compare six models: single split + 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}
rows = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    single = accuracy_score(y_test, model.predict(X_test_s))
    cv = cross_val_score(model, X_train_s, y_train, cv=skf, scoring="accuracy")
    rows[name] = {"Single Split (%)": round(single * 100, 2),
                  "5-Fold CV (%)": round(cv.mean() * 100, 2),
                  "CV Std (%)": round(cv.std() * 100, 2)}
results = pd.DataFrame(rows).T.sort_values("5-Fold CV (%)", ascending=False)
print(results, "\n")
results.to_csv("results.csv")

x = np.arange(len(results)); w = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - w / 2, results["Single Split (%)"], w, label="Single Split")
ax.bar(x + w / 2, results["5-Fold CV (%)"], w, label="5-Fold CV")
ax.set_xticks(x); ax.set_xticklabels(results.index, rotation=30)
ax.set_ylim(95, 100); ax.set_ylabel("Accuracy (%)")
ax.set_title("Single Split vs 5-Fold CV Accuracy by Model"); ax.legend()
plt.tight_layout(); plt.savefig("images/model_comparison.png", dpi=120); plt.close()

# 5. Confusion matrix and feature importance for the best model (SVM)
svm = SVC().fit(X_train_s, y_train)
ConfusionMatrixDisplay.from_estimator(svm, X_test_s, y_test, cmap="Greens")
plt.title("Confusion Matrix - SVM")
plt.tight_layout(); plt.savefig("images/confusion_matrix.png", dpi=120); plt.close()

imp = permutation_importance(svm, X_test_s, y_test, n_repeats=10, random_state=42)
pd.Series(imp.importances_mean, index=features).sort_values(ascending=False).plot(
    kind="bar", figsize=(10, 5), color="green")
plt.title("Feature Importance (Permutation - SVM)"); plt.ylabel("Drop in Accuracy")
plt.tight_layout(); plt.savefig("images/feature_importance.png", dpi=120); plt.close()
print("Charts saved in images/")
