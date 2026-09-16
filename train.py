"""
DSAIC Club - Supervised ML I: Logistic Regression
Predicting HELB funding band (1-5) from Means-Testing-Instrument-style factors.

This is MULTINOMIAL logistic regression: the same sigmoid idea generalized
from 2 classes to 5, using a softmax over 5 linear scores instead of one
sigmoid. Great bridge from binary to multi-class classification.

Run:  python3 train.py
Produces:
  model/helb_band_pipeline.joblib   <- deployable pipeline (preprocessing + model)
  plots/*.png                       <- EDA + evaluation charts for the slides
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

DATA_PATH = "data/helb_band_placement.csv"
MODEL_PATH = "model/helb_band_pipeline.joblib"
PLOTS = "plots"
BAND_LABELS = ["Band 1\n(highest need)", "Band 2", "Band 3", "Band 4", "Band 5\n(lowest need)"]

# ---------- 2. Get the data ----------
df = pd.read_csv(DATA_PATH)

# ---------- 3. Explore & visualize ----------
plt.figure(figsize=(5.5, 4))
counts = df["band"].value_counts().sort_index()
plt.bar([str(b) for b in counts.index], counts.values, color="#00A896")
plt.xlabel("Band")
plt.ylabel("Number of students")
plt.title("Students per band")
plt.tight_layout()
plt.savefig(f"{PLOTS}/band_counts.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
df.boxplot(column="household_monthly_income_kes", by="band", ax=plt.gca())
plt.suptitle("")
plt.title("Household income by band")
plt.xlabel("Band")
plt.ylabel("Monthly income (KES)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/income_by_band.png", dpi=150)
plt.close()

orphan_band = pd.crosstab(df["orphan_status"], df["band"], normalize="columns") * 100
plt.figure(figsize=(6.5, 4))
bottom = np.zeros(5)
colors = {"not_orphan": "#00A896", "single_orphan": "#F4A259", "double_orphan": "#990011"}
for status in ["not_orphan", "single_orphan", "double_orphan"]:
    vals = orphan_band.loc[status].values if status in orphan_band.index else np.zeros(5)
    plt.bar([str(b) for b in orphan_band.columns], vals, bottom=bottom, label=status, color=colors[status])
    bottom += vals
plt.xlabel("Band")
plt.ylabel("% of students")
plt.title("Orphan status by band")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOTS}/orphan_by_band.png", dpi=150)
plt.close()

# ---------- 4. Prepare the data ----------
numeric_cols = ["household_monthly_income_kes", "dependents_count", "siblings_in_college"]
binary_cols = ["disability"]
categorical_cols = ["parent_occupation", "orphan_status", "residence_type", "gender"]
feature_cols = numeric_cols + binary_cols + categorical_cols
target_col = "band"

X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("bin", "passthrough", binary_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# ---------- 5. Select and train a model ----------
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000)),  # sklearn auto-selects multinomial for >2 classes
])
pipeline.fit(X_train, y_train)
pred0 = pipeline.predict(X_test)
print(f"Plain multinomial logistic regression -> Accuracy: {accuracy_score(y_test, pred0):.3f}")

# ---------- 6. Fine-tune (grid search over C) ----------
param_grid = {"model__C": [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_macro")
grid.fit(X_train, y_train)
final_pipeline = grid.best_estimator_
print(f"Best C: {grid.best_params_['model__C']} | CV macro-F1: {grid.best_score_:.3f}")

# ---------- 7. Present the solution ----------
pred = final_pipeline.predict(X_test)
acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
report = classification_report(y_test, pred, target_names=[f"Band {b}" for b in sorted(y.unique())])
print(f"Accuracy: {acc:.3f} | Macro F1: {macro_f1:.3f}")
print(report)

cm = confusion_matrix(y_test, pred, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(cm, display_labels=[f"B{b}" for b in sorted(y.unique())])
fig, ax = plt.subplots(figsize=(5.5, 5.5))
disp.plot(ax=ax, cmap="BuGn", colorbar=False)
plt.title("Confusion matrix (5 bands)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/confusion_matrix.png", dpi=150)
plt.close()

# per-band probability example: show how income alone shifts predicted band probabilities
income_range = np.linspace(2000, 80000, 200)
example = pd.DataFrame({
    "household_monthly_income_kes": income_range,
    "dependents_count": 3,
    "siblings_in_college": 1,
    "disability": 0,
    "parent_occupation": "informal_business",
    "orphan_status": "not_orphan",
    "residence_type": "rural_low_poverty",
    "gender": "female",
})
probs = final_pipeline.predict_proba(example)
plt.figure(figsize=(6.5, 4.5))
for i, b in enumerate(sorted(y.unique())):
    plt.plot(income_range / 1000, probs[:, i], label=f"Band {b}", linewidth=2)
plt.xlabel("Household monthly income (KES, thousands)")
plt.ylabel("Predicted probability")
plt.title("How income shifts predicted band\n(other factors held fixed)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOTS}/probability_by_income.png", dpi=150)
plt.close()

with open("model/metrics.txt", "w") as f:
    f.write(f"Best C: {grid.best_params_['model__C']}\n")
    f.write(f"Accuracy: {acc:.3f}\nMacro F1: {macro_f1:.3f}\n\n")
    f.write(report)

# ---------- save the deployable pipeline ----------
joblib.dump(final_pipeline, MODEL_PATH)
print(f"\nSaved pipeline to {MODEL_PATH}")
