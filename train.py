"""Train the potability model and save it to model.joblib (run automatically at Render build)."""
import json
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score

RS = 42
df = pd.read_csv("water_potability.csv")
FEATURES = [c for c in df.columns if c != "Potability"]
X, y = df[FEATURES], df["Potability"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

def pipe(m):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", m)])

candidates = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=RS, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
scores = {n: cross_val_score(pipe(m), X_tr, y_tr, cv=cv, scoring="f1").mean() for n, m in candidates.items()}
print("CV F1:", {k: round(v, 3) for k, v in scores.items()})
best_name = max(scores, key=scores.get)
model = pipe(candidates[best_name])

# choose decision threshold on out-of-fold train predictions (never on test)
oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
ths = np.linspace(0.2, 0.8, 61)
thr = float(ths[int(np.argmax([f1_score(y_tr, oof >= t) for t in ths]))])

model.fit(X_tr, y_tr)
p = model.predict_proba(X_te)[:, 1]; pred = p >= thr
metrics = {"model": best_name, "threshold": round(thr, 3),
           "accuracy": round(accuracy_score(y_te, pred), 3), "precision": round(precision_score(y_te, pred), 3),
           "recall": round(recall_score(y_te, pred), 3), "f1": round(f1_score(y_te, pred), 3),
           "roc_auc": round(roc_auc_score(y_te, p), 3),
           "majority_baseline_accuracy": round(1 - y_te.mean(), 3)}
print(metrics)

model.fit(X, y)  # final model on all data
joblib.dump({"model": model, "threshold": thr, "features": FEATURES}, "model.joblib")
json.dump(metrics, open("metrics.json", "w"), indent=2)
