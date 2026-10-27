"""Train on lungs_cancer.csv, compare models with cross-validation, save ONE pipeline."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "PULMONARY_DISEASE"
NUMERIC = ["AGE", "ENERGY_LEVEL", "OXYGEN_SATURATION"]  # binary columns need no scaling

df = pd.read_csv("lungs_cancer.csv")
df[TARGET] = df[TARGET].map({"YES": 1, "NO": 0})

X = df.drop(columns=[TARGET])
y = df[TARGET]
FEATURES = list(X.columns)

# stratify keeps the YES/NO ratio the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_num = ColumnTransformer([("num", StandardScaler(), NUMERIC)], remainder="passthrough")
candidates = {
    "logistic_regression": Pipeline([("pre", scale_num), ("clf", LogisticRegressionCV(max_iter=2000))]),
    "random_forest": Pipeline([("pre", scale_num), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = {}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_auc[name] = scores.mean()
    print(f"{name}: CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

best_name = max(cv_auc, key=cv_auc.get)
best = candidates[best_name].fit(X_train, y_train)
print("\nSelected:", best_name)

proba = best.predict_proba(X_test)[:, 1]
pred = best.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("Test ROC-AUC:", round(roc_auc_score(y_test, proba), 4))

if best_name == "random_forest":
    imp = pd.Series(best.named_steps["clf"].feature_importances_,
                    index=NUMERIC + [c for c in FEATURES if c not in NUMERIC])
    print("\nTop features:\n", imp.sort_values(ascending=False).head(8).round(3))

joblib.dump({"pipeline": best, "features": FEATURES}, "model.joblib")
print("\nSaved model.joblib")
