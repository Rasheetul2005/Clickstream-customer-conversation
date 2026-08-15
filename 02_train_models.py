from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.cluster import KMeans

DATA, MODELS = Path("data"), Path("models")
MODELS.mkdir(exist_ok=True)
df = pd.read_csv(DATA / "train.csv")

drop_cols = ["session_id","converted_proxy","potential_revenue_proxy"]
X = df.drop(columns=drop_cols, errors="ignore")

cat = X.select_dtypes(include="object").columns.tolist()
num = [c for c in X.columns if c not in cat]

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ])

# Classification
Xtr, Xv, ytr, yv = train_test_split(
    X, df["converted_proxy"], test_size=.20, random_state=42,
    stratify=df["converted_proxy"]
)
classifiers = {
    "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1
    )
}
class_results = []
for name, model in classifiers.items():
    pipe = Pipeline([("preprocess", make_preprocessor()), ("model", model)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xv)
    prob = pipe.predict_proba(Xv)[:, 1]
    class_results.append({
        "model": name,
        "accuracy": accuracy_score(yv,pred),
        "precision": precision_score(yv,pred,zero_division=0),
        "recall": recall_score(yv,pred,zero_division=0),
        "f1": f1_score(yv,pred,zero_division=0),
        "roc_auc": roc_auc_score(yv,prob)
    })
    joblib.dump(pipe, MODELS / f"{name}_classifier.joblib")
pd.DataFrame(class_results).to_csv(DATA / "classification_results.csv", index=False)

# Regression
Xtr, Xv, ytr, yv = train_test_split(
    X, df["potential_revenue_proxy"], test_size=.20, random_state=42
)
regressors = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
}
reg_results = []
for name, model in regressors.items():
    pipe = Pipeline([("preprocess", make_preprocessor()), ("model", model)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xv)
    reg_results.append({
        "model": name,
        "mae": mean_absolute_error(yv,pred),
        "rmse": mean_squared_error(yv,pred) ** 0.5,
        "r2": r2_score(yv,pred)
    })
    joblib.dump(pipe, MODELS / f"{name}_regressor.joblib")
pd.DataFrame(reg_results).to_csv(DATA / "regression_results.csv", index=False)

# Clustering
cluster_cols = [
    "click_count","max_order","avg_price","max_price","avg_page","max_page",
    "product_model_diversity","category_diversity","colour_diversity"
]
Z = df[cluster_cols].fillna(df[cluster_cols].median())
scaler = StandardScaler()
Zs = scaler.fit_transform(Z)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(Zs)
df["cluster"] = labels

sil = silhouette_score(Zs, labels)
db = davies_bouldin_score(Zs, labels)
pd.DataFrame({
    "metric":["silhouette_score","davies_bouldin_index"],
    "value":[sil,db]
}).to_csv(DATA / "clustering_results.csv", index=False)

joblib.dump((scaler,kmeans,cluster_cols), MODELS / "kmeans.joblib")
df[["session_id","cluster"]].to_csv(DATA / "cluster_assignments.csv", index=False)

print("Classification results")
print(pd.DataFrame(class_results))
print("Regression results")
print(pd.DataFrame(reg_results))
print("Clustering: silhouette =", sil, "DB index =", db)
