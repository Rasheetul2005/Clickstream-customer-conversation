from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

DATA = Path("data")
DATA.mkdir(exist_ok=True)

dataset = fetch_ucirepo(id=553)
df = dataset.data.features.copy()
df.columns = [str(c).strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

# Keep a copy of the official raw data
df.to_csv(DATA / "clickstream_raw.csv", index=False)

# Expected UCI names can differ slightly between versions.
rename = {
    "session_id": "session_id",
    "page_1_main_category": "page_1_main_category",
    "page_2_clothing_model": "page_2_clothing_model",
    "model_photography": "model_photography",
}
df = df.rename(columns=rename)

required = ["session_id","order","price","page","country","page_1_main_category","page_2_clothing_model","colour"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}. Columns found: {df.columns.tolist()}")

g = df.groupby("session_id", sort=False)
session = g.agg(
    click_count=("session_id","size"),
    max_order=("order","max"),
    avg_price=("price","mean"),
    max_price=("price","max"),
    avg_page=("page","mean"),
    max_page=("page","max"),
    country=("country","first"),
    main_category=("page_1_main_category","first"),
    colour=("colour","first")
).reset_index()

for source, target in [
    ("page_2_clothing_model","product_model_diversity"),
    ("page_1_main_category","category_diversity"),
    ("colour","colour_diversity")
]:
    session[target] = g[source].nunique().values

# Documented behavioral proxy: top quartile click-depth sessions.
threshold = session["click_count"].quantile(0.75)
session["converted_proxy"] = (session["click_count"] >= threshold).astype(int)

# Documented potential-value proxy; this is NOT observed transaction revenue.
session["potential_revenue_proxy"] = (
    session["max_price"].fillna(0) * np.log1p(session["click_count"])
)

session.to_csv(DATA / "session_features.csv", index=False)

train, test = train_test_split(
    session,
    test_size=0.20,
    random_state=42,
    stratify=session["converted_proxy"]
)
train.to_csv(DATA / "train.csv", index=False)
test.to_csv(DATA / "test.csv", index=False)

print("Prepared dataset")
print("Raw rows:", len(df))
print("Sessions:", len(session))
print("Train:", len(train), "Test:", len(test))
