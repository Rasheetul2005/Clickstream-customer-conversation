# Customer Conversion Analysis for Online Shopping Using Clickstream Data

## Capstone Project

This project implements the requirements in the supplied project brief:
- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Classification
- Regression
- Customer clustering
- Model evaluation
- Scikit-learn pipelines
- Streamlit deployment

## Dataset
UCI Machine Learning Repository — Clickstream Data for Online Shopping (Dataset ID 553).

The raw UCI dataset contains 165,474 clickstream records and 14 features. It is a sequential e-commerce clickstream dataset. The project brief specifically names this dataset.

## Important target note
The raw UCI Clickstream dataset does not provide an explicit purchase/revenue target in its 14 listed features. Therefore this project does NOT claim that a purchase label or actual revenue exists in the raw data.

For the capstone's requested classification/regression demonstrations, the pipeline creates documented behavioral proxy targets at session level:
- `converted_proxy`: high-intent session based on session click depth
- `potential_revenue_proxy`: price/click-depth based estimated value

These are derived analytical targets and must be described as such during evaluation.

## Run locally

```bash
pip install -r requirements.txt
python src/01_download_and_prepare.py
python src/02_train_models.py
streamlit run app.py
```

The first script downloads the official UCI dataset through `ucimlrepo`, creates session-level features, and generates train/test CSV files.
