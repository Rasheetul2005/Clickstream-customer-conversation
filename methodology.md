# Methodology

## 1. Data source
The project uses UCI Clickstream Data for Online Shopping.

## 2. Preprocessing
- Standardize column names
- Verify data types
- Check duplicates
- Check missing values
- Convert categorical variables using one-hot encoding inside the ML pipeline
- Scale numeric variables where required

## 3. Session feature engineering
Clickstream rows are aggregated by Session ID. Features include:
- click count
- maximum click order
- average/max product price
- average/max website page
- category diversity
- product-model diversity
- colour diversity
- country and category indicators

## 4. Classification
A proxy conversion/intention label is created from session click depth. Logistic Regression and Random Forest are compared.

## 5. Regression
A potential revenue proxy is calculated from product price and session engagement. Linear Regression and Random Forest Regressor are compared.

## 6. Clustering
K-Means is applied to standardized behavioral features. Silhouette Score and Davies-Bouldin Index are reported.

## 7. Limitations
The raw UCI dataset does not contain an explicit purchase-completion or actual revenue variable. Therefore classification and revenue outputs are behavioral proxies, not observed transactions.
