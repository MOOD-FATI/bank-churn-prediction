# 🏦 Bank Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to leave (churn) based on their demographic and account information.

---

## 📁 Project Structure

```
├── Churn_Modelling.ipynb        # Full ML pipeline notebook
├── Churn_Modelling.csv          # Dataset
├── streamlit_app.py             # Interactive prediction web app
├── random_forest_model.joblib   # Trained Random Forest model
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle – Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling)
- **Rows:** 10,000 customers
- **Target:** `Exited` (1 = churned, 0 = stayed)

**Key features:** Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary.

---

## 🔬 Notebook Walkthrough

1. **Data Overview** — shape, nulls, duplicates, descriptive statistics
2. **Feature Engineering** — ratios, interaction terms, binary flags
3. **EDA** — class distribution, boxplots, distribution plots, correlation heatmap
4. **Encoding** — One-Hot Encoding for categorical variables
5. **Train-Test Split** — 80/20 stratified split
6. **Class Imbalance** — handled with SMOTETomek
7. **Model Training** — GridSearchCV on Random Forest, XGBoost, LightGBM
8. **Evaluation** — Accuracy, Classification Report, ROC-AUC curves
9. **Model Saving** — best model persisted with `joblib`

---

## 🤖 Models Compared

| Model | Notes |
|---|---|
| Random Forest | Bagging ensemble; robust and interpretable |
| XGBoost | Gradient boosting with regularization |
| LightGBM | Fast histogram-based gradient boosting |

All models were tuned with 5-fold cross-validated GridSearchCV and evaluated on a held-out test set.

---

## 🚀 Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `LightGBM` · `imbalanced-learn` · `Matplotlib` · `Seaborn` · `Streamlit` · `Joblib`
