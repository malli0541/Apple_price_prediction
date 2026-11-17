# 🍎 Apple Stock Price Prediction

A Streamlit web app that performs EDA, feature engineering, and stock price
prediction using Linear Regression, Random Forest, and XGBoost.

---

## ⚠️ Important – Deployment Platform

**Vercel does NOT support Streamlit.**  
Vercel is for static sites and serverless functions only.

✅ **Use Streamlit Community Cloud** (free, purpose-built for Streamlit):
👉 https://share.streamlit.io

---

## 🚀 Deploy to Streamlit Community Cloud

1. Push this folder to a **public GitHub repository**.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"** → select your repo → set **Main file path** to `app.py`.
4. Click **Deploy** – done!

---

## 🗂️ Project Structure

```
answer/
├── app.py                  ← Streamlit application
├── requirements.txt        ← Python dependencies
├── P721 DATASET.csv        ← Apple stock data (2012–2019)
└── .streamlit/
    └── config.toml         ← Theme configuration
```

---

## 📦 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Features

| Tab | Content |
|-----|---------|
| 📊 EDA | Dataset overview, distributions, correlation heatmap |
| 🔧 Feature Engineering | Lag features, SMAs, EMA, calendar features |
| 🤖 Models | LR / RF / XGBoost training, metrics, feature importance |
| 🔮 Predict | Enter OHLCV → get next-day price prediction |
