# 🍎 Apple Stock Price Prediction

A Streamlit web app that performs EDA, feature engineering, and stock price
prediction using Linear Regression, Random Forest, and XGBoost.

---

## 🚀 Deploy to Vercel

The project exposes a Streamlit ASGI application for Vercel's Python runtime.
Set the Vercel project root to this folder and deploy with the default settings.
The deployment uses Python 3.12, as required by the TensorFlow version range.

Streamlit uses WebSockets for interactive sessions, so Fluid Compute must be
enabled. TensorFlow also exceeds Vercel's standard Python function bundle size;
for an existing Vercel project, add the environment variable
`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in Project Settings and redeploy. Large
Functions are in beta and require an eligible Vercel project.

## 🚀 Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"** → select your repo → set **Main file path** to `app.py`.
4. Click **Deploy**.

---

## 🗂️ Project Structure

```
answer/
├── app.py                  ← ASGI entrypoint for deployment
├── streamlit_ui.py         ← Streamlit application
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
