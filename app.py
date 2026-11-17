import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── TensorFlow (optional) ─────────────────────────────────────────────────────
try:
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Apple Stock Price Prediction", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Open", "High", "Low", "Adj Close", "Volume",
    "Daily_return", "Price_Change", "High_low_pct",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
    "Volume_Lag_1", "Volume_Lag_2",
    "SMA_7", "SMA_14", "SMA_30", "EMA_14",
    "Day_of_Week", "Month", "Quarter", "Is_Month_End",
]
TARGET_COL = "Close"


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "P721 DATASET.csv"
    return pd.read_csv(csv_path)


@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Daily_return"]  = df["Close"].pct_change() * 100
    df["Price_Change"]  = df["Close"] - df["Open"]
    df["High_low_pct"]  = (df["High"] - df["Low"]) / df["Close"] * 100
    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
    for lag in [1, 2]:
        df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)
    df["SMA_7"]  = df["Close"].rolling(7).mean()
    df["SMA_14"] = df["Close"].rolling(14).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["EMA_14"] = df["Close"].ewm(span=14, adjust=False).mean()
    df["Date"]         = pd.to_datetime(df["Date"])
    df["Day_of_Week"]  = df["Date"].dt.dayofweek
    df["Month"]        = df["Date"].dt.month
    df["Quarter"]      = df["Date"].dt.quarter
    df["Is_Month_End"] = df["Date"].dt.is_month_end.astype(int)
    return df.dropna().reset_index(drop=True)


def ts_split(df: pd.DataFrame, test_ratio: float = 0.2):
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def calc_metrics(y_true, y_pred) -> dict:
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R²":   round(float(r2_score(y_true, y_pred)), 4),
    }


# ── Cached model training ─────────────────────────────────────────────────────
@st.cache_resource
def train_ml_models(test_ratio: float):
    raw = load_data()
    df  = engineer_features(raw)
    train_df, test_df = ts_split(df, test_ratio)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1,
                              random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    results = {
        "Linear Regression": calc_metrics(y_test, lr_pred),
        "Random Forest":     calc_metrics(y_test, rf_pred),
        "XGBoost":           calc_metrics(y_test, xgb_pred),
    }
    preds_map = {
        "Linear Regression": lr_pred,
        "Random Forest":     rf_pred,
        "XGBoost":           xgb_pred,
    }
    return lr, rf, xgb_model, results, preds_map, test_df["Date"].values, y_test


# ── LSTM helpers ──────────────────────────────────────────────────────────────
def _make_sequences(scaled: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


def _build_lstm(look_back: int):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ── Load data ─────────────────────────────────────────────────────────────────
raw_df = load_data()
df     = engineer_features(raw_df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
test_ratio = st.sidebar.slider("Test set ratio", 0.10, 0.30, 0.20, 0.05)

if not TF_AVAILABLE:
    st.sidebar.warning("TensorFlow not installed – LSTM Forecast tab disabled.")

st.title("🍎 Apple Stock Price Prediction")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_eda, tab_feat, tab_models, tab_predict, tab_lstm = st.tabs(
    ["📊 EDA", "🔧 Feature Engineering", "🤖 Models", "🔮 Predict", "📡 LSTM Forecast"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – EDA
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",        raw_df.shape[0])
    c2.metric("Columns",     raw_df.shape[1])
    c3.metric("Null values", int(raw_df.isnull().sum().sum()))
    c4, c5 = st.columns(2)
    c4.metric("Duplicates",  int(raw_df.duplicated().sum()))
    c5.metric("Date range",  f"{raw_df['Date'].iloc[0]}  →  {raw_df['Date'].iloc[-1]}")

    st.dataframe(raw_df.head(10), use_container_width=True)
    st.write("**Descriptive Statistics**")
    st.dataframe(raw_df.describe().round(4), use_container_width=True)

    st.subheader("Distributions & Box-plots")
    num_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(num_cols), 2, figsize=(14, 4 * len(num_cols)))
    for i, col in enumerate(num_cols):
        sns.histplot(data=raw_df, x=col, kde=True, ax=axes[i, 0], color="#0071E3")
        axes[i, 0].set_title(f"Histogram – {col}"); axes[i, 0].set_xlabel("")
        sns.boxplot(data=raw_df, x=col, ax=axes[i, 1], color="#34C759")
        axes[i, 1].set_title(f"Box Plot – {col}");  axes[i, 1].set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.subheader("Closing Price Over Time")
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(pd.to_datetime(raw_df["Date"]), raw_df["Close"], color="#0071E3", linewidth=1)
    ax2.set_xlabel("Date"); ax2.set_ylabel("Close Price (USD)")
    ax2.set_title("Apple Closing Price (2012 – 2019)")
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(raw_df[num_cols].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax3)
    plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Feature Engineering
# ════════════════════════════════════════════════════════════════════════════
with tab_feat:
    st.subheader("Engineered Features (first 10 rows)")
    show_cols = ["Date", "Close", "Daily_return", "Price_Change",
                 "High_low_pct", "SMA_7", "SMA_14", "SMA_30", "EMA_14",
                 "Close_Lag_1", "Day_of_Week", "Month"]
    st.dataframe(df[show_cols].head(10), use_container_width=True)

    st.subheader("Moving Averages vs Close Price")
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    for col, color, lbl in [
        ("Close",  "#0071E3", "Close"),
        ("SMA_7",  "#FF9F0A", "SMA 7"),
        ("SMA_14", "#34C759", "SMA 14"),
        ("SMA_30", "#AF52DE", "SMA 30"),
        ("EMA_14", "#FF3B30", "EMA 14"),
    ]:
        ax4.plot(df["Date"], df[col], label=lbl, linewidth=1, color=color)
    ax4.legend(); ax4.set_xlabel("Date"); ax4.set_ylabel("Price (USD)")
    ax4.set_title("Moving Averages")
    plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    st.subheader("Daily Return Distribution")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Daily_return"].dropna(), kde=True, ax=ax5, color="#0071E3")
    ax5.set_xlabel("Daily Return (%)")
    plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)

    st.subheader("High-Low % Volatility Over Time")
    fig6, ax6 = plt.subplots(figsize=(14, 4))
    ax6.plot(df["Date"], df["High_low_pct"], color="#FF9F0A", linewidth=0.8)
    ax6.set_xlabel("Date"); ax6.set_ylabel("High-Low %")
    ax6.set_title("Daily Intraday Volatility")
    plt.tight_layout(); st.pyplot(fig6); plt.close(fig6)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Models  (cached – only re-trains when test_ratio changes)
# ════════════════════════════════════════════════════════════════════════════
with tab_models:
    with st.spinner("Training models…"):
        lr, rf, xgb_model, results, preds_map, test_dates, y_test = train_ml_models(test_ratio)

    st.subheader("📋 Model Comparison")
    metrics_df = pd.DataFrame(results).T
    st.dataframe(
        metrics_df.style
            .highlight_min(axis=0, subset=["MAE", "RMSE"], color="#d4edda")
            .highlight_max(axis=0, subset=["R²"],          color="#d4edda"),
        use_container_width=True,
    )

    st.subheader("📈 Actual vs Predicted – Test Set")
    for model_name, preds in preds_map.items():
        plot_dates  = test_dates[-len(preds):]
        plot_actual = y_test[-len(preds):]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(plot_dates, plot_actual, label="Actual",     color="#0071E3", linewidth=1.2)
        ax.plot(plot_dates, preds,       label=model_name,   color="#FF3B30", linewidth=1.2, linestyle="--")
        ax.set_title(f"{model_name} – Actual vs Predicted")
        ax.set_xlabel("Date"); ax.set_ylabel("Close Price (USD)")
        ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("🔍 Feature Importance")
    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        st.write("**Random Forest**")
        fi_rf = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        fig_rf, ax_rf = plt.subplots(figsize=(6, 6))
        fi_rf.head(15).plot(kind="barh", ax=ax_rf, color="#0071E3")
        ax_rf.invert_yaxis(); ax_rf.set_xlabel("Importance")
        plt.tight_layout(); st.pyplot(fig_rf); plt.close(fig_rf)
    with col_fi2:
        st.write("**XGBoost**")
        fi_xgb = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        fig_xgb, ax_xgb = plt.subplots(figsize=(6, 6))
        fi_xgb.head(15).plot(kind="barh", ax=ax_xgb, color="#34C759")
        ax_xgb.invert_yaxis(); ax_xgb.set_xlabel("Importance")
        plt.tight_layout(); st.pyplot(fig_xgb); plt.close(fig_xgb)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – Predict (next-day, all 3 ML models)
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("🔮 Predict Next-Day Close Price")
    st.info("Enter today's OHLCV values. The trained models will estimate tomorrow's closing price.")

    # ensure models are loaded even if user lands here first
    with st.spinner("Loading models…"):
        lr, rf, xgb_model, _, _, _, _ = train_ml_models(test_ratio)

    last = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    open_  = c1.number_input("Open",      value=float(last["Open"]),      step=0.01, format="%.4f", key="p_open")
    high_  = c2.number_input("High",      value=float(last["High"]),      step=0.01, format="%.4f", key="p_high")
    low_   = c3.number_input("Low",       value=float(last["Low"]),       step=0.01, format="%.4f", key="p_low")
    close_ = c1.number_input("Close",     value=float(last["Close"]),     step=0.01, format="%.4f", key="p_close")
    adj_   = c2.number_input("Adj Close", value=float(last["Adj Close"]), step=0.01, format="%.4f", key="p_adj")
    vol_   = c3.number_input("Volume",    value=float(last["Volume"]),    step=1.0,  format="%.0f", key="p_vol")

    if st.button("🚀 Predict", use_container_width=True, key="predict_btn"):
        prev_close = float(last["Close"])
        row = {
            "Open":         open_,
            "High":         high_,
            "Low":          low_,
            "Adj Close":    adj_,
            "Volume":       vol_,
            "Daily_return": (close_ - prev_close) / prev_close * 100,
            "Price_Change": close_ - open_,
            "High_low_pct": (high_ - low_) / close_ * 100,
            "Close_Lag_1":  close_,
            "Close_Lag_2":  float(df.iloc[-1]["Close"]),
            "Close_Lag_3":  float(df.iloc[-2]["Close"]),
            "Close_Lag_5":  float(df.iloc[-4]["Close"]),
            "Volume_Lag_1": vol_,
            "Volume_Lag_2": float(last["Volume"]),
            "SMA_7":        (df["Close"].iloc[-6:].sum() + close_) / 7,
            "SMA_14":       (df["Close"].iloc[-13:].sum() + close_) / 14,
            "SMA_30":       (df["Close"].iloc[-29:].sum() + close_) / 30,
            "EMA_14":       float(last["EMA_14"]),
            "Day_of_Week":  (int(last["Day_of_Week"]) + 1) % 5,
            "Month":        int(last["Month"]),
            "Quarter":      int(last["Quarter"]),
            "Is_Month_End": 0,
        }
        X_new = np.array([[row[f] for f in FEATURE_COLS]])

        pred_lr  = float(lr.predict(X_new)[0])
        pred_rf  = float(rf.predict(X_new)[0])
        pred_xgb = float(xgb_model.predict(X_new)[0])

        st.success("✅ Predictions for the next trading day:")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Linear Regression", f"${pred_lr:.2f}",  delta=f"{pred_lr  - close_:+.2f}")
        pc2.metric("Random Forest",     f"${pred_rf:.2f}",  delta=f"{pred_rf  - close_:+.2f}")
        pc3.metric("XGBoost",           f"${pred_xgb:.2f}", delta=f"{pred_xgb - close_:+.2f}")
        st.caption("Delta = predicted price − today's close price you entered.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 – LSTM Future Forecast (30 / 60 / 90 days)
# ════════════════════════════════════════════════════════════════════════════
with tab_lstm:
    st.subheader("📡 LSTM Future Price Forecast")

    if not TF_AVAILABLE:
        st.error(
            "TensorFlow is not installed. "
            "Add `tensorflow` to requirements.txt and redeploy to enable LSTM forecasting."
        )
    else:
        st.info(
            "Train LSTM on the full dataset, then forecast 30 / 60 / 90 trading days "
            "into the future. Each step feeds the previous prediction back as input."
        )

        col_a, col_b, col_c = st.columns(3)
        forecast_horizon = col_a.selectbox("Forecast horizon (days)", [30, 60, 90], index=0)
        lstm_lb = col_b.slider("Look-back window", 30, 120, 60, 10, key="lstm_lb")
        lstm_ep = col_c.slider("Epochs",           5,  50,  20,  5,  key="lstm_ep")

        st.markdown("**Seed the forecast with the last known values** (edit to simulate scenarios):")
        last_row = df.iloc[-1]
        fc1, fc2, fc3 = st.columns(3)
        f_open  = fc1.number_input("Open",      value=float(last_row["Open"]),      step=0.01, format="%.4f", key="f_open")
        f_high  = fc2.number_input("High",      value=float(last_row["High"]),      step=0.01, format="%.4f", key="f_high")
        f_low   = fc3.number_input("Low",       value=float(last_row["Low"]),       step=0.01, format="%.4f", key="f_low")
        f_close = fc1.number_input("Close",     value=float(last_row["Close"]),     step=0.01, format="%.4f", key="f_close")
        f_adj   = fc2.number_input("Adj Close", value=float(last_row["Adj Close"]), step=0.01, format="%.4f", key="f_adj")
        f_vol   = fc3.number_input("Volume",    value=float(last_row["Volume"]),    step=1.0,  format="%.0f", key="f_vol")

        if st.button("🚀 Train LSTM & Forecast", use_container_width=True, key="lstm_btn"):
            with st.spinner(f"Training LSTM on full dataset ({lstm_ep} epochs)… this may take a minute."):
                full_close = df["Close"].copy()
                full_close.iloc[-1] = f_close  # inject user value

                scaler_fc = MinMaxScaler()
                scaled_fc = scaler_fc.fit_transform(full_close.values.reshape(-1, 1))

                X_fc, y_fc = _make_sequences(scaled_fc, lstm_lb)
                X_fc = X_fc.reshape(X_fc.shape[0], lstm_lb, 1)

                fc_model = _build_lstm(lstm_lb)
                fc_model.fit(X_fc, y_fc, epochs=lstm_ep, batch_size=32, verbose=0)

            st.success("✅ LSTM trained! Generating forecast…")

            # ── Iterative forecast ────────────────────────────────────────
            seed = scaled_fc[-lstm_lb:].flatten().tolist()
            future_scaled = []
            for _ in range(forecast_horizon):
                x_in = np.array(seed[-lstm_lb:]).reshape(1, lstm_lb, 1)
                nxt  = float(fc_model.predict(x_in, verbose=0)[0, 0])
                future_scaled.append(nxt)
                seed.append(nxt)

            future_prices = scaler_fc.inverse_transform(
                np.array(future_scaled).reshape(-1, 1)
            ).flatten()

            # ── Date index (business days) ────────────────────────────────
            last_date    = pd.to_datetime(df["Date"].iloc[-1])
            future_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1), periods=forecast_horizon
            )
            forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_prices})

            # ── Chart ─────────────────────────────────────────────────────
            history_tail = df[["Date", "Close"]].tail(lstm_lb)
            fig_fc, ax_fc = plt.subplots(figsize=(14, 5))
            ax_fc.plot(history_tail["Date"], history_tail["Close"],
                       color="#0071E3", linewidth=1.5, label="Historical (last window)")
            ax_fc.plot(forecast_df["Date"], forecast_df["Predicted Close"],
                       color="#FF3B30", linewidth=2, linestyle="--",
                       label=f"LSTM Forecast ({forecast_horizon} days)")
            ax_fc.fill_between(
                forecast_df["Date"],
                forecast_df["Predicted Close"] * 0.97,
                forecast_df["Predicted Close"] * 1.03,
                alpha=0.15, color="#FF3B30", label="±3% band",
            )
            ax_fc.axvline(x=last_date, color="gray", linestyle=":", linewidth=1, label="Today")
            ax_fc.set_xlabel("Date"); ax_fc.set_ylabel("Close Price (USD)")
            ax_fc.set_title(f"Apple Stock – LSTM {forecast_horizon}-Day Forecast")
            ax_fc.legend()
            plt.tight_layout()
            st.pyplot(fig_fc); plt.close(fig_fc)

            # ── Summary metrics ───────────────────────────────────────────
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Day 1 forecast",            f"${future_prices[0]:.2f}")
            sm2.metric(f"Day {forecast_horizon} forecast", f"${future_prices[-1]:.2f}",
                       delta=f"{future_prices[-1] - f_close:+.2f}")
            sm3.metric("Total change vs today",
                       f"{(future_prices[-1] - f_close) / f_close * 100:+.2f}%")

            st.dataframe(forecast_df.set_index("Date").round(4), use_container_width=True)
