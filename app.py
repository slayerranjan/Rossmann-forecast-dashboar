import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.plot import plot_components_plotly
import plotly.io as pio

# Force redeploy — dummy comment to trigger Streamlit Cloud refresh

# Optional: Plotly renderer (safe default for cloud)
pio.renderers.default = "plotly_mimetype"

st.set_page_config(page_title="Rossmann Sales Forecasting", layout="wide")

# -----------------------------
# Data loading
# -----------------------------
RAW_TRAIN_URL = "https://raw.githubusercontent.com/slayerranjan/Rossmann-forecast-dashboar/main/train.csv"
RAW_STORE_URL = "https://raw.githubusercontent.com/slayerranjan/Rossmann-forecast-dashboar/main/store.csv"

@st.cache_data(show_spinner=False)
def load_data(train_file=None, store_file=None):
    try:
        # Prefer uploaded files if both present
        if train_file is not None and store_file is not None:
            df = pd.read_csv(train_file)
            store_df = pd.read_csv(store_file)
        else:
            # Fallback to GitHub raw URLs (no local file paths)
            df = pd.read_csv(RAW_TRAIN_URL)
            store_df = pd.read_csv(RAW_STORE_URL)
    except Exception as e:
        st.error(f"Failed to load data. Details: {e}")
        return pd.DataFrame()

    # Basic cleaning and merge
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"])
    df = df[df["Open"] == 1].dropna(subset=["Sales", "Customers"])

    df["StateHoliday"] = (
        df["StateHoliday"]
        .astype(str)
        .str.strip()
        .replace({"0": "None", "a": "Public", "b": "Easter", "c": "Christmas", "nan": "None", "": "None"})
    )
    df["SchoolHoliday"] = df["SchoolHoliday"].astype("int64")

    df = pd.merge(df, store_df, on="Store", how="left")
    return df

# -----------------------------
# Helpers
# -----------------------------
def get_series(df: pd.DataFrame, store_id: int) -> pd.Series:
    ts = df[df["Store"] == store_id].groupby("Date")["Sales"].sum().asfreq("D").fillna(0)
    return ts

def safe_mape(y_true, y_pred):
    mask = np.abs(y_true) >= 1e-2
    y_true_safe = y_true[mask]
    y_pred_safe = y_pred[mask]
    if len(y_true_safe) == 0:
        return np.nan
    return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe))

def plot_arima(train, test, forecast, store_id):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train, label="Train", color="#1f77b4", alpha=0.7)
    ax.plot(test.index, test, label="Test (Actual)", color="black", linewidth=1.5)
    ax.plot(forecast.index, forecast, label="ARIMA Forecast", color="red", linewidth=1.5)
    ax.set_title(f"Store {store_id} Sales Forecast (ARIMA)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True, alpha=0.2)
    ax.legend()
    st.pyplot(fig)

def plot_prophet_all(model, forecast_df, store_id):
    fig1 = model.plot(forecast_df.reset_index())
    ax1 = fig1.gca()
    ax1.set_title(f"Store {store_id} Sales Forecast (Prophet)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales")
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast_df.reset_index())
    st.pyplot(fig2)

    fig3 = plot_components_plotly(model, forecast_df.reset_index())
    st.plotly_chart(fig3, use_container_width=True)

def train_arima(train, test, order):
    model = ARIMA(train, order=order).fit()
    fc = model.forecast(steps=len(test))
    fc.index = test.index
    rmse = np.sqrt(mean_squared_error(test.values, fc.values))
    mae = mean_absolute_error(test.values, fc.values)
    mape = safe_mape(test.values, fc.values)
    return fc, rmse, mae, mape

def train_prophet(train, test, horizon):
    prophet_train = train.reset_index().rename(columns={"Date": "ds", "Sales": "y"})
    model = Prophet()
    model.fit(prophet_train)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast_df = model.predict(future).set_index("ds")
    fc = forecast_df.loc[test.index, "yhat"]
    rmse = np.sqrt(mean_squared_error(test.values, fc.values))
    mae = mean_absolute_error(test.values, fc.values)
    mape = safe_mape(test.values, fc.values)
    return model, forecast_df, fc, rmse, mae, mape

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controls")
data_mode = st.sidebar.selectbox("Data source", ["Default Rossmann CSV", "Upload CSVs"])

if data_mode == "Upload CSVs":
    up_train = st.sidebar.file_uploader("Upload train.csv", type=["csv"])
    up_store = st.sidebar.file_uploader("Upload store.csv", type=["csv"])
    df = load_data(up_train if up_train is not None else None,
                   up_store if up_store is not None else None)
else:
    df = load_data()

if df.empty:
    st.error("Data failed to load. Upload CSVs or ensure GitHub raw URLs are accessible.")
    st.stop()

store_ids = sorted(df["Store"].unique().tolist())
model_type = st.sidebar.selectbox("Model", ["ARIMA", "Prophet"], index=1)
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=60, value=30, step=1)
p = st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=0, step=1)
q = st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1, step=1)
st.sidebar.markdown("---")

# -----------------------------
# Main
# -----------------------------
st.title("Rossmann Sales Forecasting Dashboard")
tab1, tab2, tab3 = st.tabs(["Single Store Forecast", "Multi-Store Comparison", "Model Comparison"])

# Tab 1: Single store
with tab1:
    st.write("Forecast store sales using ARIMA or Prophet, view visualizations, and compare metrics.")
    store_id = st.selectbox("Select store", store_ids, index=store_ids.index(1) if 1 in store_ids else 0)
    run_button = st.button("Run forecast")

    if run_button:
        ts = get_series(df, store_id)
        if len(ts) <= horizon:
            st.error("Time series too short for selected horizon.")
        else:
            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            if model_type == "ARIMA":
                with st.spinner("Training ARIMA..."):
                    try:
                        arima_fc, rmse, mae, mape = train_arima(train, test, (p, d, q))
                        plot_arima(train, test, arima_fc, store_id)
                        st.subheader("ARIMA metrics")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("RMSE", f"{rmse:.2f}")
                        c2.metric("MAE", f"{mae:.2f}")
                        c3.metric("MAPE", f"{mape*100:.2f} %")
                    except Exception as e:
                        st.error(f"ARIMA error: {e}")
            else:
                with st.spinner("Training Prophet..."):
                    try:
                        model, forecast_df, prophet_fc, rmse, mae, mape = train_prophet(train, test, horizon)
                        plot_prophet_all(model, forecast_df, store_id)
                        st.subheader("Prophet metrics")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("RMSE", f"{rmse:.2f}")
                        c2.metric("MAE", f"{mae:.2f}")
                        c3.metric("MAPE", f"{mape*100:.2f} %")
                    except Exception as e:
                        st.error(f"Prophet error: {e}")
    else:
        st.info("Configure options and click Run forecast.")

# Tab 2: Multi-store comparison
with tab2:
    st.header("Multi-Store Forecast Comparison")
    default_selection = store_ids[:5] if len(store_ids) >= 5 else store_ids
    selected_stores = st.multiselect("Select multiple stores", store_ids, default=default_selection)
    model_choice = st.radio("Model for all stores", ["ARIMA", "Prophet"], horizontal=True)
    compare_button = st.button("Compare forecasts")

    if compare_button:
        fig, ax = plt.subplots(figsize=(12, 6))
        for sid in selected_stores:
            ts = get_series(df, sid)
            if len(ts) <= horizon:
                st.warning(f"Store {sid} skipped: not enough data.")
                continue
            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]
            try:
                if model_choice == "ARIMA":
                    fc, _, _, _ = train_arima(train, test, (p, d, q))
                    ax.plot(fc.index, fc, label=f"Store {sid}")
                else:
                    _, forecast_df, fc, _, _, _ = train_prophet(train, test, horizon)
                    ax.plot(fc.index, fc, label=f"Store {sid}")
            except Exception:
                st.warning(f"{model_choice} failed for Store {sid}")

        ax.set_title(f"{model_choice} Forecast Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Select stores and click Compare forecasts.")

# Tab 3: Model comparison
with tab3:
    st.header("Model Comparison: ARIMA vs Prophet (Single Store)")
    cmp_store_id = st.selectbox("Select store for comparison", store_ids, index=store_ids.index(1) if 1 in store_ids else 0)
    cmp_button = st.button("Compare ARIMA vs Prophet")

    if cmp_button:
        ts = get_series(df, cmp_store_id)
        if len(ts) <= horizon:
            st.error("Time series too short for selected horizon.")
        else:
            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            with st.spinner("Training ARIMA and Prophet..."):
                try:
                    arima_fc, arima_rmse, arima_mae, arima_mape = train_arima(train, test, (p, d, q))
                    _, forecast_df, prophet_fc, prophet_rmse, prophet_mae, prophet_mape = train_prophet(train, test, horizon)

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(test.index, test, label="Test (Actual)", color="black", linewidth=1.5)
                    ax.plot(arima_fc.index, arima_fc, label="ARIMA Forecast", color="red", linewidth=1.5)
                    ax.plot(prophet_fc.index, prophet_fc, label="Prophet Forecast", color="#1f77b4", linewidth=1.5)
                    ax.set_title(f"Store {cmp_store_id}: ARIMA vs Prophet")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Sales")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    st.subheader("Metrics")
                    cA, cB, cC, cD, cE, cF = st.columns(6)
                    cA.metric("ARIMA RMSE", f"{arima_rmse:.2f}")
                    cB.metric("ARIMA MAE", f"{arima_mae:.2f}")
                    cC.metric("ARIMA MAPE", f"{arima_mape*100:.2f} %")
                    cD.metric("Prophet RMSE", f"{prophet_rmse:.2f}")
                    cE.metric("Prophet MAE", f"{prophet_mae:.2f}")
                    cF.metric("Prophet MAPE", f"{prophet_mape*100:.2f} %")

                    result_df = pd.DataFrame({
                        "Date": test.index,
                        "Actual": test.values,
                        "ARIMA_Forecast": arima_fc.values,
                        "Prophet_Forecast": prophet_fc.values
                    })
                    metrics_df = pd.DataFrame({
                        "Model": ["ARIMA", "Prophet"],
                        "RMSE": [arima_rmse, prophet_rmse],
                        "MAE": [arima_mae, prophet_mae],
                        "MAPE": [arima_mape*100, prophet_mape*100]
                    })

                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    csv_buffer.write("\n\n")
                    metrics_df.to_csv(csv_buffer, index=False)

                    st.download_button(
                        label="Download forecasts + metrics (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"store_{cmp_store_id}_arima_vs_prophet.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Comparison error: {e}")
    else:
        st.info("Select store and click Compare ARIMA vs Prophet.")
