# Rossmann Forecast Dashboard

A modular Streamlit dashboard for forecasting retail sales using ARIMA and Prophet. Built for model comparison, multi-store analysis, and recruiter-grade visibility.

## 🔍 Features
- Single-store forecasting with ARIMA and Prophet
- Multi-store forecast comparison
- ARIMA vs Prophet side-by-side benchmarking
- RMSE, MAE, MAPE metrics
- CSV export of forecasts and metrics

## 📦 Tech Stack
- Python, Streamlit
- ARIMA (statsmodels), Prophet (Facebook)
- Matplotlib, Plotly
- scikit-learn, pandas, numpy

## 📸 Screenshots

### 🔹 Single Store Forecast
<img src="screenshots/prophet_forecast_store1git.png" width="600"/>

### 🔹 Prophet components
<img src="screenshots/prophet_components_trend_weekly.png" width="400"/>
<img src="screenshots/prophet_components_yearly_weekly.png" width="400"/>

### 🔹 Prophet full components + metrics
<img src="screenshots/prophet_components_full_metrics.png" width="600"/>

### 🔹 Multi-store forecast comparison
<img src="screenshots/multi_store_forecast_prophet.png" width="400"/>
<img src="screenshots/multi_store_forecast_arima.png" width="400"/>

### 🔹 Model comparison: ARIMA vs Prophet (Store 111)
<img src="screenshots/model_comparison_forecast_store111.png" width="400"/>
<img src="screenshots/model_comparison_metrics_store111.png" width="400"/>



## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
