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

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

## 📸 Dashboard Screenshots

### 🔹 Single Store Forecast (Store 1)
<img src="prophet_forecast_store1.png" width="600"/>

### 🔹 Prophet Components
<table>
  <tr>
    <td><img src="prophet_components_trend_weekly.png" width="400"/></td>
    <td><img src="prophet_components_yearly_weekly.png" width="400"/></td>
  </tr>
</table>

### 🔹 Prophet Full Components + Metrics
<img src="prophet_components_full_metrics.png" width="600"/>

### 🔹 Multi-Store Forecast Comparison
<table>
  <tr>
    <td><img src="multi_store_forecast_prophet.png" width="400"/></td>
    <td><img src="multi_store_forecast_arima.png" width="400"/></td>
  </tr>
</table>

### 🔹 Model Comparison: ARIMA vs Prophet (Store 111)
<table>
  <tr>
    <td><img src="model_comparison_forecast_store111.png" width="400"/></td>
    <td><img src="model_comparison_metrics_store111.png" width="400"/></td>
  </tr>
</table>

