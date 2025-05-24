import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


def decompose_time_series(series: pd.Series, period: int, output_dir: str = "reports/decomposition"):
    """
    Decompose a time series into trend, seasonal, residual components and save CSVs and PNGs.
    Returns JSON strings of each component.
    """
    os.makedirs(output_dir, exist_ok=True)
    series.index = pd.to_datetime(series.index)
    # drop missing values to avoid decomposition errors
    series = series.dropna()
    decomposed = seasonal_decompose(series, model="additive", period=period)
    components = {
        "trend": decomposed.trend,
        "seasonal": decomposed.seasonal,
        "resid": decomposed.resid
    }
    result_json = {}
    for comp_name, comp in components.items():
        comp_clean = comp.dropna()
        comp_clean.to_csv(os.path.join(output_dir, f"{comp_name}.csv"))
        plt.figure()
        comp_clean.plot(title=comp_name)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{comp_name}.png"))
        plt.close()
        result_json[comp_name] = comp_clean.to_json()
    return result_json


def forecast_arima(series: pd.Series, order: tuple = (1, 1, 1), steps: int = 10):
    """
    Fit ARIMA model and forecast specified number of steps. Returns forecast JSON.
    """
    series.index = pd.to_datetime(series.index)
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast.to_json() 