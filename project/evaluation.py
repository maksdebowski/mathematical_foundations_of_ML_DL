import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def qlike_loss(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    epsilon = 1e-10

    mask = (
        ~np.isnan(actual)
        & ~np.isnan(predicted)
        & (actual > epsilon)
        & (predicted > epsilon)
    )
    if np.sum(mask) == 0:
        return np.nan

    actual = actual[mask]
    predicted = predicted[mask]

    loss = actual / predicted - np.log(actual / predicted) - 1
    return np.mean(loss)


def calculate_price_metrics(actual_prices, median_sim_prices):
    if not isinstance(actual_prices, pd.Series):
        actual_prices = pd.Series(actual_prices)
    if not isinstance(median_sim_prices, pd.Series):
        median_sim_prices = pd.Series(median_sim_prices)

    combined = pd.concat(
        [actual_prices.rename("Actual"), median_sim_prices.rename("Median_Sim")], axis=1
    ).dropna()

    if combined.empty:
        print(
            "Warning: No overlapping non-NaN data between actual and simulated prices."
        )
        return {"RMSE_price": np.nan, "MAE_price": np.nan}

    actual = combined["Actual"]
    median_sim = combined["Median_Sim"]

    rmse = np.sqrt(mean_squared_error(actual, median_sim))
    mae = mean_absolute_error(actual, median_sim)
    metrics = {"RMSE_price": rmse, "MAE_price": mae}
    return metrics


def calculate_volatility_metrics(actual_volatility, predicted_volatility):
    if not isinstance(actual_volatility, pd.Series):
        actual_volatility = pd.Series(actual_volatility)
    if not isinstance(predicted_volatility, pd.Series):
        predicted_volatility = pd.Series(predicted_volatility)

    combined = pd.concat(
        [
            actual_volatility.rename("Actual_Vol"),
            predicted_volatility.rename("Predicted_Vol"),
        ],
        axis=1,
    ).dropna()

    if combined.empty:
        print(
            "Warning: No overlapping non-NaN data between actual and predicted\
                volatility."
        )
        return {"RMSE_vol": np.nan, "MAE_vol": np.nan, "QLIKE_var": np.nan}

    actual_vol = combined["Actual_Vol"]
    predicted_vol = combined["Predicted_Vol"]

    rmse_vol = np.sqrt(mean_squared_error(actual_vol, predicted_vol))
    mae_vol = mean_absolute_error(actual_vol, predicted_vol)
    actual_var = np.square(actual_vol[actual_vol > 0])
    predicted_var = np.square(predicted_vol[predicted_vol > 0])
    common_index = actual_var.index.intersection(predicted_var.index)
    if common_index.empty:
        qlike_var = np.nan
    else:
        qlike_var = qlike_loss(
            actual_var.loc[common_index], predicted_var.loc[common_index]
        )

    metrics = {"RMSE_vol": rmse_vol, "MAE_vol": mae_vol, "QLIKE_var": qlike_var}
    return metrics


def display_combined_metrics(price_metrics, vol_metrics, model_name="Model"):
    print(f"\n--- Evaluation Metrics: {model_name} ---")

    data = []
    if price_metrics:
        for key, value in price_metrics.items():
            metric_name, metric_type = key.split("_")
            data.append(
                {
                    "Metric": metric_name,
                    "Type": metric_type.capitalize(),
                    "Value": value,
                }
            )
    if vol_metrics:
        for key, value in vol_metrics.items():
            metric_name, metric_type = key.split("_")
            if metric_type == "var":
                metric_type = "Variance (QLIKE)"
            else:
                metric_type = metric_type.capitalize()
            data.append({"Metric": metric_name, "Type": metric_type, "Value": value})

    if not data:
        print("No metrics provided.")
        return

    metrics_df = pd.DataFrame(data)

    try:
        if pd.api.types.is_numeric_dtype(metrics_df["Value"]):
            metrics_df["Value"] = metrics_df["Value"].map("{:,.4f}".format)
        else:
            metrics_df["Value"] = metrics_df["Value"].astype(str)
    except (TypeError, ValueError) as e:
        print(f"Could not format metrics values: {e}")
        pass  #

    print(metrics_df.to_string(index=False))
