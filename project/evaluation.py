import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def qlike_loss(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    epsilon = 1e-10 # Avoid division by zero or log(0)
    predicted = np.maximum(predicted, epsilon)
    actual = np.maximum(actual, epsilon) # Also ensure actual is positive for log
    # Ensure alignment and handle NaNs
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if np.sum(mask) == 0:
        return np.nan # No valid pairs to compare
    actual = actual[mask]
    predicted = predicted[mask]
    if np.any(predicted <= 0) or np.any(actual <= 0):
         print("Warning: Non-positive values encountered in QLIKE calculation after masking.")
         return np.nan
    return np.mean(actual / predicted - np.log(actual / predicted) - 1)


def calculate_simulation_metrics(actual_prices, median_sim_prices):
    if not isinstance(actual_prices, pd.Series):
        actual_prices = pd.Series(actual_prices)
    if not isinstance(median_sim_prices, pd.Series):
        median_sim_prices = pd.Series(median_sim_prices)

    combined = pd.concat([actual_prices.rename('Actual'),
                          median_sim_prices.rename('Median_Sim')], axis=1).dropna()

    if combined.empty:
        print("Warning: No overlapping non-NaN data between actual and simulated prices.")
        metrics = {'RMSE': np.nan, 'MAE': np.nan, 'QLIKE': np.nan}
        if sim_prices_stats is not None:
             metrics['Custom_Cost'] = np.nan
        return metrics

    actual = combined['Actual']
    median_sim = combined['Median_Sim']

    rmse = np.sqrt(mean_squared_error(actual, median_sim))
    mae = mean_absolute_error(actual, median_sim)
    qlike = qlike_loss(actual, median_sim)

    metrics = {'RMSE': rmse, 'MAE': mae, 'QLIKE': qlike}
    return metrics

def display_metrics(metrics, model_name = "Model"):
    print(f"\n--- Evaluation Metrics: {model_name} ---")
    if not metrics:
        print("No metrics provided.")
        return
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])

    try:
        metrics_df['Value'] = metrics_df['Value'].map('{:,.4f}'.format)
    except (TypeError, ValueError):
        pass  # Keep original format if mapping fails
    print(metrics_df.to_string(index=False))
if __name__ == "__main__":
    actual = pd.Series([100, 102, 101, 103, 105, 104])
    median_sim = pd.Series([100, 101, 101.5, 102, 104, 103])

    print("Testing QLIKE:")
    print(f"QLIKE: {qlike_loss(actual, median_sim):.4f}")


    print("\nTesting Metrics Calculation:")
    metrics = calculate_simulation_metrics(actual, median_sim)
    display_metrics(metrics, "Test Calculation")

    print("\nTesting with NaNs:")
    actual_nan = pd.Series([100, 102, np.nan, 103, 105, 104])
    median_sim_nan = pd.Series([100, np.nan, 101.5, 102, 104, 103])
    metrics_nan = calculate_simulation_metrics(actual_nan, median_sim_nan)
    display_metrics(metrics_nan, "Test Calculation with NaNs")
