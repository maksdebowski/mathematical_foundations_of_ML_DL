import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import config

def analyze_volatility(df_before, df_after):
    vol_before = df_before["Realized_Volatility_Daily"].dropna()
    vol_after = df_after["Realized_Volatility_Daily"].dropna()

    print("\nVolatility Statistics:")
    vol_stats = pd.DataFrame({
        'Before Shock': vol_before.describe(),
        'After Shock': vol_after.describe()
    })

    print(vol_stats[["Before Shock", "After Shock"]].loc[['mean', '50%', 'std', 'min', 'max']])
    return vol_stats

def analyze_log_returns(df_before, df_after):
    returns_before = df_before["Log_Return"].dropna()
    returns_after = df_after["Log_Return"].dropna()

    return_stats = pd.DataFrame({
        'Before Shock': {
            'Mean': returns_before.mean(),
            'Std Dev': returns_before.std(),
            'Skewness': returns_before.skew(),
            'Kurtosis': returns_before.kurtosis()
        },
        'After Shock': {
            'Mean': returns_after.mean(),
            'Std Dev': returns_after.std(),
            'Skewness': returns_after.skew(),
            'Kurtosis': returns_after.kurtosis()
        }
    })

    print("\nLog Return Statistics:")
    print(return_stats)

    shapiro_before = stats.shapiro(returns_before)
    shapiro_after = stats.shapiro(returns_after)
    print(
        f"Shapiro-Wilk Test (Before Shock): Statistic={shapiro_before.statistic:.4f},"
        f"p-value={shapiro_before.pvalue:.4f}")
    print(
        f"Shapiro-Wilk Test (After Shock):  Statistic={shapiro_after.statistic:.4f},"
        f"p-value={shapiro_after.pvalue:.4f}")
    normality_results = {'shapiro_before': shapiro_before,
                         'shapiro_after': shapiro_after}

    return return_stats, normality_results, (returns_before, returns_after)