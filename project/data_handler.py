import yfinance as yf
import pandas as pd
import numpy as np
import config

def fetch_stock_data(ticker = config.TICKER, start_date = config.START_DATE, end_date = config.EMD_DATE, interval = config.INTERVAL):
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    df = yf.download(tickers = ticker,
                     start = start_date,
                     interval = interval,
                     actions = True,
                     auto_adjust = False)

    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) > 1 and ticker in df.columns.levels[1]:
            df = df.xs(ticker, level=1, axis=1)
        elif len(df.columns.levels) == 1:
            df.columns = df.columns.droplevel(0)

        print(f"Initial data fetched. Shape: {df.shape}")
    return df
def preprocess_data(df, window=config.WINDOW, annualization_factor=config.ANNUALIZATION_FACTOR):
    print("Preprocessing data...")
    df.index = pd.to_datetime(df.index)

    columns_to_check = [col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                       if col in df.columns]
    initial_rows = len(df)
    df.dropna(subset=columns_to_check, inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows with missing essential price/volume data.")

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    if price_col not in df.columns:
        raise ValueError(f"Neither 'Adj Close' nor 'Close' found in DataFrame columns.")

    df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
    df.dropna(subset=['Log_Return'], inplace=True)
    print(f"Calculated Log Returns using '{price_col}'.")

    df['Realized_Volatility_Daily'] = df['Log_Return'].rolling(window=window).std() * np.sqrt(annualization_factor) * 100
    print(f"Calculated Realized Volatility (annualized %) with window {window}.")
    print(f"Preprocessing complete. Final data shape: {df.shape}")
    return df


def split_data_by_shock(df, shock_date = config.SHOCK_DATE):
    shock_datetime = pd.to_datetime(shock_date)
    df_before_shock = df[df.index < shock_datetime].copy()
    df_after_shock = df[df.index >= shock_datetime].copy()
    print(f"Data split: {len(df_before_shock)} points before {shock_date},"
          f"{len(df_after_shock)} points on/after.")
    return df_before_shock, df_after_shock
