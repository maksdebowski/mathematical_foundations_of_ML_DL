import pandas as pd
import numpy as np
import config

def run_garch_simulation(model_results, df_after_shock, last_actual_price,
                         n_sims=config.N_SIMS, annualization_factor=config.ANNUALIZATION_FACTOR):
    print(f"\n--- Running GARCH Counterfactual Simulation ({n_sims} paths) ---")
    n_steps = len(df_after_shock)
    print(f"Simulation horizon: {n_steps} steps (trading days after shock)")
    print(f"Starting price for simulation: {last_actual_price:.2f}")
    forecast = model_results.forecast(horizon = n_steps, method = 'simulation', simulations = n_sims)

    sim_mean_scaled = forecast.simulations.values[0].T
    sim_variance_unscaled = forecast.simulations.residual_variances[0].T

    sim_log_returns  = sim_mean_scaled / 100.0

    sim_prices = np.zeros((n_steps, n_sims))
    current_prices = np.full(n_sims, last_actual_price)

    for _ in range(n_steps):
        current_prices = current_prices * np.exp(sim_log_returns[_, :])
        sim_prices[_, :] = current_prices
    sim_prices_df = pd.DataFrame(sim_prices, index=df_after_shock.index,
                                 columns=[f'Sim_{i+1}' for i in range(n_sims)])
    sim_prices_stats = pd.DataFrame({
        'Median': sim_prices_df.median(axis=1),
        'Mean': sim_prices_df.mean(axis=1),
        'Quantile_05': sim_prices_df.quantile(0.05, axis=1),
        'Quantile_95': sim_prices_df.quantile(0.95, axis=1)
    }, index=df_after_shock.index)

    sim_volatility_annualized = np.sqrt(sim_variance_unscaled * annualization_factor)

    sim_vol_df = pd.DataFrame(sim_volatility_annualized, index=df_after_shock.index,
                              columns=[f'Sim_{i+1}' for i in range(n_sims)])
    sim_vol_stats = pd.DataFrame({
        'Median': sim_vol_df.median(axis=1),
        'Mean': sim_vol_df.mean(axis=1),
        'Quantile_05': sim_vol_df.quantile(0.05, axis=1),
        'Quantile_95': sim_vol_df.quantile(0.95, axis=1)
    }, index=df_after_shock.index)

    print("GARCH simulation processing complete.")
    return sim_prices_stats, sim_vol_stats

def run_dynamic_naive_simulation(df, df_after_shock, last_actual_price,
                                 n_sims = config.N_SIMS, window = config.WINDOW,
                                 annualization_factor = config.ANNUALIZATION_FACTOR):
    if 'Log_Return' in df.columns:
        print("Calculating 'Realized_Volatility_Daily' for dynamic naive simulation.")
        df['Realized_Volatility_Daily'] = df['Log_Return'].rolling(window=window).std() * np.sqrt(annualization_factor) * 100
    else:
        print("Error: 'Realized_Volatility_Daily' missing and cannot be calculated.")
        return None, None
    if 'Rolling_Mean_Log_Return' not in df.columns:
        print("Calculating 'Rolling_Mean_Log_Return' for dynamic naive simulation.")
        df['Rolling_Mean_Log_Return'] = df['Log_Return'].rolling(window=window).mean()

    simulation_dates = df_after_shock.index
    n_steps = len(simulation_dates)
    if n_steps == 0:
        print("No post-shock period to simulate.")
        return None, None

    print(f"Dynamic simulation horizon: {n_steps} days")
    print(f"Number of sims: {n_sims}")
    print(f"Starting price: {last_actual_price:.2f}")

    sim_prices_dyn = np.zeros((n_steps, n_sims))
    current_prices_dyn = np.full(n_sims, last_actual_price)
    used_hist_vol = np.zeros(n_steps)
    used_hist_mean_ret = np.zeros(n_steps)
    fallback_cnt = 0

    last_valid_vol = df['Realized_Volatility_Daily'].dropna().iloc[-1] if not df['Realized_Volatility_Daily'].dropna().empty else 20.0
    last_valid_mean = df['Rolling_Mean_Log_Return'].dropna().iloc[-1] if not df['Rolling_Mean_Log_Return'].dropna().empty else 0.0

    for _, current_sim_date in enumerate(simulation_dates):
        target_hist_date = current_sim_date - pd.DateOffset(years = 1)
        hist_vol_percent = last_valid_vol
        hist_mean_ret = last_valid_mean

        try:
            hist_index_loc = df.index.get_indexer([target_hist_date], method = 'ffill')[0]

            if hist_index_loc != -1:
                actual_hist_date = df.index[hist_index_loc]
                vol_val = df.loc[actual_hist_date, 'Realized_Volatility_Daily']
                mean_val = df.loc[actual_hist_date, 'Rolling_Mean_Log_Return']

                is_fallback = False
                if pd.notna(vol_val):
                    hist_vol_percent = vol_val
                    last_valid_vol = vol_val
                else:
                    is_fallback = True

                if is_fallback:
                    fallback_cnt += 1
            else:
                fallback_cnt += 1
        except Exception as e:
            print(e)
            fallback_cnt += 1
        used_hist_vol[_] = hist_vol_percent
        used_hist_mean_ret[_] = hist_mean_ret

        daily_std_dev = max((hist_vol_percent / 100.0) / np.sqrt(annualization_factor), 1e-6)
        sim_daily_returns = np.random.normal(loc=hist_mean_ret, scale=daily_std_dev, size=n_sims)

        current_prices_dyn = current_prices_dyn * np.exp(sim_daily_returns)
        sim_prices_dyn[_, :] = current_prices_dyn
        print(f"Dynamic simulation finished. Fallback values used {fallback_cnt} times.")

        sim_prices_df_dyn = pd.DataFrame(sim_prices_dyn, index=simulation_dates,
                                         columns=[f'Sim_{i + 1}' for i in range(n_sims)])
        sim_prices_stats_dyn = pd.DataFrame({
            'Median': sim_prices_df_dyn.median(axis=1),
            'Mean': sim_prices_df_dyn.mean(axis=1),
            'Quantile_05': sim_prices_df_dyn.quantile(0.05, axis=1),
            'Quantile_95': sim_prices_df_dyn.quantile(0.95, axis=1)
        }, index=simulation_dates)

        used_params_df = pd.DataFrame({
            'Used_Historical_Volatility': used_hist_vol,
            'Used_Historical_Mean_Return': used_hist_mean_ret
        }, index=simulation_dates)

        return sim_prices_stats_dyn, used_params_df
def run_fixed_naive_simulation(df, df_after_shock, last_actual_price,
                               shock_date = config.SHOCK_DATE, n_sims = config.N_SIMS,
                               window = config.WINDOW, annualization_factor = config.ANNUALIZATION_FACTOR):
    try:
        hist_end_date = pd.to_datetime(shock_date) - pd.Timedelta(days = 1)
        hist_start_date = pd.to_datetime(shock_date) - pd.DateOffset(years = 1)
        print(f"Calculating fixed parameters from: {hist_start_date.date()} to {hist_end_date.date()}")
    except Exception as crowley:
        print(crowley)
        return None, None
    historical_mask = (df.index >= hist_start_date) & (df.index <= hist_end_date)
    df_hist_period = df[historical_mask].copy()

    if df_hist_period.empty:
        print("No data found in the historical period.")
        return None, None

    if 'Realized_Volatility_Daily' not in df_hist_period.columns:
         df_hist_period['Realized_Volatility_Daily'] = df_hist_period['Log_Return'].rolling(window=window).std() * np.sqrt(annualization_factor) * 100
    if 'Rolling_Mean_Log_Return' not in df_hist_period.columns:
         df_hist_period['Rolling_Mean_Log_Return'] = df_hist_period['Log_Return'].rolling(window=window).mean()

    valid_means = df_hist_period['Rolling_Mean_Log_Return'].dropna()
    valid_vols = df_hist_period['Realized_Volatility_Daily'].dropna()

    if valid_means.empty or valid_vols.empty:
        print("No valid mean return or volatility data in the historical period.")
        return None, None

    fixed_hist_mean_ret = valid_means.mean()
    fixed_hist_vol_percent = valid_vols.mean()
    fixed_daily_std_dev = max((fixed_hist_vol_percent / 100.0) / np.sqrt(annualization_factor), 1e-6)

    fixed_params = {
        'mean_log_return': fixed_hist_mean_ret,
        'annualized_vol_percent': fixed_hist_vol_percent,
        'daily_std_dev': fixed_daily_std_dev
    }
    print("\nFixed Naive parameters:")
    for key, val in fixed_params.items():
        print(f"{key}: {val:.6f}")
    simulation_dates = df_after_shock.index
    n_steps = len(simulation_dates)
    if n_steps == 0:
        print("No data found in the historical period.")
        return None, None
    print(f"\nFixed naive simulation horizon: {n_steps} days.")
    print(f"Number of simulations: {n_sims}")
    print(f"Starting price: {last_actual_price:.2f}")

    sim_prices_fixed = np.zeros((n_steps, n_sims))
    current_prices_fixed = np.full(n_sims, last_actual_price)

    for _ in range(n_steps):
        sim_daily_returns = np.random.normal(loc = fixed_hist_mean_ret,
                                             scale = fixed_daily_std_dev,
                                             size = n_sims)
        current_prices_fixed = current_prices_fixed * np.exp(sim_daily_returns)
        sim_prices_fixed[_, :] = current_prices_fixed

        print("Fixed parameter naive simulation finished.")

        sim_prices_df_fixed = pd.DataFrame(sim_prices_fixed, index=simulation_dates,
                                           columns=[f'Sim_{i + 1}' for i in range(n_sims)])
        sim_prices_stats_fixed = pd.DataFrame({
            'Median': sim_prices_df_fixed.median(axis=1),
            'Mean': sim_prices_df_fixed.mean(axis=1),
            'Quantile_05': sim_prices_df_fixed.quantile(0.05, axis=1),
            'Quantile_95': sim_prices_df_fixed.quantile(0.95, axis=1)
        }, index=simulation_dates)

        return sim_prices_stats_fixed, fixed_params