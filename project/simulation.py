import pandas as pd
import numpy as np
import config
import modeling


def run_one_day_naive_forecast(
    df, # Pełny dataframe z rzeczywistymi danymi
    df_after_shock, # Tylko daty/struktura dla okresu prognozy
    price_col="Adj Close",
    vol_col="Realized_Volatility_Daily",
):
    """
    Uruchamia prostą prognozę naiwną na 1 dzień.
    Przewiduje cenę(t) = cena(t-1) i zmienność(t) = zmienność(t-1).
    """
    print("\n--- [DEBUG] Uruchamianie prognozy naiwnej 1-dniowej ---") # Dodano [DEBUG]
    forecast_dates = df_after_shock.index
    n_steps = len(forecast_dates)
    if n_steps == 0:
        print("[DEBUG] Brak okresu po szoku dla prognozy naiwnej 1-dniowej. Zwracam None.")
        return None, None

    print(f"[DEBUG] Liczba kroków prognozy: {n_steps}")
    print(f"[DEBUG] Pierwsza data prognozy: {forecast_dates[0].date()}, Ostatnia: {forecast_dates[-1].date()}")
    print(f"[DEBUG] Zakres dat w pełnym df: {df.index.min().date()} do {df.index.max().date()}")


    predicted_prices = []
    predicted_vols = []
    actual_dates_used = [] # Śledź daty, dla których faktycznie mogliśmy prognozować

    # Znajdź datę tuż przed pierwszą datą prognozy
    first_forecast_date = forecast_dates[0]
    try:
        day_before_first = first_forecast_date - pd.Timedelta(days=1)
        print(f"[DEBUG] Szukam daty startowej <= {day_before_first.date()}")
        last_actual_date_loc = df.index.get_indexer([day_before_first], method='ffill')[0]
        print(f"[DEBUG] Znaleziona lokalizacja indeksu startowego: {last_actual_date_loc}")

        if last_actual_date_loc == -1:
             print(f"[DEBUG] BŁĄD: Nie można znaleźć danych przed pierwszą datą prognozy ({first_forecast_date.date()}). Zwracam None.")
             raise IndexError(f"Nie można znaleźć danych przed pierwszą datą prognozy ({first_forecast_date.date()}).")

        last_actual_date = df.index[last_actual_date_loc]
        print(f"[DEBUG] Znaleziona data startowa (ostatni dzień przed prognozą): {last_actual_date.date()}")
        # Sprawdźmy dane dla tej daty startowej
        start_price = df.loc[last_actual_date, price_col]
        start_vol = df.loc[last_actual_date, vol_col]
        print(f"[DEBUG] Dane dla daty startowej - Cena: {start_price}, Zmienność: {start_vol}")
        if pd.isna(start_price):
             print(f"[DEBUG] OSTRZEŻENIE: Cena dla daty startowej {last_actual_date.date()} jest NaN!")


    except (KeyError, IndexError) as e:
        print(f"[DEBUG] BŁĄD przy znajdowaniu punktu startowego: {e}. Zwracam None.")
        return None, None
    except Exception as e:
        print(f"[DEBUG] NIESPODZIEWANY BŁĄD przy znajdowaniu punktu startowego: {e}. Zwracam None.")
        return None, None


    # Pętla przez daty, dla których potrzebujemy prognozy
    print("[DEBUG] Rozpoczynam pętlę prognozowania...")
    skipped_count = 0
    for i, current_forecast_date in enumerate(forecast_dates):
        # print(f"\n[DEBUG] Krok {i+1}/{n_steps}: Prognoza dla: {current_forecast_date.date()}") # Można odkomentować dla bardzo szczegółowego logu
        day_before = current_forecast_date - pd.Timedelta(days=1)
        # print(f"[DEBUG] Szukam rzeczywistego dnia <= {day_before.date()}")

        try:
            actual_prev_date_loc = df.index.get_indexer([day_before], method='ffill')[0]
            # print(f"[DEBUG] Lokalizacja indeksu poprzedniego dnia: {actual_prev_date_loc}")

            if actual_prev_date_loc == -1:
                 # print(f"[DEBUG] OSTRZEŻENIE: Nie można znaleźć indeksu poprzedniego dnia dla {current_forecast_date.date()}. Pomijam.")
                 skipped_count += 1
                 continue

            actual_prev_date = df.index[actual_prev_date_loc]
            # print(f"[DEBUG] Używam danych z rzeczywistej poprzedniej daty: {actual_prev_date.date()}")

            prev_price = df.loc[actual_prev_date, price_col]
            prev_vol = df.loc[actual_prev_date, vol_col]
            # print(f"[DEBUG] Poprzednia Cena: {prev_price}, Poprzednia Zmienność: {prev_vol}")

            if pd.isna(prev_price):
                # print(f"[DEBUG] OSTRZEŻENIE: Poprzednia cena ({actual_prev_date.date()}) jest NaN. Pomijam prognozę dla {current_forecast_date.date()}.")
                skipped_count += 1
                continue

            # Jeśli doszliśmy tutaj, dane wydają się OK dla tego kroku
            # print(f"[DEBUG] Dodaję dane dla {current_forecast_date.date()}")
            predicted_prices.append(prev_price)
            predicted_vols.append(prev_vol) # Zapisz nawet jeśli zmienność jest NaN
            actual_dates_used.append(current_forecast_date)

        except KeyError as ke:
             print(f"[DEBUG] BŁĄD Klucza podczas dostępu do danych dla {actual_prev_date.date() if 'actual_prev_date' in locals() else 'nieznanej daty'}: {ke}. Pomijam.")
             skipped_count += 1
             continue
        except Exception as e:
             print(f"[DEBUG] NIESPODZIEWANY BŁĄD dla daty prognozy {current_forecast_date.date()}: {e}")
             skipped_count += 1
             continue

    print(f"[DEBUG] Pętla zakończona. Liczba pomyślnie przetworzonych dat: {len(actual_dates_used)}. Liczba pominiętych kroków: {skipped_count}")

    if not actual_dates_used:
        print("[DEBUG] BŁĄD: Lista 'actual_dates_used' jest pusta po pętli. Zwracam None.")
        return None, None

    # Utwórz DataFrames z wynikami
    print("[DEBUG] Tworzenie DataFrame'ów z wynikami...")
    try:
        forecast_price_df = pd.DataFrame(
            {"Predicted_Price": predicted_prices}, index=pd.Index(actual_dates_used, name="Date")
        )
        forecast_vol_df = pd.DataFrame(
            {"Predicted_Volatility": predicted_vols}, index=pd.Index(actual_dates_used, name="Date")
        )
        print(f"[DEBUG] Prognoza naiwna 1-dniowa zakończona pomyślnie dla {len(forecast_price_df)} kroków.")
        # print("[DEBUG] Przykładowa prognoza cen naiwnych:\n", forecast_price_df.head())
        # print("[DEBUG] Przykładowa prognoza zmienności naiwnych:\n", forecast_vol_df.head())
        return forecast_price_df, forecast_vol_df
    except Exception as e:
        print(f"[DEBUG] BŁĄD podczas tworzenia DataFrame'ów wynikowych: {e}")
        return None, None

def run_garch_simulation(
    model_results,
    df_after_shock,
    last_actual_price,
    n_sims=config.N_SIMS,
    annualization_factor=config.ANNUALIZATION_FACTOR,
):
    print(f"\n--- Running GARCH Counterfactual Simulation ({n_sims} paths) ---")
    n_steps = len(df_after_shock)
    print(f"Simulation horizon: {n_steps} steps (trading days after shock)")
    print(f"Starting price for simulation: {last_actual_price:.2f}")
    forecast = model_results.forecast(
        horizon=n_steps, method="simulation", simulations=n_sims
    )

    sim_mean_scaled = forecast.simulations.values[0].T
    sim_variance_unscaled = forecast.simulations.residual_variances[0].T

    sim_log_returns = sim_mean_scaled / 100.0

    sim_prices = np.zeros((n_steps, n_sims))
    current_prices = np.full(n_sims, last_actual_price)

    for _ in range(n_steps):
        current_prices = current_prices * np.exp(sim_log_returns[_, :])
        sim_prices[_, :] = current_prices
    sim_prices_df = pd.DataFrame(
        sim_prices,
        index=df_after_shock.index,
        columns=[f"Sim_{i+1}" for i in range(n_sims)],
    )
    sim_prices_stats = pd.DataFrame(
        {
            "Median": sim_prices_df.median(axis=1),
            "Mean": sim_prices_df.mean(axis=1),
            "Quantile_05": sim_prices_df.quantile(0.05, axis=1),
            "Quantile_95": sim_prices_df.quantile(0.95, axis=1),
        },
        index=df_after_shock.index,
    )

    sim_volatility_annualized = np.sqrt(sim_variance_unscaled * annualization_factor)

    sim_vol_df = pd.DataFrame(
        sim_volatility_annualized,
        index=df_after_shock.index,
        columns=[f"Sim_{i+1}" for i in range(n_sims)],
    )
    sim_vol_stats = pd.DataFrame(
        {
            "Median": sim_vol_df.median(axis=1),
            "Mean": sim_vol_df.mean(axis=1),
            "Quantile_05": sim_vol_df.quantile(0.05, axis=1),
            "Quantile_95": sim_vol_df.quantile(0.95, axis=1),
        },
        index=df_after_shock.index,
    )

    print("GARCH simulation processing complete.")
    return sim_prices_stats, sim_vol_stats


def run_dynamic_naive_simulation(
    df,
    df_after_shock,
    last_actual_price,
    n_sims=config.N_SIMS,
    window=config.WINDOW,
    annualization_factor=config.ANNUALIZATION_FACTOR,
):

    if "Log_Return" not in df.columns:
        print("Error: 'Log_Return' missing in input DataFrame for dynamic naive.")
        return None, None

    vol_col = "Realized_Volatility_Daily"
    mean_ret_col = "Rolling_Mean_Log_Return"

    if vol_col not in df.columns:
        print(f"Calculating '{vol_col}' for dynamic naive simulation.")
        df_copy = df.copy()
        df_copy[vol_col] = (
            df_copy["Log_Return"].rolling(window=window).std()
            * np.sqrt(annualization_factor)
            * 100
        )
    else:
        df_copy = df

    if mean_ret_col not in df_copy.columns:
        print(f"Calculating '{mean_ret_col}' for dynamic naive simulation.")
        df_copy[mean_ret_col] = df_copy["Log_Return"].rolling(window=window).mean()

    simulation_dates = df_after_shock.index
    n_steps = len(simulation_dates)
    if n_steps == 0:
        print("No post-shock period to simulate.")
        return None, None

    print(f"\nDynamic naive simulation horizon: {n_steps} days")
    print(f"Number of sims: {n_sims}")
    print(f"Starting price: {last_actual_price:.2f}")

    sim_prices_dyn = np.zeros((n_steps, n_sims))
    current_prices_dyn = np.full(n_sims, last_actual_price)
    used_hist_vol = np.zeros(n_steps)
    used_hist_mean_ret = np.zeros(n_steps)
    fallback_cnt = 0

    last_valid_vol = (
        df_copy[vol_col].dropna().iloc[-1]
        if not df_copy[vol_col].dropna().empty
        else 20.0
    )
    last_valid_mean = (
        df_copy[mean_ret_col].dropna().iloc[-1]
        if not df_copy[mean_ret_col].dropna().empty
        else 0.0
    )

    for i, current_sim_date in enumerate(simulation_dates):
        target_hist_date = current_sim_date - pd.DateOffset(years=1)

        hist_vol_percent = last_valid_vol
        hist_mean_ret = last_valid_mean
        found_hist_data = False

        try:
            hist_index_loc = df_copy.index.get_indexer(
                [target_hist_date], method="ffill"
            )[0]

            if hist_index_loc != -1:
                actual_hist_date = df_copy.index[hist_index_loc]
                vol_val = df_copy.loc[actual_hist_date, vol_col]
                mean_val = df_copy.loc[actual_hist_date, mean_ret_col]

                if pd.notna(vol_val):
                    hist_vol_percent = vol_val
                    last_valid_vol = vol_val
                else:
                    fallback_cnt += 1

                if pd.notna(mean_val):
                    hist_mean_ret = mean_val
                    last_valid_mean = mean_val

                found_hist_data = True  # noqa

            else:
                fallback_cnt += 1

        except KeyError as e:
            print(
                f"KeyError accessing data for date {actual_hist_date}: {e}.\
                    Using fallback."
            )
            fallback_cnt += 1
        except Exception as e:
            print(
                f"Error getting historical data for {target_hist_date.date()}: {e}.\
                    Using fallback."
            )
            fallback_cnt += 1

        used_hist_vol[i] = hist_vol_percent
        used_hist_mean_ret[i] = hist_mean_ret

        daily_std_dev = max(
            (hist_vol_percent / 100.0) / np.sqrt(annualization_factor), 1e-8
        )
        sim_daily_returns = np.random.normal(
            loc=hist_mean_ret, scale=daily_std_dev, size=n_sims
        )

        current_prices_dyn = current_prices_dyn * np.exp(sim_daily_returns)
        current_prices_dyn = np.maximum(current_prices_dyn, 1e-6)
        sim_prices_dyn[i, :] = current_prices_dyn

    print(
        f"Dynamic simulation finished.\
            Fallback values used {fallback_cnt} times out of {n_steps} steps."
    )

    sim_prices_df_dyn = pd.DataFrame(
        sim_prices_dyn,
        index=simulation_dates,
        columns=[f"Sim_{i + 1}" for i in range(n_sims)],
    )

    sim_prices_stats_dyn = pd.DataFrame(
        {
            "Median": sim_prices_df_dyn.median(axis=1),
            "Mean": sim_prices_df_dyn.mean(axis=1),
            "Quantile_05": sim_prices_df_dyn.quantile(0.05, axis=1),
            "Quantile_95": sim_prices_df_dyn.quantile(0.95, axis=1),
        },
        index=simulation_dates,
    )
    used_params_df = pd.DataFrame(
        {
            "Used_Historical_Volatility": used_hist_vol,
            "Used_Historical_Mean_Return": used_hist_mean_ret,
        },
        index=simulation_dates,
    )

    return sim_prices_stats_dyn, used_params_df


def run_fixed_naive_simulation(
    df,
    df_after_shock,
    last_actual_price,
    shock_date=config.SHOCK_DATE,
    n_sims=config.N_SIMS,
    window=config.WINDOW,
    annualization_factor=config.ANNUALIZATION_FACTOR,
):
    try:
        hist_end_date = pd.to_datetime(shock_date) - pd.Timedelta(days=1)
        hist_start_date = pd.to_datetime(shock_date) - pd.DateOffset(years=1)
        print(
            f"Calculating fixed parameters from:\
                {hist_start_date.date()} to {hist_end_date.date()}"
        )
    except Exception as crowley:
        print(f"Error setting up dates: {crowley}")
        return None, None

    historical_mask = (df.index >= hist_start_date) & (df.index <= hist_end_date)
    df_hist_period = df[historical_mask].copy()

    if df_hist_period.empty:
        print("No data found in the historical period for fixed naive params.")
        return None, None

    if "Log_Return" not in df_hist_period.columns:
        print("Error: 'Log_Return' missing in historical period.")
        return None, None

    if "Realized_Volatility_Daily" not in df_hist_period.columns:
        print("Calculating 'Realized_Volatility_Daily' for historical period...")
        df_hist_period["Realized_Volatility_Daily"] = (
            df_hist_period["Log_Return"].rolling(window=window).std()
            * np.sqrt(annualization_factor)
            * 100
        )
    if "Rolling_Mean_Log_Return" not in df_hist_period.columns:
        print("Calculating 'Rolling_Mean_Log_Return' for historical period...")
        df_hist_period["Rolling_Mean_Log_Return"] = (
            df_hist_period["Log_Return"].rolling(window=window).mean()
        )

    valid_means = df_hist_period["Rolling_Mean_Log_Return"].dropna()
    valid_vols = df_hist_period["Realized_Volatility_Daily"].dropna()

    if valid_means.empty or valid_vols.empty:
        print(
            "No valid mean return or volatility data in the historical period after\
                dropna."
        )
        return None, None

    fixed_hist_mean_ret = valid_means.mean()
    fixed_hist_vol_percent = valid_vols.mean()
    fixed_daily_std_dev = max(
        (fixed_hist_vol_percent / 100.0) / np.sqrt(annualization_factor), 1e-8
    )

    fixed_params = {
        "mean_log_return": fixed_hist_mean_ret,
        "annualized_vol_percent": fixed_hist_vol_percent,
        "daily_std_dev": fixed_daily_std_dev,
    }
    print("\nFixed Naive parameters:")
    for key, val in fixed_params.items():
        print(f"- {key}: {val:.6f}")

    simulation_dates = df_after_shock.index
    n_steps = len(simulation_dates)
    if n_steps == 0:
        print("No post-shock period to simulate.")
        return None, None

    print(f"\nFixed naive simulation horizon: {n_steps} days.")
    print(f"Number of simulations: {n_sims}")
    print(f"Starting price: {last_actual_price:.2f}")

    sim_prices_fixed = np.zeros((n_steps, n_sims))
    current_prices_fixed = np.full(n_sims, last_actual_price)

    for i in range(n_steps):
        sim_daily_returns = np.random.normal(
            loc=fixed_hist_mean_ret, scale=fixed_daily_std_dev, size=n_sims
        )
        current_prices_fixed = current_prices_fixed * np.exp(sim_daily_returns)
        current_prices_fixed = np.maximum(current_prices_fixed, 1e-6)
        sim_prices_fixed[i, :] = current_prices_fixed
    print("Fixed parameter naive simulation finished.")

    sim_prices_df_fixed = pd.DataFrame(
        sim_prices_fixed,
        index=simulation_dates,
        columns=[f"Sim_{i + 1}" for i in range(n_sims)],
    )

    sim_prices_stats_fixed = pd.DataFrame(
        {
            "Median": sim_prices_df_fixed.median(axis=1),
            "Mean": sim_prices_df_fixed.mean(axis=1),
            "Quantile_05": sim_prices_df_fixed.quantile(0.05, axis=1),
            "Quantile_95": sim_prices_df_fixed.quantile(0.95, axis=1),
        },
        index=simulation_dates,
    )

    return sim_prices_stats_fixed, fixed_params


def run_adaptive_garch_simulation(
    df,
    shock_date,
    p,
    q,
    o=0,
    model_type="GARCH",
    n_sims=config.N_SIMS,
    annualization_factor=config.ANNUALIZATION_FACTOR,
    dist=config.MODEL_DISTRIBUTION,
):
    print(
        f"\n--- Running Adaptive {model_type}({p},{o},{q})\
            Simulation ({n_sims} paths) ---"
    )

    shock_datetime = pd.to_datetime(shock_date)
    df_post_shock = df[df.index >= shock_datetime].copy()
    df_full_history = df.copy()

    if df_post_shock.empty:
        print("No post-shock period to simulate.")
        return None, None

    n_steps = len(df_post_shock)
    simulation_dates = df_post_shock.index
    last_actual_price = df_full_history.loc[
        simulation_dates[0] - pd.Timedelta(days=1), "Adj Close"
    ]

    print(f"Adaptive simulation horizon: {n_steps} steps")
    print(f"Starting price: {last_actual_price:.2f}")

    sim_prices_all = np.zeros((n_steps, n_sims))
    sim_vol_forecast_median = np.zeros(n_steps)
    current_prices = np.full(n_sims, last_actual_price)
    fit_fail_count = 0

    for i, current_sim_date in enumerate(simulation_dates):
        window_end_date = current_sim_date - pd.Timedelta(days=1)
        expanding_window_df = df_full_history[df_full_history.index <= window_end_date]
        returns_expanding = expanding_window_df["Log_Return"].dropna()

        if len(returns_expanding) < 30:
            print(
                f"Warning: Insufficient data ({len(returns_expanding)} points) to fit\
                    model on {current_sim_date.date()}. Using previous forecast/vol."
            )
            # TODO: maybe reuse previous forecast, or use a default vol?
            sim_vol_forecast_median[i] = np.nan
            sim_daily_log_returns = np.zeros(n_sims)
            fit_fail_count += 1
        else:
            print(
                f"Refitting model for {current_sim_date.date()} using data up to\
                    {window_end_date.date()}..."
            )
            refit_model_results = modeling.fit_garch_model(
                returns_expanding, p=p, q=q, o=o, dist=dist
            )

            if refit_model_results:
                try:
                    forecast = refit_model_results.forecast(
                        horizon=1, method="simulation", simulations=n_sims
                    )
                    sim_mean_scaled_step = forecast.simulations.values[0, 0, :]
                    sim_variance_step = forecast.simulations.residual_variances[0, 0, :]

                    sim_daily_log_returns = sim_mean_scaled_step / 100.0
                    sim_daily_vol_step = np.sqrt(
                        sim_variance_step * annualization_factor
                    )

                    sim_vol_forecast_median[i] = np.median(sim_daily_vol_step)

                except Exception as forecast_err:
                    print(
                        f"Error during forecast on {current_sim_date.date()}:\
                            {forecast_err}. Using fallback."
                    )
                    sim_vol_forecast_median[i] = np.nan
                    sim_daily_log_returns = np.zeros(n_sims)
                    fit_fail_count += 1

            else:
                print(
                    f"Warning: Model fitting failed for {current_sim_date.date()}.\
                        Using fallback."
                )
                sim_vol_forecast_median[i] = np.nan
                sim_daily_log_returns = np.zeros(n_sims)
                fit_fail_count += 1

        current_prices = current_prices * np.exp(sim_daily_log_returns)
        sim_prices_all[i, :] = current_prices

    print(f"Adaptive simulation finished. Model refit failed {fit_fail_count} times.")

    sim_prices_df = pd.DataFrame(
        sim_prices_all,
        index=simulation_dates,
        columns=[f"Sim_{i+1}" for i in range(n_sims)],
    )
    sim_prices_stats = pd.DataFrame(
        {
            "Median": sim_prices_df.median(axis=1),
            "Mean": sim_prices_df.mean(axis=1),
            "Quantile_05": sim_prices_df.quantile(0.05, axis=1),
            "Quantile_95": sim_prices_df.quantile(0.95, axis=1),
        },
        index=simulation_dates,
    )

    sim_vol_stats = pd.DataFrame(
        {"Median": sim_vol_forecast_median}, index=simulation_dates
    )

    return sim_prices_stats, sim_vol_stats
