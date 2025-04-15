import pandas as pd
import numpy as np
import config
import data_handler
import analysis
import modeling
import simulation
from evaluation import (
    calculate_price_metrics,
    calculate_volatility_metrics,
    display_combined_metrics, # Upewnij się, że importujesz poprawną wersję
    qlike_loss,
)
import visualization # Zmieniono import, aby pasował do nazwy pliku


def run_analysis():
    print("--- 1. Fetching and Preprocessing Data ---")
    raw_df = data_handler.fetch_stock_data()
    if raw_df.empty:
        print("Failed to fetch data. Exiting.")
        return

    df = data_handler.preprocess_data(raw_df)
    if df.empty:
        print("Data preprocessing failed. Exiting.")
        return

    actual_vol_col_name = "Realized_Volatility_Daily"
    if "Log_Return" not in df.columns or actual_vol_col_name not in df.columns:
        print(
            f"Error: Essential columns ('Log_Return', '{actual_vol_col_name}')\
                not found after preprocessing. Exiting."
        )
        return
    print(f"Using volatility column: '{actual_vol_col_name}'")

    print("\n--- 2. Initial Visualization ---")
    visualization.plot_initial_data(
        df,
        ticker=config.TICKER,
        shock_date=config.SHOCK_DATE,
        vol_col=actual_vol_col_name,
    )

    print("\n--- 3. Empirical Analysis ---")
    df_before_shock, df_after_shock = data_handler.split_data_by_shock(
        df, config.SHOCK_DATE
    )

    if df_before_shock.empty or df_after_shock.empty:
        print("Insufficient data before or after the shock date for analysis. Exiting.")
        return

    vol_stats = analysis.analyze_volatility(df_before_shock, df_after_shock)
    return_stats, normality_results, returns_data = analysis.analyze_log_returns(
        df_before_shock, df_after_shock
    )

    if returns_data and vol_stats is not None:
        returns_before, returns_after = returns_data
        vol_before = df_before_shock[actual_vol_col_name].dropna()
        vol_after = df_after_shock[actual_vol_col_name].dropna()
        visualization.plot_analysis_distributions(
            returns_before, returns_after, vol_before, vol_after, config.SHOCK_DATE
        )

    print("\n--- 4. GARCH Modeling (Pre-Shock) ---")
    returns_before_fit = df_before_shock["Log_Return"].dropna()
    if returns_before_fit.empty:
        print("No pre-shock returns available for GARCH modeling. Exiting.")
        return

    print("Fitting GARCH(1,1)...")
    garch_11_model_results = modeling.fit_garch_model(returns_before_fit, p=1, q=1, o=0)
    garch_11_params = None # noqa
    garch_11_fit_price = None
    garch_11_fit_vol = None
    if garch_11_model_results:
        garch_11_params = {"p": 1, "q": 1, "o": 0}
        print("GARCH(1,1) fitted successfully. Calculating fitted values...")
        garch_11_fit_price, garch_11_fit_vol = modeling.calculate_fitted_values(
            garch_11_model_results, returns_before_fit, df_before_shock
        )
    else:
        print("GARCH(1,1) fitting failed.")

    print("Fitting GJR-GARCH(1,1,1)...")
    gjr_111_model_results = modeling.fit_garch_model(returns_before_fit, p=1, o=1, q=1)
    gjr_111_params = None # noqa
    gjr_111_fit_price = None
    gjr_111_fit_vol = None
    if gjr_111_model_results:
        gjr_111_params = {"p": 1, "q": 1, "o": 1}
        print("GJR-GARCH(1,1,1) fitted successfully. Calculating fitted values...")
        gjr_111_fit_price, gjr_111_fit_vol = modeling.calculate_fitted_values(
            gjr_111_model_results, returns_before_fit, df_before_shock
        )
    else:
        print("GJR-GARCH(1,1,1) fitting failed.")

    print("\n--- 5. Fixed Pre-Shock GARCH/GJR Simulation & Evaluation ---")
    # Sprawdzenie czy df_before_shock nie jest pusty przed dostępem do iloc[-1]
    if df_before_shock.empty:
        print("Pre-shock data is empty, cannot determine last actual price. Skipping simulations.")
        return # Lub inna obsługa błędu

    last_actual_price = df_before_shock["Adj Close"].iloc[-1]
    actual_prices_post_shock = df_after_shock["Adj Close"]
    actual_vol_post_shock = df_after_shock[actual_vol_col_name]

    model_name_garch_fixed = "GARCH(1,1) Fixed Pre-Shock"
    if garch_11_model_results:
        print(f"\nSimulating with {model_name_garch_fixed} parameters...")
        sim_prices_garch_fixed, sim_vol_garch_fixed = simulation.run_garch_simulation(
            garch_11_model_results,
            df_after_shock,
            last_actual_price,
            n_sims=config.N_SIMS,
        )
        if sim_prices_garch_fixed is not None and sim_vol_garch_fixed is not None:
            # Dopasowanie indeksów przed ewaluacją
            common_idx_price = actual_prices_post_shock.index.intersection(sim_prices_garch_fixed.index)
            common_idx_vol = actual_vol_post_shock.index.intersection(sim_vol_garch_fixed.index)

            garch_fixed_price_metrics = calculate_price_metrics(
                actual_prices_post_shock.loc[common_idx_price],
                sim_prices_garch_fixed.loc[common_idx_price, "Median"]
            )
            garch_fixed_vol_metrics = calculate_volatility_metrics(
                actual_vol_post_shock.loc[common_idx_vol],
                sim_vol_garch_fixed.loc[common_idx_vol, "Median"]
            )
            display_combined_metrics(
                garch_fixed_price_metrics,
                garch_fixed_vol_metrics,
                model_name_garch_fixed,
            )

            visualization.plot_simulation_results(
                df=df,
                pre_shock_fit_price_df=garch_11_fit_price,
                pre_shock_fit_vol_df=garch_11_fit_vol,
                sim_prices_stats=sim_prices_garch_fixed,
                sim_vol_stats=sim_vol_garch_fixed,
                ticker=config.TICKER,
                shock_date=config.SHOCK_DATE,
                model_name=model_name_garch_fixed,
            )
        else:
            print(f"{model_name_garch_fixed} simulation failed.")

    model_name_gjr_fixed = "GJR-GARCH(1,1,1) Fixed Pre-Shock"
    if gjr_111_model_results:
        print(f"\nSimulating with {model_name_gjr_fixed} parameters...")
        sim_prices_gjr_fixed, sim_vol_gjr_fixed = simulation.run_garch_simulation(
            gjr_111_model_results,
            df_after_shock,
            last_actual_price,
            n_sims=config.N_SIMS,
        )
        if sim_prices_gjr_fixed is not None and sim_vol_gjr_fixed is not None:
            # Dopasowanie indeksów przed ewaluacją
            common_idx_price = actual_prices_post_shock.index.intersection(sim_prices_gjr_fixed.index)
            common_idx_vol = actual_vol_post_shock.index.intersection(sim_vol_gjr_fixed.index)

            gjr_fixed_price_metrics = calculate_price_metrics(
                actual_prices_post_shock.loc[common_idx_price],
                sim_prices_gjr_fixed.loc[common_idx_price, "Median"]
            )
            gjr_fixed_vol_metrics = calculate_volatility_metrics(
                actual_vol_post_shock.loc[common_idx_vol],
                sim_vol_gjr_fixed.loc[common_idx_vol, "Median"]
            )
            display_combined_metrics(
                gjr_fixed_price_metrics, gjr_fixed_vol_metrics, model_name_gjr_fixed
            )

            visualization.plot_simulation_results(
                df=df,
                pre_shock_fit_price_df=gjr_111_fit_price,
                pre_shock_fit_vol_df=gjr_111_fit_vol,
                sim_prices_stats=sim_prices_gjr_fixed,
                sim_vol_stats=sim_vol_gjr_fixed,
                ticker=config.TICKER,
                shock_date=config.SHOCK_DATE,
                model_name=model_name_gjr_fixed,
            )
        else:
            print(f"{model_name_gjr_fixed} simulation failed.")

    print("\n--- 6. Adaptive GARCH/GJR Simulation & Evaluation ---")
    model_name_garch_adapt = "GARCH(1,1) Adaptive"
    print(f"\nSimulating with {model_name_garch_adapt} parameters...")
    sim_prices_garch_adapt, sim_vol_garch_adapt = (
        simulation.run_adaptive_garch_simulation(
            df, config.SHOCK_DATE, p=1, q=1, model_type="GARCH", n_sims=config.N_SIMS
        )
    )
    if sim_prices_garch_adapt is not None and sim_vol_garch_adapt is not None:
        # Dopasowanie indeksów przed ewaluacją
        common_idx_price = actual_prices_post_shock.index.intersection(sim_prices_garch_adapt.index)
        common_idx_vol = actual_vol_post_shock.index.intersection(sim_vol_garch_adapt.index)

        garch_adapt_price_metrics = calculate_price_metrics(
            actual_prices_post_shock.loc[common_idx_price],
            sim_prices_garch_adapt.loc[common_idx_price, "Median"]
        )
        garch_adapt_vol_metrics = calculate_volatility_metrics(
            actual_vol_post_shock.loc[common_idx_vol],
            sim_vol_garch_adapt.loc[common_idx_vol, "Median"]
        )
        display_combined_metrics(
            garch_adapt_price_metrics, garch_adapt_vol_metrics, model_name_garch_adapt
        )

        visualization.plot_simulation_results(
            df=df,
            pre_shock_fit_price_df=None, # Brak dopasowania pre-shock dla adaptive
            pre_shock_fit_vol_df=None,  # Brak dopasowania pre-shock dla adaptive
            sim_prices_stats=sim_prices_garch_adapt,
            sim_vol_stats=sim_vol_garch_adapt,
            ticker=config.TICKER,
            shock_date=config.SHOCK_DATE,
            model_name=model_name_garch_adapt,
        )
    else:
        print(f"{model_name_garch_adapt} simulation failed.")

    model_name_gjr_adapt = "GJR-GARCH(1,1,1) Adaptive"
    print(f"\nSimulating with {model_name_gjr_adapt} parameters...")
    sim_prices_gjr_adapt, sim_vol_gjr_adapt = simulation.run_adaptive_garch_simulation(
        df, config.SHOCK_DATE, p=1, o=1, q=1, model_type="GJR", n_sims=config.N_SIMS
    )
    if sim_prices_gjr_adapt is not None and sim_vol_gjr_adapt is not None:
        # Dopasowanie indeksów przed ewaluacją
        common_idx_price = actual_prices_post_shock.index.intersection(sim_prices_gjr_adapt.index)
        common_idx_vol = actual_vol_post_shock.index.intersection(sim_vol_gjr_adapt.index)

        gjr_adapt_price_metrics = calculate_price_metrics(
            actual_prices_post_shock.loc[common_idx_price],
            sim_prices_gjr_adapt.loc[common_idx_price, "Median"]
        )
        gjr_adapt_vol_metrics = calculate_volatility_metrics(
            actual_vol_post_shock.loc[common_idx_vol],
            sim_vol_gjr_adapt.loc[common_idx_vol, "Median"]
        )
        display_combined_metrics(
            gjr_adapt_price_metrics, gjr_adapt_vol_metrics, model_name_gjr_adapt
        )

        visualization.plot_simulation_results(
            df=df,
            pre_shock_fit_price_df=None, # Brak dopasowania pre-shock dla adaptive
            pre_shock_fit_vol_df=None,  # Brak dopasowania pre-shock dla adaptive
            sim_prices_stats=sim_prices_gjr_adapt,
            sim_vol_stats=sim_vol_gjr_adapt,
            ticker=config.TICKER,
            shock_date=config.SHOCK_DATE,
            model_name=model_name_gjr_adapt,
        )
    else:
        print(f"{model_name_gjr_adapt} simulation failed.")

    print("\n--- 7. Naive Approaches Simulation & Evaluation ---")

    approach_name_dyn = "Dynamic Naive (T-1yr)"
    print(f"\nSimulating with {approach_name_dyn} approach...")
    sim_prices_dyn, used_params_df = simulation.run_dynamic_naive_simulation(
        df, df_after_shock, last_actual_price, n_sims=config.N_SIMS
    )
    if sim_prices_dyn is not None and used_params_df is not None:
        # Dopasowanie indeksów przed ewaluacją
        common_idx_price = actual_prices_post_shock.index.intersection(sim_prices_dyn.index)
        common_idx_vol = actual_vol_post_shock.index.intersection(used_params_df.index)

        dyn_naive_price_metrics = calculate_price_metrics(
            actual_prices_post_shock.loc[common_idx_price],
            sim_prices_dyn.loc[common_idx_price, "Median"]
        )
        predicted_vol_dyn = used_params_df.loc[common_idx_vol, "Used_Historical_Volatility"]
        dyn_naive_vol_metrics = calculate_volatility_metrics(
            actual_vol_post_shock.loc[common_idx_vol],
            predicted_vol_dyn
        )
        display_combined_metrics(
            dyn_naive_price_metrics, dyn_naive_vol_metrics, approach_name_dyn
        )

        visualization.plot_naive_simulation_results(
            df_post_shock=df_after_shock,
            sim_prices_stats_naive=sim_prices_dyn,
            ticker=config.TICKER,
            shock_date=config.SHOCK_DATE,
            approach_name=approach_name_dyn,
            vol_data=used_params_df["Used_Historical_Volatility"], # Przekaż całą serię do wykresu
            vol_label="Historical Vol Used (T-1yr)",
        )
    else:
        print(f"{approach_name_dyn} simulation failed or did not return parameters.")


    approach_name_fixed = "Fixed Naive (Year Before Shock)"
    print(f"\nSimulating with {approach_name_fixed} approach...") # Dodano print
    sim_prices_fixed, fixed_params = simulation.run_fixed_naive_simulation(
        df, df_after_shock, last_actual_price, n_sims=config.N_SIMS
    )
    if sim_prices_fixed is not None and fixed_params is not None:
        # Dopasowanie indeksów przed ewaluacją
        common_idx_price = actual_prices_post_shock.index.intersection(sim_prices_fixed.index)

        fixed_naive_price_metrics = calculate_price_metrics(
            actual_prices_post_shock.loc[common_idx_price],
            sim_prices_fixed.loc[common_idx_price, "Median"]
        )
        constant_vol_val = fixed_params.get("annualized_vol_percent", np.nan)
        if pd.isna(constant_vol_val):
            print(
                "Warning: Could not retrieve constant volatility for Fixed Naive model."
            )
            # Nie można obliczyć metryk zmienności
            fixed_naive_vol_metrics = None
        else:
            # Utwórz serię stałej zmienności dopasowaną do indeksu rzeczywistej zmienności
            predicted_vol_fixed = pd.Series(
                constant_vol_val, index=actual_vol_post_shock.index
            ).reindex(actual_vol_post_shock.index) # Upewnij się, że indeks pasuje

            fixed_naive_vol_metrics = calculate_volatility_metrics(
                actual_vol_post_shock, # Przekaż całą serię rzeczywistą
                predicted_vol_fixed # Przekaż całą serię przewidywaną
            )

        display_combined_metrics(
            fixed_naive_price_metrics, fixed_naive_vol_metrics, approach_name_fixed
        )

        if not pd.isna(constant_vol_val):
            visualization.plot_naive_simulation_results(
                df_post_shock=df_after_shock,
                sim_prices_stats_naive=sim_prices_fixed,
                ticker=config.TICKER,
                shock_date=config.SHOCK_DATE,
                approach_name=approach_name_fixed,
                vol_data=constant_vol_val, # Przekaż pojedynczą wartość do wykresu
                vol_label="Assumed Constant Volatility",
            )
        else:
            print(
                f"Skipping plot for {approach_name_fixed}\
                    due to missing volatility parameter."
            )
    else:
        print(f"{approach_name_fixed} simulation failed or did not return parameters.")

    # --- 8. 1-Day Naive Forecast & Evaluation --- NOWA SEKCJA ---
    approach_name_1day = "1-Day Naive"
    print(f"\n--- 8. {approach_name_1day} Forecast & Evaluation ---")
    print(f"\nRunning {approach_name_1day} forecast...")
    one_day_naive_price_fcst, one_day_naive_vol_fcst = simulation.run_one_day_naive_forecast(
        df, df_after_shock # Przekaż pełny df i df po szoku
    )

    if one_day_naive_price_fcst is not None:
        # Ewaluacja
        print(f"Evaluating {approach_name_1day}...")
        # Dopasuj indeksy przed ewaluacją
        common_index_price = actual_prices_post_shock.index.intersection(one_day_naive_price_fcst.index)
        if common_index_price.empty:
             print(f"Warning: No common dates found for {approach_name_1day} price evaluation.")
             one_day_price_metrics = None
        else:
            one_day_price_metrics = calculate_price_metrics(
                actual_prices_post_shock.loc[common_index_price],
                one_day_naive_price_fcst.loc[common_index_price, "Predicted_Price"]
            )

        one_day_vol_metrics = None
        if one_day_naive_vol_fcst is not None:
             common_index_vol = actual_vol_post_shock.index.intersection(one_day_naive_vol_fcst.index)
             # Sprawdź czy indeksy nie są puste po przecięciu
             if not common_index_vol.empty:
                 one_day_vol_metrics = calculate_volatility_metrics(
                     actual_vol_post_shock.loc[common_index_vol],
                     one_day_naive_vol_fcst.loc[common_index_vol, "Predicted_Volatility"]
                 )
             else:
                 print(f"Warning: No common dates found for {approach_name_1day} volatility evaluation.")

        display_combined_metrics(one_day_price_metrics, one_day_vol_metrics, model_name=approach_name_1day)

        # Wizualizacja
        print(f"Plotting {approach_name_1day} results...")
        visualization.plot_single_forecast_results(
            df_post_shock=df_after_shock,
            forecast_price_df=one_day_naive_price_fcst,
            forecast_vol_df=one_day_naive_vol_fcst, # Przekaż prognozę zmienności
            ticker=config.TICKER,
            shock_date=config.SHOCK_DATE,
            model_name=approach_name_1day,
            price_col_forecast='Predicted_Price', # Określ nazwę kolumny
            vol_col_forecast='Predicted_Volatility' # Określ nazwę kolumny
        )
    else:
        print(f"{approach_name_1day} forecast failed.")


    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    run_analysis()