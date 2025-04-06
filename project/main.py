
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
import data_handler
import analysis
import modeling
import simulation
import evaluation
import visualization

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import data_handler
import analysis
import modeling
import simulation
import evaluation
import visualization

def run_analysis():
    # --- 1. Data Fetching and Preprocessing ---
    raw_df = data_handler.fetch_stock_data()
    if raw_df.empty:
        print("Failed to fetch data. Exiting.")
        return
    df = data_handler.preprocess_data(raw_df)
    if df.empty:
        print("Data preprocessing failed. Exiting.")
        return

    # --- 2. Initial Visualization ---
    visualization.plot_initial_data(df, ticker=config.TICKER, shock_date=config.SHOCK_DATE)

    # --- 3. Empirical Analysis ---
    df_before_shock, df_after_shock = data_handler.split_data_by_shock(df, config.SHOCK_DATE)

    if df_before_shock.empty or df_after_shock.empty:
         print("Insufficient data before or after the shock date for analysis. Exiting.")
         return

    print("\n--- Empirical Analysis ---")
    vol_stats = analysis.analyze_volatility(df_before_shock, df_after_shock)
    return_stats, normality_results, returns_data = analysis.analyze_log_returns(df_before_shock, df_after_shock)

    if returns_data and vol_stats is not None:
        returns_before, returns_after = returns_data
        vol_before = df_before_shock['Realized_Volatility_Daily'].dropna()
        vol_after = df_after_shock['Realized_Volatility_Daily'].dropna()
        visualization.plot_analysis_distributions(returns_before, returns_after, vol_before, vol_after, config.SHOCK_DATE)

    # --- 4. GARCH Modeling ---
    returns_before_fit = df_before_shock['Log_Return'].dropna()
    if returns_before_fit.empty:
        print("No pre-shock returns available for GARCH modeling. Exiting.")
        return

    # Select Best GARCH Model
    best_garch_model, best_garch_params = modeling.select_best_garch_model(
        returns_before_fit, model_type='GARCH', criterion='BIC'
    )

    # Select Best GJR-GARCH Model
    best_gjr_model, best_gjr_params = modeling.select_best_garch_model(
        returns_before_fit, model_type='GJR', criterion='BIC'
    )

    # Choose the overall best model (e.g., based on BIC between GARCH and GJR)
    best_model = None
    best_params = None
    model_name = "No Model Selected"

    if best_garch_model and best_gjr_model:
        if best_gjr_model.bic < best_garch_model.bic:
            best_model = best_gjr_model
            best_params = best_gjr_params
            model_name = f"GJR-GARCH({best_params['p']},{best_params['o']},{best_params['q']})"
            print(f"\nSelected GJR-GARCH({best_params['p']},{best_params['o']},{best_params['q']}) as overall best (BIC: {best_model.bic:.2f}).")
        else:
            best_model = best_garch_model
            best_params = best_garch_params
            model_name = f"GARCH({best_params['p']},{best_params['q']})"
            print(f"\nSelected GARCH({best_params['p']},{best_params['q']}) as overall best (BIC: {best_model.bic:.2f}).")
    elif best_gjr_model:
        best_model = best_gjr_model
        best_params = best_gjr_params
        model_name = f"GJR-GARCH({best_params['p']},{best_params['o']},{best_params['q']})"
        print("\nOnly GJR-GARCH model available.")
    elif best_garch_model:
        best_model = best_garch_model
        best_params = best_garch_params
        model_name = f"GARCH({best_params['p']},{best_params['q']})"
        print("\nOnly GARCH model available.")
    else:
        print("\nError: No suitable GARCH or GJR-GARCH model could be fitted. Cannot proceed with simulation.")
        return

    # --- 5. GARCH Simulation & Evaluation ---
    last_actual_price = df_before_shock['Adj Close'].iloc[-1]

    # Calculate fitted values for the chosen best model
    fit_price_df, fit_vol_df = modeling.calculate_fitted_values(
        best_model, returns_before_fit, df_before_shock
    )

    # Run simulation
    sim_prices_stats, sim_vol_stats = simulation.run_garch_simulation(
        best_model, df_after_shock, last_actual_price
    )

    if sim_prices_stats is not None:
        # Evaluate simulation
        actual_prices_post_shock = df_after_shock['Adj Close']
        garch_metrics = evaluation.calculate_simulation_metrics(
            actual_prices_post_shock, sim_prices_stats['Median'])
        evaluation.display_metrics(garch_metrics, model_name)

        # Plot simulation results
        visualization.plot_simulation_results(
            df, fit_price_df, fit_vol_df, sim_prices_stats, sim_vol_stats,
            ticker=config.TICKER, shock_date=config.SHOCK_DATE, model_name=model_name
        )
    else:
        print("GARCH simulation failed.")


    # --- 6. Naive Approaches ---

    # Dynamic Naive
    sim_prices_stats_dyn, used_params_df = simulation.run_dynamic_naive_simulation(
        df, df_after_shock, last_actual_price
    )
    if sim_prices_stats_dyn is not None:
        dyn_naive_metrics = evaluation.calculate_simulation_metrics(
            df_after_shock['Adj Close'], sim_prices_stats_dyn['Median'])
        evaluation.display_metrics(dyn_naive_metrics, "Dynamic Naive (T-1yr)")
        visualization.plot_naive_simulation_results(
            df_after_shock, sim_prices_stats_dyn,
            ticker=config.TICKER, shock_date=config.SHOCK_DATE,
            approach_name="Dynamic Naive (T-1yr)",
            vol_data=used_params_df['Used_Historical_Volatility'],
            vol_label="Historical Vol Used (T-1yr)"
        )

    # Fixed Naive
    sim_prices_stats_fixed, fixed_params = simulation.run_fixed_naive_simulation(
        df, df_after_shock, last_actual_price
    )
    if sim_prices_stats_fixed is not None:
        fixed_naive_metrics = evaluation.calculate_simulation_metrics(
            df_after_shock['Adj Close'], sim_prices_stats_fixed['Median'])
        evaluation.display_metrics(fixed_naive_metrics, "Fixed Naive (Year Before Shock)")
        visualization.plot_naive_simulation_results(
            df_after_shock, sim_prices_stats_fixed,
            ticker=config.TICKER, shock_date=config.SHOCK_DATE,
            approach_name="Fixed Naive (Year Before Shock)",
            vol_data=fixed_params['annualized_vol_percent'],
            vol_label="Assumed Constant Volatility"
        )

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    run_analysis()