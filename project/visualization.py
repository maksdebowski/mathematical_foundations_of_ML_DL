import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import pandas as pd
import config
from scipy import stats


plt.style.use(config.PLOT_STYLE)

def plot_initial_data(df, ticker=config.TICKER, shock_date=config.SHOCK_DATE):
    print("\nPlotting initial price and volatility...")
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    price_col = 'Adj Close'
    vol_col = 'Realized_Volatility_Daily'

    if price_col not in df.columns:
        print(f"Warning: Price column '{price_col}' not found for plotting.")
        return
    if vol_col not in df.columns:
        print(f"Warning: Volatility column '{vol_col}' not found for plotting.")
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(df.index, df[price_col], color='blue', label=f'Price ({price_col})')
        ax.set_ylabel(f'Price ({price_col})')
        ax.set_title(f'{ticker} Price History')
        ax.grid(True)
        if shock_date:
            ax.axvline(pd.to_datetime(shock_date), color='red', linestyle='--', lw=1, label=f'Shock ({shock_date})')
        ax.legend()
        plt.tight_layout()
        plt.show()
        return


    # Plot Price
    ax[0].plot(df.index, df[price_col], color='blue')
    ax[0].set_ylabel(f'Price ({price_col})')
    ax[0].set_title(f'Price History for {ticker}')
    ax[0].grid(True)
    if shock_date:
        ax[0].axvline(pd.to_datetime(shock_date), color='red', linestyle='--', lw=1, label=f'Shock ({shock_date})')
    ax[0].legend()
    ax[0].set_yscale('log')

    # Plot Volatility
    plot_vol = df[vol_col].dropna()
    if not plot_vol.empty:
        ax[1].plot(plot_vol.index, plot_vol, color='green')
        ax[1].set_ylabel('Annualized Volatility (%)')
        ax[1].set_title(f'Historical Volatility (Rolling {config.WINDOW}-Day) for {ticker}')
        ax[1].grid(True)
        if shock_date:
            ax[1].axvline(pd.to_datetime(shock_date), color='red', linestyle='--', lw=1, label=f'Shock ({shock_date})')
        ax[1].legend()
        ax[1].set_ylim(bottom=0) # Volatility >= 0

        ax[1].xaxis.set_major_locator(mdates.YearLocator())
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        plt.xlabel('Date')
    else:
        ax[1].set_title("Volatility Data Not Available or Empty")


    plt.tight_layout()
    plt.show()

def plot_analysis_distributions(returns_before, returns_after, vol_before, vol_after, shock_date=config.SHOCK_DATE):
    print("\nPlotting analysis distributions...")

    # --- Volatility Distribution ---
    if vol_before is not None and not vol_before.empty and vol_after is not None and not vol_after.empty:
        plt.figure(figsize=(10, 5))
        sns.histplot(vol_before, color="skyblue", label='Before Shock', kde=True, stat="density", linewidth=0)
        sns.histplot(vol_after, color="red", label='After Shock', kde=True, stat="density", linewidth=0, alpha=0.6)
        plt.title('Distribution of Annualized Volatility')
        plt.xlabel('Annualized Volatility (%)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping volatility distribution plot (insufficient data).")


    # --- Log Return Distribution ---
    if returns_before is not None and not returns_before.empty and returns_after is not None and not returns_after.empty:
        fig_ret, ax_ret = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Before Shock Histogram + Normal Fit
        sns.histplot(returns_before, ax=ax_ret[0], kde=False, stat="density", label='Empirical', color='skyblue')
        mu_before, std_before = stats.norm.fit(returns_before)
        xmin, xmax = ax_ret[0].get_xlim()
        x_before = np.linspace(xmin, xmax, 100)
        p_before = stats.norm.pdf(x_before, mu_before, std_before)
        ax_ret[0].plot(x_before, p_before, 'k', linewidth=2, label=f'Normal Fit\n(μ={mu_before:.4f}, σ={std_before:.4f})')
        ax_ret[0].set_title(f'Log Returns Distribution Before Shock ({shock_date})')
        ax_ret[0].set_xlabel('Log Return')
        ax_ret[0].legend()

        # After Shock Histogram + Normal Fit
        sns.histplot(returns_after, ax=ax_ret[1], kde=False, stat="density", label='Empirical', color='salmon')
        mu_after, std_after = stats.norm.fit(returns_after)
        xmin, xmax = ax_ret[1].get_xlim()
        x_after = np.linspace(xmin, xmax, 100)
        p_after = stats.norm.pdf(x_after, mu_after, std_after)
        ax_ret[1].plot(x_after, p_after, 'k', linewidth=2, label=f'Normal Fit\n(μ={mu_after:.4f}, σ={std_after:.4f})')
        ax_ret[1].set_title(f'Log Returns Distribution After Shock ({shock_date})')
        ax_ret[1].set_xlabel('Log Return')
        ax_ret[1].legend()

        plt.tight_layout()
        plt.show()

        # --- Q-Q Plots ---
        fig_qq, ax_qq = plt.subplots(1, 2, figsize=(12, 5))

        sm.qqplot(returns_before, stats.norm, fit=True, line='45', ax=ax_qq[0])
        ax_qq[0].set_title(f'Q-Q Plot (Log Returns) Before Shock')

        sm.qqplot(returns_after, stats.norm, fit=True, line='45', ax=ax_qq[1])
        ax_qq[1].set_title(f'Q-Q Plot (Log Returns) After Shock')

        plt.tight_layout()
        plt.show()
    else:
        print("Skipping log return distribution plots (insufficient data).")


def plot_simulation_results(df, pre_shock_fit_price_df, pre_shock_fit_vol_df,
                            sim_prices_stats, sim_vol_stats,
                            ticker=config.TICKER, shock_date=config.SHOCK_DATE, model_name="GARCH"):
    print(f"\nGenerating counterfactual simulation plots ({model_name})...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{model_name} Model: Pre-Shock Fit & Post-Shock Simulation for {ticker}', fontsize=16)

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    vol_col = 'Realized_Volatility_Daily'

    # --- Price Plot ---
    # Actual Price (Full Period)
    axes[0].plot(df.index, df[price_col], label='Actual Price', color='#1f77b4', linewidth=1.5, zorder=1)
    # Fitted Price (Pre-Shock)
    if pre_shock_fit_price_df is not None and not pre_shock_fit_price_df.empty:
        axes[0].plot(pre_shock_fit_price_df.index, pre_shock_fit_price_df['Fitted_Price'],
                     label=f'{model_name} Fitted (In-Sample)', color='#ff7f0e', linewidth=1.5, linestyle='--', zorder=2)
    # Simulated Price (Post-Shock)
    if sim_prices_stats is not None and not sim_prices_stats.empty:
        axes[0].plot(sim_prices_stats.index, sim_prices_stats['Median'],
                     label=f'{model_name} Simulated Median', color='#2ca02c', linewidth=1.5, linestyle='--', zorder=2)
        axes[0].fill_between(sim_prices_stats.index, sim_prices_stats['Quantile_05'], sim_prices_stats['Quantile_95'],
                             color='#2ca02c', alpha=0.2, label='Simulated 5%-95% Range', zorder=0)

    axes[0].axvline(pd.to_datetime(shock_date), color='red', linestyle='--', lw=1.5, label=f'Shock ({shock_date})', zorder=3)
    axes[0].set_ylabel(f"{price_col} Price", fontsize=12)
    axes[0].set_title(f'Price: Actual vs. {model_name} Fit & Simulation', fontsize=14)
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # --- Volatility Plot ---
    # Actual Volatility (Full Period, where available)
    actual_vol = df[vol_col].dropna()
    if not actual_vol.empty:
        axes[1].plot(actual_vol.index, actual_vol, label='Actual Volatility (Proxy)', color='#1f77b4', linewidth=1.5, zorder=1)
    # Fitted Volatility (Pre-Shock)
    if pre_shock_fit_vol_df is not None and not pre_shock_fit_vol_df.empty:
        axes[1].plot(pre_shock_fit_vol_df.index, pre_shock_fit_vol_df["Fitted_Volatility"],
                     label=f'{model_name} Fitted Volatility (In-Sample)', color='#ff7f0e', linewidth=1.5, linestyle='--', zorder=2)
    # Simulated Volatility (Post-Shock)
    if sim_vol_stats is not None and not sim_vol_stats.empty:
        axes[1].plot(sim_vol_stats.index, sim_vol_stats['Median'],
                     label=f'{model_name} Simulated Median Volatility', color='#2ca02c', linewidth=1.5, linestyle='--', zorder=2)
        axes[1].fill_between(sim_vol_stats.index, sim_vol_stats['Quantile_05'], sim_vol_stats['Quantile_95'],
                             color='#2ca02c', alpha=0.2, label='Simulated 5%-95% Vol Range', zorder=0)

    axes[1].axvline(pd.to_datetime(shock_date), color='red', linestyle='--', lw=1.5, label=f'Shock ({shock_date})', zorder=3)
    axes[1].set_ylabel('Annualized Volatility (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_title(f'Volatility: Actual vs. {model_name} Fit & Simulation', fontsize=14)
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[0].get_xticklabels(), visible=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_naive_simulation_results(df_post_shock, sim_prices_stats_naive,
                                  ticker=config.TICKER, shock_date=config.SHOCK_DATE,
                                  approach_name="Naive Approach",
                                  vol_data=None, vol_label="Historical Volatility Used"):
    print(f"\nGenerating plot for {approach_name}...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{approach_name} Simulation for {ticker}', fontsize=16)

    price_col = 'Adj Close'
    vol_col = 'Realized_Volatility_Daily'

    # --- Price Plot ---
    actual_prices_post_shock = df_post_shock[price_col]
    axes[0].plot(actual_prices_post_shock.index, actual_prices_post_shock,
                 label='Actual Price (Post-Shock)', color='black', linewidth=1.5, alpha=0.9, zorder=1)
    if sim_prices_stats_naive is not None and not sim_prices_stats_naive.empty:
        axes[0].plot(sim_prices_stats_naive.index, sim_prices_stats_naive['Median'],
                     label=f'Median {approach_name} Sim.', color='darkcyan', linewidth=1.5, linestyle='--', zorder=2)
        axes[0].fill_between(sim_prices_stats_naive.index, sim_prices_stats_naive['Quantile_05'], sim_prices_stats_naive['Quantile_95'],
                             color='darkcyan', alpha=0.15, label='5-95% Range Sim.', zorder=0)

    axes[0].axvline(pd.to_datetime(shock_date), color='red', linestyle='--', lw=1.5, label=f'Shock ({shock_date})', zorder=3)
    axes[0].set_ylabel(f"{price_col} Price", fontsize=12)
    axes[0].set_title('Price Comparison: Actual vs. Simulation', fontsize=14)
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # --- Volatility Plot ---
    actual_vol_post_shock = df_post_shock[vol_col].dropna()
    if not actual_vol_post_shock.empty:
        axes[1].plot(actual_vol_post_shock.index, actual_vol_post_shock,
                     label='Actual Volatility (Post-Shock)', color='black', linewidth=1.5, alpha=0.9, zorder=1)

    if vol_data is not None:
        if isinstance(vol_data, pd.Series):
             axes[1].plot(vol_data.index, vol_data,
                         label=vol_label, color='darkcyan', linewidth=1.5, linestyle=':', zorder=2)
        elif isinstance(vol_data, (int, float)):
             axes[1].axhline(vol_data, color='forestgreen', linestyle='--', linewidth=1.5,
                             label=f'{vol_label} ({vol_data:.2f}%)', zorder=2)

    axes[1].axvline(pd.to_datetime(shock_date), color='red', linestyle='--', lw=1.5, label=f'Shock ({shock_date})', zorder=3)
    axes[1].set_ylabel("Annualized Volatility (%)", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_title('Volatility Comparison: Actual vs. Assumed/Used', fontsize=14)
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[0].get_xticklabels(), visible=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

