import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import pandas as pd
import config
from scipy import stats


plt.style.use(config.PLOT_STYLE)

def plot_single_forecast_results(
    df_post_shock,
    forecast_price_df, # DataFrame z 'Predicted_Price'
    forecast_vol_df=None, # Opcjonalny DataFrame z 'Predicted_Volatility'
    ticker=config.TICKER,
    shock_date=config.SHOCK_DATE,
    model_name="Forecast Model",
    price_col_forecast='Predicted_Price', # Nazwa kolumny w forecast_price_df
    vol_col_forecast='Predicted_Volatility' # Nazwa kolumny w forecast_vol_df
):
    """
    Rysuje wyniki dla modeli, które produkują pojedynczą linię prognozy (jak 1-Day Naive).
    """
    print(f"\nGenerowanie wykresu dla {model_name}...")
    # Sprawdź, czy mamy dane do wykresu zmienności
    plot_vol = forecast_vol_df is not None and not forecast_vol_df.empty and vol_col_forecast in forecast_vol_df.columns

    n_plots = 2 if plot_vol else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes] # Uczyń iterowalnym dla spójności

    fig.suptitle(f"{model_name} Forecast for {ticker}", fontsize=16)

    price_col_actual = "Adj Close"
    vol_col_actual = "Realized_Volatility_Daily"

    # --- Wykres Ceny ---
    ax_price = axes[0]
    actual_prices_post_shock = df_post_shock[price_col_actual].dropna()
    if not actual_prices_post_shock.empty:
        ax_price.plot(
            actual_prices_post_shock.index,
            actual_prices_post_shock,
            label="Rzeczywista Cena (Po Szoku)",
            color="black",
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )
    else:
        print(f"Ostrzeżenie [{model_name}]: Brak rzeczywistych cen po szoku do narysowania.")

    if forecast_price_df is not None and not forecast_price_df.empty:
        if price_col_forecast in forecast_price_df.columns:
            predicted_prices_to_plot = forecast_price_df[price_col_forecast].dropna()
            if not predicted_prices_to_plot.empty:
                ax_price.plot(
                    predicted_prices_to_plot.index,
                    predicted_prices_to_plot,
                    label=f"Prognoza {model_name}",
                    color="purple", # Kolor dla 1-day naive
                    linewidth=1.5,
                    linestyle="--",
                    zorder=2,
                )
            else:
                 print(f"Ostrzeżenie [{model_name}]: Przewidywane ceny są puste po dropna.")
        else:
            print(f"Ostrzeżenie [{model_name}]: Kolumna '{price_col_forecast}' nie znaleziona w df prognozy cen.")
    else:
        print(f"Ostrzeżenie [{model_name}]: forecast_price_df jest None lub pusty.")

    ax_price.axvline(
        pd.to_datetime(shock_date), color="red", linestyle="--", lw=1.5, label=f"Szok ({shock_date})", zorder=3
    )
    ax_price.set_ylabel(f"Cena {price_col_actual}", fontsize=12)
    ax_price.set_title("Price Comparison: Actual vs. Forecast", fontsize=14)
    ax_price.legend(frameon=True, fontsize=10)
    ax_price.grid(True, alpha=0.3)

    # Użyj skali logarytmicznej tylko jeśli ceny są dodatnie
    try:
        use_log_scale = False
        if forecast_price_df is not None and not forecast_price_df.empty and price_col_forecast in forecast_price_df.columns:
            # Sprawdź czy wszystkie wartości są dodatnie
            all_preds_positive = (forecast_price_df[price_col_forecast].dropna() > 0).all()
            all_actuals_positive = (actual_prices_post_shock > 0).all()
            if all_preds_positive and all_actuals_positive:
                use_log_scale = True

        if use_log_scale:
            ax_price.set_yscale("log")
        else:
            print(f"Info [{model_name}]: Nie używam skali logarytmicznej dla ceny z powodu wartości nie-dodatnich.")
    except Exception as e:
        print(f"Ostrzeżenie [{model_name}]: Błąd przy ustawianiu skali logarytmicznej: {e}")


    # --- Wykres Zmienności (Opcjonalny) ---
    if plot_vol:
        ax_vol = axes[1]
        actual_vol_post_shock = df_post_shock[vol_col_actual].dropna()
        if not actual_vol_post_shock.empty:
            ax_vol.plot(
                actual_vol_post_shock.index,
                actual_vol_post_shock,
                label="Real Volatility (After Shock)",
                color="black",
                linewidth=1.5,
                alpha=0.9,
                zorder=1,
            )
        else:
             print(f"Ostrzeżenie [{model_name}]: Brak rzeczywistej zmienności po szoku do narysowania.")

        predicted_vol_to_plot = forecast_vol_df[vol_col_forecast].dropna()
        if not predicted_vol_to_plot.empty:
            ax_vol.plot(
                predicted_vol_to_plot.index,
                predicted_vol_to_plot,
                label=f"Volatility {model_name}",
                color="purple",
                linewidth=1.5,
                linestyle=":",
                zorder=2,
            )
        else:
            print(f"Ostrzeżenie [{model_name}]: Przewidywana zmienność jest pusta po dropna.")

        ax_vol.axvline(
            pd.to_datetime(shock_date), color="red", linestyle="--", lw=1.5, label=f"Szok ({shock_date})", zorder=3
        )
        ax_vol.set_ylabel("Annual Volatility (%)", fontsize=12)
        ax_vol.set_xlabel("Data", fontsize=12)
        ax_vol.legend(frameon=True, fontsize=10)
        ax_vol.grid(True, alpha=0.3)
        ax_vol.set_ylim(bottom=0)

        try:
            ax_vol.xaxis.set_major_locator(mdates.YearLocator())
            ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax_vol.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax_price.get_xticklabels(), visible=False)
        except Exception as e:
            print(f"Ostrzeżenie [{model_name}]: {e}")

    else:
         ax_price.set_xlabel("Data", fontsize=12)
         try:
            ax_price.xaxis.set_major_locator(mdates.YearLocator())
            ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax_price.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
         except Exception as e:
            print(f"Ostrzeżenie [{model_name}]: Błąd przy ustawianiu formaterów dat: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_initial_data(
    df,
    ticker=config.TICKER,
    shock_date=config.SHOCK_DATE,
    vol_col="Realized_Volatility_Daily",
):
    print("\nPlotting initial price and volatility...")
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    price_col = "Adj Close"
    if price_col not in df.columns:
        print(f"Warning: Price column '{price_col}' not found for plotting.")
        return
    if vol_col not in df.columns:
        print(f"Warning: Volatility column '{vol_col}' not found for plotting.")
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(df.index, df[price_col], color="blue", label=f"Price ({price_col})")
        ax.set_ylabel(f"Price ({price_col})")
        ax.set_title(f"{ticker} Price History")
        ax.grid(True)
        if shock_date:
            ax.axvline(
                pd.to_datetime(shock_date),
                color="red",
                linestyle="--",
                lw=1,
                label=f"Shock ({shock_date})",
            )
        ax.legend()
        plt.tight_layout()
        plt.show()
        return

    ax[0].plot(df.index, df[price_col], color="blue")
    ax[0].set_ylabel(f"Price ({price_col})")
    ax[0].set_title(f"Price History for {ticker}")
    ax[0].grid(True)
    if shock_date:
        ax[0].axvline(
            pd.to_datetime(shock_date),
            color="red",
            linestyle="--",
            lw=1,
            label=f"Shock ({shock_date})",
        )
    ax[0].legend()
    ax[0].set_yscale("log")

    plot_vol = df[vol_col].dropna()
    if not plot_vol.empty:
        ax[1].plot(plot_vol.index, plot_vol, color="green")
        ax[1].set_ylabel("Annualized Volatility (%)")
        ax[1].set_title(
            f"Historical Volatility (Rolling {config.WINDOW}-Day) for {ticker}"
        )
        ax[1].grid(True)
        if shock_date:
            ax[1].axvline(
                pd.to_datetime(shock_date),
                color="red",
                linestyle="--",
                lw=1,
                label=f"Shock ({shock_date})",
            )
        ax[1].legend()
        ax[1].set_ylim(bottom=0)

        ax[1].xaxis.set_major_locator(mdates.YearLocator())
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        plt.xlabel("Date")
    else:
        ax[1].set_title("Volatility Data Not Available or Empty")

    plt.tight_layout()
    plt.show()


def plot_analysis_distributions(
    returns_before, returns_after, vol_before, vol_after, shock_date=config.SHOCK_DATE
):
    print("\nPlotting analysis distributions...")

    if (
        vol_before is not None
        and not vol_before.empty
        and vol_after is not None
        and not vol_after.empty
    ):
        plt.figure(figsize=(10, 5))
        sns.histplot(
            vol_before,
            color="skyblue",
            label="Before Shock",
            kde=True,
            stat="density",
            linewidth=0,
        )
        sns.histplot(
            vol_after,
            color="red",
            label="After Shock",
            kde=True,
            stat="density",
            linewidth=0,
            alpha=0.6,
        )
        plt.title("Distribution of Annualized Volatility")
        plt.xlabel("Annualized Volatility (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping volatility distribution plot (insufficient data).")

    if (
        returns_before is not None
        and not returns_before.empty
        and returns_after is not None
        and not returns_after.empty
    ):
        fig_ret, ax_ret = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        sns.histplot(
            returns_before,
            ax=ax_ret[0],
            kde=False,
            stat="density",
            label="Empirical",
            color="skyblue",
        )
        mu_before, std_before = stats.norm.fit(returns_before)
        xmin, xmax = ax_ret[0].get_xlim()
        x_before = np.linspace(xmin, xmax, 100)
        p_before = stats.norm.pdf(x_before, mu_before, std_before)
        ax_ret[0].plot(
            x_before,
            p_before,
            "k",
            linewidth=2,
            label=f"Normal Fit\n(μ={mu_before:.4f}, σ={std_before:.4f})",
        )
        ax_ret[0].set_title(f"Log Returns Distribution Before Shock ({shock_date})")
        ax_ret[0].set_xlabel("Log Return")
        ax_ret[0].legend()

        sns.histplot(
            returns_after,
            ax=ax_ret[1],
            kde=False,
            stat="density",
            label="Empirical",
            color="salmon",
        )
        mu_after, std_after = stats.norm.fit(returns_after)
        xmin, xmax = ax_ret[1].get_xlim()
        x_after = np.linspace(xmin, xmax, 100)
        p_after = stats.norm.pdf(x_after, mu_after, std_after)
        ax_ret[1].plot(
            x_after,
            p_after,
            "k",
            linewidth=2,
            label=f"Normal Fit\n(μ={mu_after:.4f}, σ={std_after:.4f})",
        )
        ax_ret[1].set_title(f"Log Returns Distribution After Shock ({shock_date})")
        ax_ret[1].set_xlabel("Log Return")
        ax_ret[1].legend()

        plt.tight_layout()
        plt.show()

        fig_qq, ax_qq = plt.subplots(1, 2, figsize=(12, 5))

        sm.qqplot(returns_before, stats.norm, fit=True, line="45", ax=ax_qq[0])
        ax_qq[0].set_title("Q-Q Plot (Log Returns) Before Shock")

        sm.qqplot(returns_after, stats.norm, fit=True, line="45", ax=ax_qq[1])
        ax_qq[1].set_title("Q-Q Plot (Log Returns) After Shock")

        plt.tight_layout()
        plt.show()
    else:
        print("Skipping log return distribution plots (insufficient data).")


def plot_simulation_results(
    df,
    pre_shock_fit_price_df,
    pre_shock_fit_vol_df,
    sim_prices_stats,
    sim_vol_stats,
    ticker=config.TICKER,
    shock_date=config.SHOCK_DATE,
    model_name="GARCH",
):
    print(f"\nGenerating simulation plots ({model_name})...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"{model_name} Model: Pre-Shock Fit & Post-Shock Simulation for {ticker}",
        fontsize=16,
    )

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    vol_col = "Realized_Volatility_Daily"

    axes[0].plot(
        df.index,
        df[price_col],
        label="Actual Price",
        color="#1f77b4",
        linewidth=1.5,
        zorder=1,
    )
    if pre_shock_fit_price_df is not None and not pre_shock_fit_price_df.empty:
        axes[0].plot(
            pre_shock_fit_price_df.index,
            pre_shock_fit_price_df["Fitted_Price"],
            label=f"{model_name} Fitted (In-Sample)",
            color="#ff7f0e",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
        )
    if sim_prices_stats is not None and not sim_prices_stats.empty:
        axes[0].plot(
            sim_prices_stats.index,
            sim_prices_stats["Median"],
            label=f"{model_name} Simulated Median",
            color="#2ca02c",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
        )
        if (
            "Quantile_05" in sim_prices_stats.columns
            and "Quantile_95" in sim_prices_stats.columns
        ):
            axes[0].fill_between(
                sim_prices_stats.index,
                sim_prices_stats["Quantile_05"],
                sim_prices_stats["Quantile_95"],
                color="#2ca02c",
                alpha=0.2,
                label="Simulated 5%-95% Price Range",
                zorder=0,
            )

    axes[0].axvline(
        pd.to_datetime(shock_date),
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Shock ({shock_date})",
        zorder=3,
    )
    axes[0].set_ylabel(f"{price_col} Price", fontsize=12)
    axes[0].set_title(f"Price: Actual vs. {model_name} Fit & Simulation", fontsize=14)
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    actual_vol = df[vol_col].dropna()
    if not actual_vol.empty:
        axes[1].plot(
            actual_vol.index,
            actual_vol,
            label="Actual Volatility (Proxy)",
            color="#1f77b4",
            linewidth=1.5,
            zorder=1,
        )
    if pre_shock_fit_vol_df is not None and not pre_shock_fit_vol_df.empty:
        axes[1].plot(
            pre_shock_fit_vol_df.index,
            pre_shock_fit_vol_df["Fitted_Volatility"],
            label=f"{model_name} Fitted Volatility (In-Sample)",
            color="#ff7f0e",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
        )

    if sim_vol_stats is not None and not sim_vol_stats.empty:
        axes[1].plot(
            sim_vol_stats.index,
            sim_vol_stats["Median"],
            label=f"{model_name} Simulated Median Volatility",
            color="#2ca02c",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
        )

        if (
            "Quantile_05" in sim_vol_stats.columns
            and "Quantile_95" in sim_vol_stats.columns
        ):
            axes[1].fill_between(
                sim_vol_stats.index,
                sim_vol_stats["Quantile_05"],
                sim_vol_stats["Quantile_95"],
                color="#2ca02c",
                alpha=0.2,
                label="Simulated 5%-95% Vol Range",
                zorder=0,
            )

    axes[1].axvline(
        pd.to_datetime(shock_date),
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Shock ({shock_date})",
        zorder=3,
    )
    axes[1].set_ylabel("Annualized Volatility (%)", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_title(
        f"Volatility: Actual vs. {model_name} Fit & Simulation", fontsize=14
    )
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[0].get_xticklabels(), visible=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_naive_simulation_results(
    df_post_shock,
    sim_prices_stats_naive,
    ticker=config.TICKER,
    shock_date=config.SHOCK_DATE,
    approach_name="Naive Approach",
    vol_data=None,
    vol_label="Historical Volatility Used",
):
    print(f"\nGenerating plot for {approach_name}...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{approach_name} Simulation for {ticker}", fontsize=16)

    price_col = "Adj Close"
    vol_col = "Realized_Volatility_Daily"

    actual_prices_post_shock = df_post_shock[price_col].dropna()
    if not actual_prices_post_shock.empty:
        axes[0].plot(
            actual_prices_post_shock.index,
            actual_prices_post_shock,
            label="Actual Price (Post-Shock)",
            color="black",
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )
    else:
        print(f"Warning [{approach_name}]: No actual post-shock prices to plot.")

    if sim_prices_stats_naive is not None and not sim_prices_stats_naive.empty:
        median_prices_to_plot = sim_prices_stats_naive["Median"].dropna()
        if not median_prices_to_plot.empty:
            if median_prices_to_plot.min() <= 0:
                print(
                    f"Warning [{approach_name}]: Simulated median\
                        prices contain non-positive values. Log scale might hide data."
                )
            axes[0].plot(
                median_prices_to_plot.index,
                median_prices_to_plot,
                label=f"Median {approach_name} Sim.",
                color="darkcyan",
                linewidth=1.5,
                linestyle="--",
                zorder=2,
            )
            print(
                f"Debug [{approach_name}]: Plotted median prices.\
                Min: {median_prices_to_plot.min():.2f}, Max:\
                {median_prices_to_plot.max():.2f}"
            )
        else:
            print(
                f"Warning [{approach_name}]:\
                    Median simulated prices are empty after dropna."
            )

        if (
            "Quantile_05" in sim_prices_stats_naive.columns
            and "Quantile_95" in sim_prices_stats_naive.columns
        ):
            q05 = sim_prices_stats_naive["Quantile_05"].dropna()
            q95 = sim_prices_stats_naive["Quantile_95"].dropna()
            if not q05.empty and not q95.empty:
                common_idx = q05.index.intersection(q95.index)
                q05 = q05.loc[common_idx]
                q95 = q95.loc[common_idx]
                if not q05.empty:
                    if q05.min() <= 0:
                        print(
                            f"Warning [{approach_name}]: Simulated 5% quantile price\
                                contain non-positive values. Log scale might hide data."
                        )
                    axes[0].fill_between(
                        common_idx,
                        q05,
                        q95,
                        color="darkcyan",
                        alpha=0.15,
                        label="5-95% Range Sim.",
                        zorder=0,
                    )
                    print(f"Debug [{approach_name}]: Plotted price quantiles.")
                else:
                    print(
                        f"Warning [{approach_name}]: Price quantiles empty after\
                            intersection/dropna."
                    )
            else:
                print(
                    f"Warning [{approach_name}]: Price quantiles empty after initial\
                        dropna."
                )
        else:
            print(
                f"Warning [{approach_name}]: Quantile columns missing in\
                    sim_prices_stats_naive."
            )
    else:
        print(f"Warning [{approach_name}]: sim_prices_stats_naive is None or empty.")

    axes[0].axvline(
        pd.to_datetime(shock_date),
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Shock ({shock_date})",
        zorder=3,
    )
    axes[0].set_ylabel(f"{price_col} Price", fontsize=12)
    axes[0].set_title("Price Comparison: Actual vs. Simulation", fontsize=14)
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    actual_vol_post_shock = df_post_shock[vol_col].dropna()
    if not actual_vol_post_shock.empty:
        axes[1].plot(
            actual_vol_post_shock.index,
            actual_vol_post_shock,
            label="Actual Volatility (Post-Shock)",
            color="black",
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )
    else:
        print(f"Warning [{approach_name}]: No actual post-shock volatility to plot.")

    if vol_data is not None:
        if isinstance(vol_data, pd.Series):
            vol_to_plot = vol_data.dropna()
            if not vol_to_plot.empty:
                axes[1].plot(
                    vol_to_plot.index,
                    vol_to_plot,
                    label=vol_label,
                    color="darkcyan",
                    linewidth=1.5,
                    linestyle=":",
                    zorder=2,
                )
                print(
                    f"Debug [{approach_name}]: Plotted dynamic volatility.\
                        Min: {vol_to_plot.min():.2f}, Max: {vol_to_plot.max():.2f},\
                        Mean: {vol_to_plot.mean():.2f}"
                )
            else:
                print(
                    f"Warning [{approach_name}]: Dynamic volatility data\
                        (vol_data Series) is empty after dropna."
                )
        elif isinstance(vol_data, (int, float)):
            if pd.notna(vol_data):
                axes[1].axhline(
                    vol_data,
                    color="forestgreen",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{vol_label} ({vol_data:.2f}%)",
                    zorder=2,
                )
                print(
                    f"Debug [{approach_name}]: Plotted fixed volatility line\
                        at {vol_data:.2f}%."
                )
            else:
                print(
                    f"Warning [{approach_name}]: Fixed volatility value\
                        (vol_data float) is NaN."
                )
        else:
            print(
                f"Warning [{approach_name}]: vol_data has unexpected type:\
                    {type(vol_data)}"
            )
    else:
        print(f"Warning [{approach_name}]: vol_data is None.")

    axes[1].axvline(
        pd.to_datetime(shock_date),
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Shock ({shock_date})",
        zorder=3,
    )
    axes[1].set_ylabel("Annualized Volatility (%)", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_title("Volatility Comparison: Actual vs. Assumed/Used", fontsize=14)
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    try:
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        plt.setp(axes[0].get_xticklabels(), visible=False)
    except Exception as e:
        print(f"Warning [{approach_name}]: Error setting date formatters: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
