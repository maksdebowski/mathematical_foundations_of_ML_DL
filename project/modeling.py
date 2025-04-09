import pandas as pd
import numpy as np
from arch import arch_model
import config


def fit_garch_model(
    returns, p=1, q=1, o=0, dist=config.MODEL_DISTRIBUTION, mean="Constant"
):
    scaled_returns = returns * 100

    try:
        model = arch_model(
            scaled_returns, mean=mean, vol="Garch", p=p, o=o, q=q, dist=dist
        )
        results = model.fit(update_freq=0, disp="off")
        return results
    except Exception as killmepls:
        print(
            f"Error fitting {('GJR-' if o > 0 else '')}GARCH({p},{o},{q}): {killmepls}"
        )
        return None


def select_best_garch_model(
    returns,
    max_p=config.MAX_GARCH_P,
    max_q=config.MAX_GARCH_Q,
    criterion="BIC",
    model_type="GARCH",
):
    if returns.isnull().any():
        returns = returns[~returns.isnull()]
    results_list = []
    print(f"\n--- {model_type} Model Selection with criterion {criterion} ---")
    o = config.GJR_ORDER_O if model_type == "GARCH" else 0
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            print(f"Fitting {model_type}({p},{o},{q})", end=" ")
            results = fit_garch_model(
                returns, p=p, q=q, o=o, dist=config.MODEL_DISTRIBUTION
            )

            if results:
                # stationarity (more or less)
                alpha_coeffs = [f"alpha[{i}]" for i in range(1, p + 1)]
                beta_coeffs = [f"beta[{i}]" for i in range(1, q + 1)]
                gamma_coeffs = [f"gamma[{i}]" for i in range(1, o + 1)] if o > 0 else []

                alpha_sum = results.params[alpha_coeffs].sum() if alpha_coeffs else 0
                beta_sum = results.params[beta_coeffs].sum() if beta_coeffs else 0
                gamma_sum = (  # noqa
                    results.params[gamma_coeffs].sum() if gamma_coeffs else 0
                )

                alpha_beta_sum = alpha_sum + beta_sum
                is_stationary = alpha_beta_sum < 1

                results_list.append(
                    {
                        "p": p,
                        "q": q,
                        "o": o,
                        "ModelType": model_type,
                        "AIC": results.aic,
                        "BIC": results.bic,
                        "LogLikelihood": results.loglikelihood,
                        "alpha_beta_sum": alpha_beta_sum,
                        "Stationary": is_stationary,
                        "Converged": results.convergence_flag == 0,
                        "ModelObject": results,
                    }
                )
                print(f"BIC: {results.bic:.2f}. Stationary(approx): {is_stationary}")
                if not is_stationary:
                    print(
                        f" {model_type}({p},{o},{q}) alpha+beta sum >= 1\
                            ({alpha_beta_sum})"
                    )
            else:
                print("Failed.")
                results_list.append(
                    {
                        "p": p,
                        "q": q,
                        "o": o,
                        "ModelType": model_type,
                        "AIC": np.inf,
                        "BIC": np.inf,
                        "LogLikelihood": -np.inf,
                        "alpha_beta_sum": np.nan,
                        "Stationary": False,
                        "Converged": False,
                        "ModelObject": None,
                    }
                )

        results_df = pd.DataFrame(results_list)
        print(
            results_df.sort_values(by=criterion)[
                ["p", "o", "q", "AIC", "BIC", "Stationary", "Converged"]
            ]
        )

        valid_models = results_df[
            results_df["Converged"] & results_df["Stationary"]
        ].copy()

        if valid_models.empty:
            print(f"No valid {model_type} found.")
            converged_models = results_df[results_df["Converged"]].copy()
            if not converged_models.empty:
                best_model_row = converged_models.loc[
                    converged_models[criterion].idxmin()
                ]
                print(
                    f"Best model: {best_model_row['ModelType']}\
                        (but potentially non-stationary)"
                )
            else:
                print(f"No valid {model_type} found.")
                return None, None
        else:
            best_model_row = valid_models.loc[valid_models[criterion].idxmin()]
            print(
                f"\nSelected Best Valid {model_type}\
                    Model based on {criterion}: {model_type}"
                f"({int(best_model_row['p'])},{int(best_model_row['o'])},\
                    {int(best_model_row['q'])})"
            )
            best_model_results = best_model_row["ModelObject"]
            best_params = {
                "p": int(best_model_row["p"]),
                "q": int(best_model_row["q"]),
                "o": int(best_model_row["o"]),
            }
            print("\nBest Model Summary:")
            print(best_model_results.summary())

            return best_model_results, best_params


def calculate_fitted_values(
    model_results,
    returns,
    pre_shock_df,
    annualization_factor=config.ANNUALIZATION_FACTOR,
):

    if model_results is None:
        print("Error: Model results are None. Cannot calculate fitted values.")
        return None, None
    if returns.isnull().any():
        returns = returns.dropna()
    if returns.empty:
        print("Error: Returns series is empty after dropping NaNs.")
        return None, None

    try:
        fitted_vol_scaled_daily = model_results.conditional_volatility
        fitted_vol_annualized = fitted_vol_scaled_daily * np.sqrt(annualization_factor)

        num_skipped_vol = len(returns) - len(fitted_vol_annualized)
        if num_skipped_vol < 0:
            print(
                "Warning: More fitted volatility values than returns. Check alignment."
            )
            fitted_vol_index = returns.index[abs(num_skipped_vol) :]
            fitted_vol_annualized = fitted_vol_annualized.iloc[-len(fitted_vol_index) :]
        elif num_skipped_vol > 0:
            fitted_vol_index = returns.index[num_skipped_vol:]
        else:
            fitted_vol_index = returns.index

        pre_shock_fit_vol_df = pd.DataFrame(
            {"Fitted_Volatility": fitted_vol_annualized.values}, index=fitted_vol_index
        )
        print(
            f"Fitted volatility calculated, aligned with returns index from\
                {fitted_vol_index[0].date()}."
        )

    except Exception as e:
        print(f"Error calculating fitted volatility: {e}")
        pre_shock_fit_vol_df = None

    try:
        fitted_scaled_residuals = model_results.resid
        fitted_mu_scaled = model_results.params["mu"]

        fitted_log_returns = (fitted_mu_scaled + fitted_scaled_residuals) / 100.0

        num_skipped_resid = len(returns) - len(fitted_log_returns)
        if num_skipped_resid < 0:
            print(
                "Warning: More fitted log returns than original returns.\
                    Check alignment."
            )
            fitted_price_index = returns.index[abs(num_skipped_resid) :]
            fitted_log_returns = fitted_log_returns.iloc[-len(fitted_price_index) :]
        elif num_skipped_resid > 0:
            fitted_price_index = returns.index[num_skipped_resid:]
        else:
            fitted_price_index = returns.index

        if fitted_price_index.empty:
            print("Error: Cannot determine index for fitted prices.")
            return None, pre_shock_fit_vol_df

        first_fit_date = fitted_price_index[0]
        try:
            loc_in_preshock = pre_shock_df.index.get_loc(first_fit_date)
            if loc_in_preshock > 0:
                start_price_date = pre_shock_df.index[loc_in_preshock - 1]
                start_price = pre_shock_df.loc[start_price_date, "Adj Close"]
                print(
                    f"Starting price for fitted values reconstruction:\
                        {start_price:.2f} on {start_price_date.date()}"
                )
            else:
                print(
                    "Warning: First fitted return is the first date in pre_shock_df.\
                        Estimating start price."
                )
                first_actual_price = pre_shock_df.loc[first_fit_date, "Adj Close"]
                first_log_ret = fitted_log_returns.iloc[0]
                start_price = first_actual_price / np.exp(first_log_ret)
                print(f"Estimated starting price: {start_price:.2f}")
        except KeyError:
            print(
                f"Error: Date {first_fit_date} not found in pre_shock_df index.\
                    Cannot determine start price."
            )
            return None, pre_shock_fit_vol_df

        fitted_prices_list = []
        current_price = start_price
        for log_ret in fitted_log_returns:
            current_price = current_price * np.exp(log_ret)
            fitted_prices_list.append(current_price)

        pre_shock_fit_price_df = pd.DataFrame(
            {"Fitted_Price": fitted_prices_list}, index=fitted_price_index
        )
        print(
            f"Fitted prices calculated, length: {len(pre_shock_fit_price_df)},\
                starting from index date {fitted_price_index[0].date()}"
        )

    except Exception as e:
        import traceback

        print(f"Error calculating fitted prices: {e}")
        traceback.print_exc()
        pre_shock_fit_price_df = None

    return pre_shock_fit_price_df, pre_shock_fit_vol_df
