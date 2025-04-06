import pandas as pd
import numpy as np
from arch import arch_model
import config

def fit_garch_model(returns, p=1, q = 1, o = 0, dist = config.MODEL_DISTRIBUTION,
                    mean = 'Constant'):
    scaled_returns = returns * 100

    try:
        model = arch_model(scaled_returns, mean=mean, vol='Garch',
                           p=p, o=o, q=q, dist=dist)
        results = model.fit(update_freq = 0, disp = 'off')
        return results
    except Exception as killmepls:
        print(f"Error fitting {('GJR-' if o > 0 else '')}GARCH({p},{o},{q}): {killmepls}")
        return None

def select_best_garch_model(returns, max_p = config.MAX_GARCH_P,max_q = config.MAX_GARCH_Q,
                            criterion = 'BIC', model_type = 'GARCH'):
    if returns.isnull().any():
        returns = returns[~returns.isnull()]
    results_list = []
    print(f"\n--- {model_type} Model Selection with criterion {criterion} ---")
    o = config.GJR_ORDER_O if model_type == 'GARCH' else 0
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            print(f"Fitting {model_type}({p},{o},{q})", end = " ")
            results = fit_garch_model(returns, p=p, q=q, o=o, dist=config.MODEL_DISTRIBUTION)

            if results:
                # stationarity (more or less)
                alpha_coeffs = [f'alpha[{i}]' for i in range(1, p + 1)]
                beta_coeffs = [f'beta[{i}]' for i in range(1, q + 1)]
                gamma_coeffs = [f'gamma[{i}]' for i in range(1, o + 1)] if o > 0 else []

                alpha_sum = results.params[alpha_coeffs].sum() if alpha_coeffs else 0
                beta_sum = results.params[beta_coeffs].sum() if beta_coeffs else 0
                gamma_sum = results.params[gamma_coeffs].sum() if gamma_coeffs else 0

                alpha_beta_sum = alpha_sum + beta_sum
                is_stationary = alpha_beta_sum < 1

                results_list.append({
                    'p': p, 'q': q, 'o': o, 'ModelType': model_type,
                    'AIC': results.aic, 'BIC': results.bic,
                    'LogLikelihood': results.loglikelihood,
                    'alpha_beta_sum': alpha_beta_sum,
                    'Stationary': is_stationary,
                    'Converged': results.convergence_flag == 0,
                    'ModelObject': results
                })
                print(f"BIC: {results.bic:.2f}. Stationary(approx): {is_stationary}")
                if not is_stationary:
                    print(f" {model_type}({p},{o},{q}) alpha+beta sum >= 1 ({alpha_beta_sum})")
            else:
                print("Failed.")
                results_list.append({
                    'p': p, 'q': q, 'o': o, 'ModelType': model_type,
                    'AIC': np.inf, 'BIC': np.inf, 'LogLikelihood': -np.inf,
                    'alpha_beta_sum': np.nan, 'Stationary': False,
                    'Converged': False, 'ModelObject': None
                })

        results_df = pd.DataFrame(results_list)
        print(results_df.sort_values(by=criterion)[['p', 'o', 'q', 'AIC', 'BIC', 'Stationary', 'Converged']])

        valid_models = results_df[results_df['Converged'] & results_df['Stationary']].copy()

        if valid_models.empty:
            print(f"No valid {model_type} found.")
            converged_models = results_df[results_df['Converged']].copy()
            if not converged_models.empty:
                best_model_row = converged_models.loc[converged_models[criterion].idxmin()]
                print(f"Best model: {best_model_row['ModelType']} (but potentially non-stationary)")
            else:
                print(f"No valid {model_type} found.")
                return None, None
        else:
            best_model_row = valid_models.loc[valid_models[criterion].idxmin()]
            print(f"\nSelected Best Valid {model_type} Model based on {criterion}: {model_type}"
                  f"({int(best_model_row['p'])},{int(best_model_row['o'])},{int(best_model_row['q'])})")
            best_model_results = best_model_row['ModelObject']
            best_params = {'p': int(best_model_row['p']), 'q': int(best_model_row['q']),
                           'o': int(best_model_row['o'])}
            print("\nBest Model Summary:")
            print(best_model_results.summary())

            return best_model_results, best_params

def calculate_fitted_values(model_results, returns, pre_shock_df,
                            annualization_factor=config.ANNUALIZATION_FACTOR):
    if returns.isnull().any():
        returns = returns.dropna()

    fitted_vol_scaled_daily = model_results.conditional_volatility
    fitted_vol_annualized = fitted_vol_scaled_daily * np.sqrt(annualization_factor)

    num_skipped_vol = len(returns) - len(fitted_vol_annualized)
    fitted_vol_index = returns.index[num_skipped_vol:]
    pre_shock_fit_vol_df = pd.DataFrame({'Fitted_Volatility': fitted_vol_annualized},
                                        index = fitted_vol_index)
    print(f"Fitted volatility calculated, starting from index {num_skipped_vol}"
          f"({fitted_vol_index[0].date()})")

    fitted_scaled_residuals = model_results.resid
    fitted_std_residuals = model_results.std_resid

    fitted_mu_scaled = model_results.params['mu']
    fitted_log_returns = (fitted_mu_scaled + fitted_scaled_residuals) / 100.0

    num_skipped_resid = len(returns) - len(fitted_log_returns)
    fitted_price_index = returns.index[num_skipped_resid:]

    if num_skipped_resid > 0:
        returns_start_date = returns.index[0]
        pre_shock_df_returns_start_idx = pre_shock_df.index.get_loc(returns_start_date)
        fit_start_idx_in_df = pre_shock_df_returns_start_idx + num_skipped_resid

        if fit_start_idx_in_df > 0:
            start_price_index = fit_start_idx_in_df - 1
            start_price = pre_shock_df['Adj Close'].iloc[start_price_index]
            print(
                f"Starting price for fitted values reconstruction: {start_price:.2f} on {pre_shock_df.index[start_price_index].date()}")
        else:
            print("Warning: Fit starts very early. Using first available price as base for reconstruction.")
            start_price = pre_shock_df['Adj Close'].iloc[0]
    else:
        if pre_shock_df.index.get_loc(returns.index[0]) > 0:
            start_price = pre_shock_df['Adj Close'].iloc[pre_shock_df.index.get_loc(returns.index[0]) - 1]
        else:
            start_price = pre_shock_df['Adj Close'].iloc[0] / np.exp(
                fitted_log_returns.iloc[0])

    fitted_prices_list = []
    current_price = start_price
    for log_ret in fitted_log_returns:
        current_price = current_price * np.exp(log_ret)
        fitted_prices_list.append(current_price)

        pre_shock_fit_price_df = pd.DataFrame({'Fitted_Price': fitted_prices_list}, index = fitted_price_index)
        print(f"Fitted prices calculated, starting from index {num_skipped_resid} ({fitted_price_index[0].date()})")

        return pre_shock_fit_price_df, pre_shock_fit_vol_df

if __name__ == "__main__":
    # Create dummy data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=800, freq='B')
    true_vol = np.zeros(len(dates))
    true_ret = np.zeros(len(dates))
    true_vol[0] = 0.015 # Daily std dev
    mu = 0.0001
    omega = 0.000002
    alpha = 0.1
    beta = 0.88
    for t in range(1, len(dates)):
        true_vol[t] = np.sqrt(omega + alpha * true_ret[t-1]**2 + beta * true_vol[t-1]**2)
        true_ret[t] = mu + np.random.normal(0, true_vol[t])

    dummy_df = pd.DataFrame({'Log_Return': true_ret}, index=dates)
    dummy_df['Adj Close'] = 100 * np.exp(dummy_df['Log_Return'].cumsum()) # Create dummy prices

    print("Testing GARCH selection...")
    best_garch, params_garch = select_best_garch_model(dummy_df['Log_Return'], max_p=2, max_q=2, model_type='GARCH')

    if best_garch:
        print("\nTesting GJR-GARCH selection...")
        best_gjr, params_gjr = select_best_garch_model(dummy_df['Log_Return'], max_p=2, max_q=2, model_type='GJR')

        if best_gjr:
             print("\nTesting fitted value calculation (using best GJR)...")
             # Need a pre_shock_df equivalent for the dummy data
             fit_price, fit_vol = calculate_fitted_values(best_gjr, dummy_df['Log_Return'], dummy_df)
             print("\nFitted Prices Head:")
             print(fit_price.head())
             print("\nFitted Volatility Head:")
             print(fit_vol.head())

