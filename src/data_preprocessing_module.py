import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split # No longer needed here for example
import yfinance as yf
from statsmodels.tsa.stattools import adfuller # For stationarity testing
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt
import os
import pickle
from datetime import date # For cache invalidation
from arch import arch_model # For GARCH models

# Define a cache directory
CACHE_DIR = "data_cache"

# --- Module 1: Data Acquisition and Preprocessing ---

class DataPreprocessor:
    def __init__(self, stock_ticker='AEL', years_of_data=10, random_seed=42, lasso_alpha=0.005, use_differencing=False, auto_differencing_adf=False, adf_significance_level=0.05, use_log_transform=False): # Default alpha
        self.stock_ticker = stock_ticker
        self.years_of_data = years_of_data
        self.random_seed = random_seed
        self.lasso_alpha = lasso_alpha
        self.use_differencing = use_differencing # Manual override for differencing
        self.auto_differencing_adf = auto_differencing_adf # ADF-based auto differencing
        self.adf_significance_level = adf_significance_level
        self.use_log_transform = use_log_transform # Whether to apply log transform to target
        self.applied_differencing_order = 0 # To store how many orders of differencing were applied
        self.values_for_inverse_differencing = [] # To store values needed for reconstruction
        # self.log_transform_applied will be set in preprocess()
        self.log_transform_applied_to_target = False


        np.random.seed(self.random_seed)
        os.makedirs(CACHE_DIR, exist_ok=True) # Ensure cache directory exists

    def _check_stationarity(self, series, significance_level=0.05):
        """Performs ADF test and returns True if stationary, False otherwise."""
        if series.empty or series.isnull().all(): # Cannot test empty or all-NaN series
            print("Warning: Series is empty or all NaN, cannot perform ADF test. Assuming non-stationary.")
            return False

        # Drop NaNs that might result from prior operations like .diff() before testing
        series_cleaned = series.dropna()
        if len(series_cleaned) < 10: # Heuristic: ADF test needs a minimum number of samples
            print(f"Warning: Series has less than 10 non-NaN values ({len(series_cleaned)}), ADF test might be unreliable. Assuming non-stationary.")
            return False

        try:
            adf_result = adfuller(series_cleaned)
            p_value = adf_result[1]
            # print(f"ADF Test: Test Statistic: {adf_result[0]}, p-value: {p_value}")
            if p_value <= significance_level:
                # print(f"  Series is stationary (p-value {p_value:.4f} <= {significance_level}).")
                return True  # Reject null hypothesis -> series is stationary
            else:
                # print(f"  Series is non-stationary (p-value {p_value:.4f} > {significance_level}).")
                return False # Fail to reject null hypothesis -> series is non-stationary
        except Exception as e:
            print(f"Error during ADF test: {e}. Assuming non-stationary.")
            return False


    def _download_yfinance_data(self):
        # Simplified cache naming: ticker_years_raw.pkl
        # More robust caching might include start/end dates in filename if they are variable beyond 'years_of_data'
        cache_filename = os.path.join(CACHE_DIR, f"{self.stock_ticker}_{self.years_of_data}_raw_data_{date.today()}.pkl")

        if os.path.exists(cache_filename):
            print(f"Loading raw data for {self.stock_ticker} from cache: {cache_filename}")
            try:
                df = pd.read_pickle(cache_filename)
                print("Successfully loaded raw data from cache.")
                # Basic validation: check if DataFrame is empty or if columns are as expected
                if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    print("Cached raw data is invalid. Re-downloading.")
                    os.remove(cache_filename) # Remove invalid cache
                    raise FileNotFoundError # Trigger re-download
                return df
            except Exception as e:
                print(f"Error loading raw data from cache: {e}. Re-downloading.")
                if os.path.exists(cache_filename): # Ensure removal if loading failed
                    os.remove(cache_filename)

        print(f"Downloading {self.years_of_data} years of stock data for {self.stock_ticker} from Yahoo Finance...")
        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=self.years_of_data * 365.25)

        df = yf.download(self.stock_ticker, start=start_date_dt.strftime('%Y-%m-%d'), end=end_date_dt.strftime('%Y-%m-%d'))

        if df.empty:
            raise ValueError(f"No data downloaded for ticker {self.stock_ticker}. Check ticker symbol or date range.")

        df = df.rename(columns={'Adj Close': 'Adj_Close'})
        print(f"Downloaded data shape: {df.shape}")

        # Add time-based features before caching raw data with them
        print("Adding time-based features...")
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        print(f"Shape after adding time-based features: {df.shape}")

        try:
            print(f"Saving raw data for {self.stock_ticker} to cache: {cache_filename}")
            df.to_pickle(cache_filename)
        except Exception as e:
            print(f"Error saving raw data to cache: {e}")

        return df

    def _calculate_technical_indicators(self, df_raw): # Renamed input for clarity
        print("Calculating technical indicators...")
        # Use a copy to avoid modifying the cached raw DataFrame directly if df_raw is a reference
        df = df_raw.copy()

        # Ensure columns are correct type
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        # SMAs
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = ta.trend.SMAIndicator(close=df['Close'].squeeze(), window=window, fillna=True).sma_indicator()

        # EMAs
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'EMA_{window}'] = ta.trend.EMAIndicator(close=df['Close'].squeeze(), window=window, fillna=True).ema_indicator()

        # MACD
        macd_indicator = ta.trend.MACD(close=df['Close'].squeeze(), fillna=True)
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        df['MACD_Diff'] = macd_indicator.macd_diff() # Histogram part

        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'].squeeze(), window=14, fillna=True).rsi()

        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(close=df['Close'].squeeze(), window=20, window_dev=2, fillna=True)
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Middle'] = bb_indicator.bollinger_mavg()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        df['BB_pband'] = bb_indicator.bollinger_pband() # Percentage Bandwidth
        df['BB_wband'] = bb_indicator.bollinger_wband() # Width Bandwidth

        # ATR
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'].squeeze(), low=df['Low'].squeeze(), close=df['Close'].squeeze(), window=14, fillna=True).average_true_range()

        # OBV
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'].squeeze(), volume=df['Volume'].squeeze(), fillna=True).on_balance_volume()

        # Stochastic Oscillator
        stoch_indicator = ta.momentum.StochasticOscillator(high=df['High'].squeeze(), low=df['Low'].squeeze(), close=df['Close'].squeeze(), window=14, smooth_window=3, fillna=True)
        df['Stoch_K'] = stoch_indicator.stoch()
        df['Stoch_D'] = stoch_indicator.stoch_signal()

        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(high=df['High'].squeeze(), low=df['Low'].squeeze(), close=df['Close'].squeeze(), lbp=14, fillna=True).williams_r()

        # CCI
        df['CCI'] = ta.trend.CCIIndicator(high=df['High'].squeeze(), low=df['Low'].squeeze(), close=df['Close'].squeeze(), window=20, constant=0.015, fillna=True).cci()

        # ROC (Rate of Change)
        df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'].squeeze(), window=12, fillna=True).roc()

        # ADX (Average Directional Index)
        adx_indicator = ta.trend.ADXIndicator(
            high=df['High'].squeeze(),
            low=df['Low'].squeeze(),
            close=df['Close'].squeeze(),
            window=14,
            fillna=True
        )
        df['ADX'] = adx_indicator.adx()
        df['ADX_Pos'] = adx_indicator.adx_pos() # Positive Directional Indicator (+DI)
        df['ADX_Neg'] = adx_indicator.adx_neg() # Negative Directional Indicator (-DI)

        # Volatility Features
        # Normalized ATR
        if 'ATR' in df.columns and 'Close' in df.columns:
            # Ensure 'ATR' and 'Close' are Series for the division
            atr_series = df['ATR'].squeeze() if isinstance(df['ATR'], pd.DataFrame) else df['ATR']
            close_series = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

            # Avoid division by zero or very small close prices if necessary
            df['ATR_Normalized'] = atr_series / close_series.replace(0, np.nan)
            # Replace fillna with method
            df['ATR_Normalized'] = df['ATR_Normalized'].bfill()
            df['ATR_Normalized'] = df['ATR_Normalized'].ffill()

        # Rolling Standard Deviation of Returns
        df['Returns'] = df['Close'].pct_change() # Calculate daily percentage returns
        for window in [5, 10, 20, 60]: # Common windows for volatility
            df[f'Volatility_Ret_{window}D'] = df['Returns'].rolling(window=window, min_periods=1).std() * np.sqrt(window) # Annualized for comparison, or just raw std
            # Using min_periods=1 to avoid NaNs at the start, but this means early values are less representative.
            # Alternatively, only calculate if full window is available, then dropna later.
            # For now, let's keep min_periods=1 and let dropna handle it.
            # The multiplication by sqrt(window) is a common way to scale rolling std dev if returns are daily and window is in days,
            # often used when aiming for an "annualized" volatility, but here it just scales it. Let's stick to raw std for now for simplicity as features.
            df[f'Volatility_Ret_{window}D_Raw'] = df['Returns'].rolling(window=window, min_periods=1).std()


        # Clean up Returns column as it's intermediate
        df.drop(columns=['Returns'], inplace=True, errors='ignore')

        # GARCH Model for Volatility Forecasting
        # Calculate daily returns for GARCH model (use 'Close' price)
        # Ensure 'Close' is a Series and exists
        if 'Close' in df.columns:
            garch_returns = df['Close'].pct_change().dropna() * 100 # Multiply by 100 for GARCH
            if not garch_returns.empty:
                try:
                    # Fit GARCH(1,1) model - common choice
                    # Use 'Constant' mean, 'GARCH' vol, p=1, q=1.
                    # Turn off display output from arch_model
                    garch = arch_model(garch_returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal') # Removed show_warning from constructor
                    garch_results = garch.fit(disp='off', show_warning=False) # show_warning is valid for fit method

                    # Forecast conditional volatility (1-step ahead)
                    # The forecast object contains mean, variance, and residual variance.
                    # We are interested in the conditional variance.
                    forecast = garch_results.forecast(horizon=1, reindex=False) # reindex=False to align with end of sample
                    # The variance is typically forecast.volatility.values[-1,0]^2 or forecast.variance.values[-1,0]
                    # The .forecast() method returns an ARCHModelForecast object.
                    # The conditional variance is in forecast.variance
                    # We need to align this with the original DataFrame index.
                    # The forecast is for the next period, so shift variance back to align with current day's features
                    # The variance is for t+1, based on info at t. So, it's a feature for predicting t+1 price.
                    # We shift it to align with the day 't' features.
                    df['GARCH_Volatility'] = np.nan # Initialize column
                    # The variance forecast is for the period *after* the last observation in garch_returns.
                    # So, garch_results.conditional_volatility is for in-sample.
                    # forecast.variance.iloc[0] would be the first out-of-sample forecast.
                    # We need in-sample conditional volatility as a feature.
                    # garch_results.conditional_volatility is already aligned with garch_returns's index.
                    # Use .loc for safe assignment to avoid SettingWithCopyWarning if df is a slice
                    df.loc[garch_returns.index, 'GARCH_Volatility'] = garch_results.conditional_volatility

                    # Replace fillna with method and assign back
                    df['GARCH_Volatility'] = df['GARCH_Volatility'].bfill()
                    df['GARCH_Volatility'] = df['GARCH_Volatility'].ffill()
                    print("GARCH Volatility feature calculated and added.")
                except Exception as e:
                    print(f"Error calculating GARCH volatility: {e}. Proceeding without GARCH feature.")
                    df['GARCH_Volatility'] = 0 # Add a placeholder column if GARCH fails
            else:
                print("Not enough return data to calculate GARCH volatility. Proceeding without GARCH feature.")
                df['GARCH_Volatility'] = 0 # Placeholder
        else:
            print("Close column not found, cannot calculate GARCH returns. Proceeding without GARCH feature.")
            df['GARCH_Volatility'] = 0 # Placeholder


        print(f"Shape after adding all indicators and new volatility features: {df.shape}")
        return df

    def _apply_lasso_feature_selection(self, df, target_column='Close'): # Alpha is now instance variable
        print(f"Applying LASSO feature selection with alpha={self.lasso_alpha}...")
        # Drop rows with NaN values introduced by rolling windows or yfinance missing days
        # It's critical to drop NaNs *before* splitting features and target
        df_cleaned = df.dropna().copy() # Use .copy() to avoid SettingWithCopyWarning later
        # df_cleaned = df.fillna(method='bfill').fillna(method='ffill') # Alternative: fill NaNs

        # Check for MultiIndex columns and flatten if necessary
        if isinstance(df_cleaned.columns, pd.MultiIndex):
            print("Warning: DataFrame columns are a MultiIndex. Flattening columns.")

            new_columns = []
            updated_target_column_local = target_column # Keep track if target_column name changes

            for col_tuple in df_cleaned.columns.values:
                # col_tuple could be like ('Close', '^AEX') or just 'SMA_5' if it wasn't a tuple
                if isinstance(col_tuple, tuple):
                    new_col_name = '_'.join(str(part) for part in col_tuple if str(part)).strip('_') # Ensure all parts are strings
                    if col_tuple[0] == target_column: # If the first part of tuple is 'Close'
                        updated_target_column_local = new_col_name
                else:
                    new_col_name = str(col_tuple) # It's already a simple string name
                new_columns.append(new_col_name)

            df_cleaned.columns = new_columns

            if target_column != updated_target_column_local:
                print(f"Target column name updated from '{target_column}' to '{updated_target_column_local}' due to column flattening.")
                target_column = updated_target_column_local # Update for the rest of this method

            print(f"Flattened columns: {df_cleaned.columns.tolist()}")

        if df_cleaned.empty:
            print("Warning: DataFrame is empty after dropping NaNs. Cannot apply LASSO.")
            return pd.DataFrame(), [], df.columns.drop(target_column).tolist() # Return all features if empty

        features_df = df_cleaned.drop(columns=[target_column], errors='ignore')
        target = df_cleaned[target_column]

        if features_df.empty:
            print("Warning: No features left after dropping target. Cannot apply LASSO.")
            return df_cleaned[[target_column]], [], []


        # It's good practice to scale features before LASSO
        # Note: This scaler is only for LASSO. The main data scaling for the model happens later.
        scaler_lasso = MinMaxScaler()
        scaled_features = scaler_lasso.fit_transform(features_df)

        lasso = Lasso(alpha=self.lasso_alpha, random_state=self.random_seed, max_iter=10000) # Use instance alpha
        lasso.fit(scaled_features, target)

        selected_features_mask = lasso.coef_ != 0

        if not np.any(selected_features_mask):
            print(f"Warning: LASSO with alpha={self.lasso_alpha} selected 0 features. This might be too high or data needs review.")
            print("Returning all non-NaN features as a fallback for now.")
            # Fallback: return all features from df_cleaned if LASSO selects none
            selected_feature_names = features_df.columns.tolist()
        else:
            selected_feature_names = features_df.columns[selected_features_mask].tolist()
            print(f"LASSO selected {len(selected_feature_names)} features out of {len(features_df.columns)}.")
            # print(f"Selected features: {selected_feature_names}")


        # Return the cleaned DataFrame but subsetted to selected features + target
        # This ensures that the scaler for the model trains on the correct, LASSO-selected data

        # Defensive checks
        if target_column not in df_cleaned.columns:
            raise ValueError(f"Target column '{target_column}' not found in df_cleaned. Columns: {df_cleaned.columns.tolist()}")

        for feature_name in selected_feature_names:
            if feature_name not in df_cleaned.columns:
                raise ValueError(f"Selected feature '{feature_name}' not found in df_cleaned. Columns: {df_cleaned.columns.tolist()}")

        # Ensure target_column is not in selected_feature_names before combining
        if target_column in selected_feature_names:
            print(f"Warning: Target column '{target_column}' was also selected by LASSO. Removing from features list for final DataFrame construction.")
            selected_feature_names = [name for name in selected_feature_names if name != target_column]

        # Construct the final DataFrame for scaling
        final_df_for_scaling = df_cleaned[selected_feature_names + [target_column]]

        # Return the original df_cleaned (with all features before LASSO but after NaN drop)
        # as well, for potential volatility analysis on original features like ATR.
        # Also return the list of selected_feature_names and the determined target_column name.
        return final_df_for_scaling, selected_feature_names, lasso, df_cleaned, target_column

    def preprocess(self):
        # Step 1: Get raw data (potentially from cache)
        stock_data_downloaded = self._download_yfinance_data()

        # Step 2: Calculate technical indicators (potentially from cache)
        # The cache filename for indicators depends on the raw data.
        # Using stock_ticker, years_of_data, and today's date for the cache key.
        # This assumes that if raw data for today is updated (e.g. new download),
        # the indicators should be recalculated.

        indicators_cache_filename = os.path.join(CACHE_DIR, f"{self.stock_ticker}_{self.years_of_data}_indicators_{date.today()}.pkl")

        if os.path.exists(indicators_cache_filename):
            print(f"Loading technical indicators for {self.stock_ticker} from cache: {indicators_cache_filename}")
            try:
                stock_data_with_indicators = pd.read_pickle(indicators_cache_filename)
                print("Successfully loaded technical indicators from cache.")
                # Basic validation: check if DataFrame is empty or if an expected indicator column (e.g., 'ATR') is present.
                if stock_data_with_indicators.empty or 'ATR' not in stock_data_with_indicators.columns:
                     print("Cached indicators data seems invalid (empty or missing 'ATR'). Re-calculating.")
                     os.remove(indicators_cache_filename) # Remove invalid cache
                     raise FileNotFoundError # Trigger re-calculation
            except Exception as e:
                print(f"Error loading indicators from cache: {e}. Re-calculating.")
                if os.path.exists(indicators_cache_filename): # Ensure removal if loading failed
                     os.remove(indicators_cache_filename)
                # Use .copy() to avoid issues if stock_data_downloaded is used elsewhere or is a slice
                stock_data_with_indicators = self._calculate_technical_indicators(stock_data_downloaded.copy())
                try:
                    print(f"Saving technical indicators for {self.stock_ticker} to cache: {indicators_cache_filename}")
                    stock_data_with_indicators.to_pickle(indicators_cache_filename)
                except Exception as e_save:
                    print(f"Error saving technical indicators to cache: {e_save}")
        else:
            print("Calculating technical indicators (cache not found).")
            # Use .copy() here as well
            stock_data_with_indicators = self._calculate_technical_indicators(stock_data_downloaded.copy())
            try:
                print(f"Saving technical indicators for {self.stock_ticker} to cache: {indicators_cache_filename}")
                stock_data_with_indicators.to_pickle(indicators_cache_filename)
            except Exception as e_save:
                print(f"Error saving technical indicators to cache: {e_save}")

        # Apply LASSO feature selection
        # It now returns:
        # 1. processed_data_for_scaling (df with selected features + target, ready for scaling)
        # 2. selected_feature_names (list of names of selected features)
        # 3. lasso_model (the fitted LASSO model itself)
        # 4. df_with_all_indicators_cleaned (df with all indicators, NaNs dropped, before LASSO selection - useful for volatility analysis)
        # Pass self.lasso_alpha to _apply_lasso_feature_selection implicitly by using it from self
        # Determine target column name (it might be 'Close' or 'Close_Ticker' if flattened)
        # For simplicity, assume 'Close' is the primary target. If it gets renamed, LASSO selection handles it.
        raw_target_column = 'Close' # The original name before any potential flattening

        processed_data_for_scaling, selected_features_names, lasso_model, df_with_all_indicators_cleaned, final_target_column_name = \
            self._apply_lasso_feature_selection(stock_data_with_indicators.copy(), target_column=raw_target_column)

        self.lasso_model = lasso_model # Store for access if needed
        self.final_target_column_name = final_target_column_name # Store for internal use if needed, and for returning

        if processed_data_for_scaling.empty:
            print("Preprocessing failed: No data after feature selection.")
            return pd.DataFrame(), None, [], pd.DataFrame(), None, None, None, False # Added log_transform_applied_to_target

        target_col_name_in_df = self.final_target_column_name # Use the determined target name

        # --- Log Transformation (Optional, applied before differencing and scaling) ---
        self.log_transform_applied_to_target = False
        if self.use_log_transform:
            print(f"Applying log1p transformation to target column: {target_col_name_in_df}")
            # Ensure target values are positive before log transform, handle 0s if any (log1p handles 0)
            if (processed_data_for_scaling[target_col_name_in_df] < 0).any():
                print(f"Warning: Target column {target_col_name_in_df} has negative values. Log transform might not be appropriate or will produce NaNs.")
            processed_data_for_scaling.loc[:, target_col_name_in_df] = np.log1p(processed_data_for_scaling[target_col_name_in_df])
            self.log_transform_applied_to_target = True
            # Also apply to df_with_all_indicators_cleaned for consistency if this df is used for inverse transform reference later.
            if not df_with_all_indicators_cleaned.empty:
                 if (df_with_all_indicators_cleaned[target_col_name_in_df] < 0).any():
                      print(f"Warning: Target column {target_col_name_in_df} in df_all_indicators has negative values for log transform.")
                 df_with_all_indicators_cleaned.loc[:, target_col_name_in_df] = np.log1p(df_with_all_indicators_cleaned[target_col_name_in_df])


        # --- Differencing (Optional) ---
        self.applied_differencing_order = 0
        self.values_for_inverse_differencing = []

        current_target_series = processed_data_for_scaling[target_col_name_in_df].copy()

        if self.auto_differencing_adf:
            print(f"Auto-differencing enabled for target column: {target_col_name_in_df} using ADF test.")
            max_diff_order = 2 # Limit to 2 orders of differencing
            for order in range(1, max_diff_order + 1):
                is_stationary = self._check_stationarity(current_target_series, self.adf_significance_level)
                if is_stationary:
                    print(f"Target series is stationary after {self.applied_differencing_order} order(s) of differencing.")
                    break
                else:
                    if self.applied_differencing_order < max_diff_order:
                        print(f"Target series non-stationary. Applying {self.applied_differencing_order + 1}-order differencing.")
                        # Store value(s) needed for inverse transform from the original NON-DIFFERENCED series
                        # For 1st order: store first value. For 2nd order: store first two values of the 1st-order diffed series, or handle reconstruction carefully.
                        # Simpler: store the first value of the series *before* this diff operation.
                        self.values_for_inverse_differencing.append(current_target_series.iloc[0])

                        diff_values = current_target_series.diff()
                        current_target_series = diff_values.dropna() # Update current_target_series for next iteration or for scaling

                        self.applied_differencing_order += 1

                        # Align processed_data_for_scaling and df_with_all_indicators_cleaned by dropping the first row
                        processed_data_for_scaling = processed_data_for_scaling.iloc[1:]
                        if not df_with_all_indicators_cleaned.empty and not processed_data_for_scaling.empty:
                            df_with_all_indicators_cleaned = df_with_all_indicators_cleaned.loc[processed_data_for_scaling.index]
                    else:
                        print(f"Reached max differencing order ({max_diff_order}) but series still non-stationary. Proceeding with current series.")
                        break # Exit loop if max order reached

            if not self._check_stationarity(current_target_series, self.adf_significance_level) and self.applied_differencing_order == max_diff_order:
                 print(f"Warning: Target series may still be non-stationary after {max_diff_order} differencing orders.")

            # Update the target column in processed_data_for_scaling with the final differenced values (if any)
            if self.applied_differencing_order > 0:
                processed_data_for_scaling.loc[:, target_col_name_in_df] = current_target_series.values
                print(f"Shape after {self.applied_differencing_order}-order auto-differencing: {processed_data_for_scaling.shape}")

        elif self.use_differencing: # Manual differencing (1st order only as per original logic)
            print(f"Applying 1st order manual differencing to target column: {target_col_name_in_df}")
            self.values_for_inverse_differencing.append(processed_data_for_scaling[target_col_name_in_df].iloc[0])

            diff_values = processed_data_for_scaling[target_col_name_in_df].diff().dropna()
            self.applied_differencing_order = 1

            processed_data_for_scaling = processed_data_for_scaling.iloc[1:]
            processed_data_for_scaling.loc[:, target_col_name_in_df] = diff_values.values
            print(f"Shape after 1st order manual differencing: {processed_data_for_scaling.shape}")
            if not df_with_all_indicators_cleaned.empty and not processed_data_for_scaling.empty:
                 df_with_all_indicators_cleaned = df_with_all_indicators_cleaned.loc[processed_data_for_scaling.index]

        # --- Scaling ---
        print("Normalizing data...")
        feature_scaler = MinMaxScaler(feature_range=(0, 1)) # For features

        # For target, use StandardScaler if any differencing was applied, else MinMaxScaler
        if self.applied_differencing_order > 0:
            print(f"Using StandardScaler for differenced target column: {target_col_name_in_df} (order: {self.applied_differencing_order})")
            target_scaler = StandardScaler()
        else:
            print(f"Using MinMaxScaler for target column: {target_col_name_in_df}")
            target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Separate features and target for scaling
        features_df = processed_data_for_scaling.drop(columns=[target_col_name_in_df])
        target_series = processed_data_for_scaling[[target_col_name_in_df]] # Keep as DataFrame for scaler

        # Scale features
        scaled_features_values = feature_scaler.fit_transform(features_df)
        scaled_features_df = pd.DataFrame(scaled_features_values, columns=features_df.columns, index=features_df.index)

        # Scale target
        scaled_target_values = target_scaler.fit_transform(target_series)
        scaled_target_df = pd.DataFrame(scaled_target_values, columns=target_series.columns, index=target_series.index)

        # Combine scaled features and target
        scaled_df = pd.concat([scaled_features_df, scaled_target_df], axis=1)

        # Store the scaler for the target separately for inverse transformation
        # The main self.data_scaler in FullStockPredictionModel will be this target_scaler.
        # For features, their inverse transform is not typically needed if only target is predicted.
        # However, the current inverse transform logic in FullStockPredictionModel assumes one scaler for all.
        # This needs to be handled carefully.
        # For now, preprocess() will return the target_scaler.
        # The FullStockPredictionModel will need to be adapted if features also need inverse transform with a different scaler.

        print("Data preprocessing complete.")
        print(f"Final shape of preprocessed data to be used by model: {scaled_df.shape}")
        if not scaled_df.empty:
            print("Final columns in preprocessed data (target is last):", scaled_df.columns.tolist())

        # Return scaled_df, the scaler for the target, selected_features_names, df_with_all_indicators_cleaned,
        # applied_differencing_order, values_for_inverse_differencing, final_target_column_name, and log_transform_applied_to_target
        return scaled_df, target_scaler, selected_features_names, df_with_all_indicators_cleaned, self.applied_differencing_order, self.values_for_inverse_differencing, self.final_target_column_name, self.log_transform_applied_to_target


# Example Usage (for testing the module)
if __name__ == '__main__':
    # Test case 1: No differencing, no log transform
    print("\n--- Testing DataPreprocessor: No Diff, No Log ---")
    preprocessor_no_diff_no_log = DataPreprocessor(stock_ticker='AAPL', years_of_data=1, use_differencing=False, auto_differencing_adf=False, use_log_transform=False)
    try:
        # Update example usage to expect 8 return values
        processed_df, scaler, _, _, diff_order, inv_diff_vals, target_name, log_applied = preprocessor_no_diff_no_log.preprocess()
        if not processed_df.empty:
            print(f"  Processed DF shape: {processed_df.shape}")
            print(f"  Target scaler type: {type(scaler)}")
            print(f"  Final target name: {target_name}")
            print(f"  Applied differencing order: {diff_order}")
            print(f"  Values for inv diff: {inv_diff_vals}")
            print(f"  Log transform applied: {log_applied}")
            # Example of inverse transform for non-differenced data
            if target_scaler_no_diff and processed_df_no_diff.shape[1] > 0 : # Check if there's a target column
                target_col_name = processed_df_no_diff.columns[-1] # This is the scaled target
                dummy_scaled_target = processed_df_no_diff[[target_col_name]].iloc[[0]].copy() # Get first scaled target value
                original_target_val = target_scaler_no_diff.inverse_transform(dummy_scaled_target)
                print(f"  Example inverse transform (no diff) of {dummy_scaled_target.iloc[0,0]:.4f} -> {original_target_val[0,0]:.4f}")

        else:
            print("  Preprocessing (no diff) returned an empty DataFrame.")
    except Exception as e:
        print(f"  Error during no_diff preprocessing test: {e}")
        import traceback
        traceback.print_exc()

    # Test case 2: With manual differencing, no log transform
    print("\n--- Testing DataPreprocessor: Manual Diff, No Log ---")
    preprocessor_manual_diff = DataPreprocessor(stock_ticker='AAPL', years_of_data=1, use_differencing=True, auto_differencing_adf=False, use_log_transform=False)
    try:
        processed_df, scaler, _, _, diff_order, inv_diff_vals, target_name, log_applied = preprocessor_manual_diff.preprocess()
        if not processed_df.empty:
            print(f"  Processed DF shape: {processed_df.shape}")
            print(f"  Target scaler type: {type(scaler)}")
            print(f"  Final target name: {target_name}")
            print(f"  Applied differencing order: {diff_order}")
            print(f"  Values for inv diff: {inv_diff_vals}")
            print(f"  Log transform applied: {log_applied}")
            if scaler and inv_diff_vals and diff_order > 0 and processed_df.shape[1] > 0:
                target_col_name_in_df = processed_df.columns[-1]
                dummy_scaled_diff_target = processed_df[[target_col_name_in_df]].iloc[[0]].copy()
                original_diff_val = scaler.inverse_transform(dummy_scaled_diff_target)
                reconstructed_price = inv_diff_vals[0] + original_diff_val[0,0] # Simple 1st order inv
                print(f"  Example inverse transform of scaled_diff {dummy_scaled_diff_target.iloc[0,0]:.4f} -> original_diff {original_diff_val[0,0]:.4f}")
                print(f"  Reconstructed price: {inv_diff_vals[0]:.2f} + {original_diff_val[0,0]:.4f} = {reconstructed_price:.4f}")
        else:
            print("  Preprocessing returned an empty DataFrame.")
    except Exception as e:
        print(f"  Error during with_diff preprocessing test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- End of DataPreprocessor Module Test ---")

