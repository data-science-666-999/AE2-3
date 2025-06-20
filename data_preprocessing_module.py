import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split # No longer needed here for example
import yfinance as yf
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt
import os
import pickle
from datetime import date # For cache invalidation

# Define a cache directory
CACHE_DIR = "data_cache"

# --- Module 1: Data Acquisition and Preprocessing ---

class DataPreprocessor:
    def __init__(self, stock_ticker='AEL', years_of_data=10, random_seed=42, lasso_alpha=0.005): # Default alpha
        self.stock_ticker = stock_ticker
        self.years_of_data = years_of_data
        self.random_seed = random_seed
        self.lasso_alpha = lasso_alpha # Store alpha
        np.random.seed(self.random_seed)
        os.makedirs(CACHE_DIR, exist_ok=True) # Ensure cache directory exists

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

        df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
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
            # Avoid division by zero or very small close prices if necessary, though typically not an issue for stock prices
            df['ATR_Normalized'] = df['ATR'] / df['Close'].replace(0, np.nan) # replace 0 with NaN to avoid division error, then fillna
            df['ATR_Normalized'].fillna(method='bfill', inplace=True) # Backfill first for initial NaNs
            df['ATR_Normalized'].fillna(method='ffill', inplace=True) # Then ffill for any remaining

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
        # Also return the list of selected_feature_names.
        return final_df_for_scaling, selected_feature_names, lasso_model, df_cleaned

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
        processed_data_for_scaling, selected_features_names, lasso_model, df_with_all_indicators_cleaned = \
            self._apply_lasso_feature_selection(stock_data_with_indicators.copy(), target_column='Close')

        self.lasso_model = lasso_model # Store for access if needed

        if processed_data_for_scaling.empty:
            print("Preprocessing failed: No data after feature selection.")
            # Return empty DataFrame, None for scaler, empty list for selected features, and empty df for all indicators
            return pd.DataFrame(), None, [], pd.DataFrame()

        print("Normalizing data...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_values = scaler.fit_transform(processed_data_for_scaling)
        scaled_df = pd.DataFrame(scaled_data_values,
                                 columns=processed_data_for_scaling.columns,
                                 index=processed_data_for_scaling.index)

        print("Data preprocessing complete.")
        print(f"Final shape of preprocessed data to be used by model: {scaled_df.shape}")
        if not scaled_df.empty:
            print("Final columns in preprocessed data (target is last):", scaled_df.columns.tolist())

        # Return scaled_df, scaler, selected_features_names, and df_with_all_indicators_cleaned
        return scaled_df, scaler, selected_features_names, df_with_all_indicators_cleaned

# Example Usage (for testing the module)
if __name__ == '__main__':
    preprocessor = DataPreprocessor(stock_ticker='AAPL', years_of_data=1)
    original_target_column = 'Close'
    try:
        # Update to reflect new return values
        processed_df, data_scaler, selected_features, df_all_indicators = preprocessor.preprocess()

        if not processed_df.empty:
            print("\n--- DataPreprocessor Module Test Results ---")
            print("\nSelected Features by LASSO:")
            print(selected_features)
            print(f"\nNumber of selected features: {len(selected_features)}")

            print("\nDataFrame with all indicators (head):")
            print(df_all_indicators.head())
            print("\n1. Processed Data Head:")
            print(processed_df.head())
            print("\n2. Processed Data Info:")
            processed_df.info()
            print("\n3. Shape of processed data:", processed_df.shape)

            print("\n4. Descriptive Statistics of Processed Data:")
            print(processed_df.describe())

            print("\n5. NaN check in final processed data (sum):")
            print(processed_df.isnull().sum())

            # LASSO feature selection details
            if hasattr(preprocessor, 'lasso_model') and preprocessor.lasso_model:
                print("\n6. LASSO Feature Selection Details:")
                # The actual features used for LASSO (excluding the target)
                # are the columns of processed_df minus the last one (which is the target)
                lasso_feature_names = processed_df.columns[:-1].tolist()

                # Verify target column name after potential flattening in _apply_lasso_feature_selection
                # The last column of processed_df is the target.
                actual_target_column_name = processed_df.columns[-1]
                print(f"   Target column for LASSO (determined dynamically): {actual_target_column_name}")

                selected_coeffs = preprocessor.lasso_model.coef_

                # Filter out zero coefficients for selected features display
                # Note: lasso_feature_names are from the *scaled* data fed to LASSO,
                # which should align with processed_df.columns[:-1]

                actual_selected_features = []
                actual_selected_coeffs = []
                print(f"   Number of features input to LASSO (excluding target): {len(lasso_feature_names)}")
                print(f"   Number of coefficients from LASSO: {len(selected_coeffs)}")

                # Reconstruct the feature list that LASSO actually saw (before it selected from them)
                # This needs to be done carefully if yfinance ticker was part of column names
                # For AEL, it becomes Close_AEL, High_AEL etc.
                # The _apply_lasso_feature_selection method handles this flattening.
                # features_df.columns within that method holds the correct names.
                # For now, let's assume lasso_feature_names is correct.

                # The features that LASSO selected are those with non-zero coefficients.
                # The `selected_feature_names` variable from `_apply_lasso_feature_selection`
                # already holds this, which are the columns of `processed_df[:-1]`
                print(f"   Selected features by LASSO (non-zero coefficients): {lasso_feature_names}")

                if lasso_feature_names: # Check if there are features to plot
                    plt.figure(figsize=(12, 8))
                    plt.bar(lasso_feature_names, selected_coeffs[:len(lasso_feature_names)]) # Ensure we only plot for available names
                    plt.xlabel("Features")
                    plt.ylabel("LASSO Coefficient Value")
                    plt.title("LASSO Feature Coefficients")
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plot_filename = "lasso_feature_coefficients.png"
                    plt.savefig(plot_filename)
                    print(f"\n   LASSO coefficients plot saved as {plot_filename} in {os.path.abspath('.')}")
                    plt.close()
                else:
                    print("   No features selected by LASSO to plot.")

            else:
                print("\n6. LASSO Model not available or no features selected.")


            if data_scaler and actual_target_column_name in processed_df.columns:
                print(f"\n7. Example of Inverse Transformation (Target: {actual_target_column_name}):")
                if len(processed_df) > 0:
                    dummy_row = processed_df.iloc[[0]].copy()
                    target_col_idx_in_processed_df = processed_df.columns.get_loc(actual_target_column_name)
                    example_scaled_target = dummy_row.iloc[0, target_col_idx_in_processed_df]

                    original_row_values = data_scaler.inverse_transform(dummy_row)
                    original_target = original_row_values[0, target_col_idx_in_processed_df]
                    print(f"   Scaled Target '{actual_target_column_name}' {example_scaled_target:.4f} from first row inverse transforms to: {original_target:.4f}")
                else:
                    print("   Cannot demonstrate inverse transform: processed_df is empty.")
            else:
                print(f"\n7. Cannot demonstrate inverse transform: Scaler or target column '{actual_target_column_name}' missing.")
            print("\n--- End of DataPreprocessor Module Test ---")
        else:
            print("Preprocessing returned an empty DataFrame. Cannot display details.")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        import traceback
        traceback.print_exc()


