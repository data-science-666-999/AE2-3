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

# --- Module 1: Data Acquisition and Preprocessing ---

class DataPreprocessor:
    def __init__(self, stock_ticker='AEL', years_of_data=10, random_seed=42): # Default to 10 years
        self.stock_ticker = stock_ticker
        self.years_of_data = years_of_data
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def _download_yfinance_data(self):
        print(f"Downloading {self.years_of_data} years of stock data for {self.stock_ticker} from Yahoo Finance...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years_of_data * 365.25) # Account for leap years

        df = yf.download(self.stock_ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if df.empty:
            raise ValueError(f"No data downloaded for ticker {self.stock_ticker}. Check ticker symbol or date range.")

        # Ensure standard column names (yfinance sometimes uses 'Adj Close')
        df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True) # Standardize if needed

        print(f"Downloaded data shape: {df.shape}")
        return df

    def _calculate_technical_indicators(self, df):
        print("Calculating technical indicators...")
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

        print(f"Shape after adding all indicators: {df.shape}")
        return df

    def _apply_lasso_feature_selection(self, df, target_column='Close', alpha=0.01):
        print("Applying LASSO feature selection...")
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

        lasso = Lasso(alpha=alpha, random_state=self.random_seed, max_iter=10000) # Increased max_iter
        lasso.fit(scaled_features, target)

        selected_features_mask = lasso.coef_ != 0

        if not np.any(selected_features_mask):
            print(f"Warning: LASSO with alpha={alpha} selected 0 features. This might be too high or data needs review.")
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

        return final_df_for_scaling, selected_feature_names, lasso # Return LASSO model

    def preprocess(self):
        stock_data = self._download_yfinance_data()
        stock_data = self._calculate_technical_indicators(stock_data)

        # For LASSO, we'll predict 'Close' based on other features.
        # The target 'Close' should be present in stock_data before this step.
        # Make a copy to avoid SettingWithCopyWarning if stock_data is a slice
        # Store the lasso model returned by _apply_lasso_feature_selection
        processed_data_for_scaling, selected_features, lasso_model = self._apply_lasso_feature_selection(stock_data.copy(), target_column='Close')
        self.lasso_model = lasso_model # Store for access in main block

        if processed_data_for_scaling.empty:
            print("Preprocessing failed: No data after feature selection.")
            # Return empty DataFrame and None for scaler, let the caller handle this
            return pd.DataFrame(), None

        print("Normalizing data...")
        # Normalize all selected features including the target for LSTM input
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit and transform on the data that has been cleaned (NaNs dropped) and feature-selected
        scaled_data_values = scaler.fit_transform(processed_data_for_scaling)

        # Create the final scaled DataFrame with correct columns and index
        scaled_df = pd.DataFrame(scaled_data_values,
                                 columns=processed_data_for_scaling.columns,
                                 index=processed_data_for_scaling.index)

        print("Data preprocessing complete.")
        print(f"Final shape of preprocessed data to be used by model: {scaled_df.shape}")
        if not scaled_df.empty:
            print("Final columns in preprocessed data:", scaled_df.columns.tolist())
        return scaled_df, scaler # Return scaler to inverse transform predictions later

# Example Usage (for testing the module)
if __name__ == '__main__':
    # Test with AAPL for 1 year for faster local testing
    preprocessor = DataPreprocessor(stock_ticker='AAPL', years_of_data=1)
    original_target_column = 'Close' # yfinance specific target before potential flattening
    try:
        processed_df, data_scaler = preprocessor.preprocess()

        if not processed_df.empty:
            print("\n--- DataPreprocessor Module Test Results ---")
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


