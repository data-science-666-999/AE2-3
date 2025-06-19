import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# --- Module 1: Data Acquisition and Preprocessing ---

class DataPreprocessor:
    def __init__(self, stock_ticker='AEL', start_date='2010-01-01', end_date='2023-12-31', random_seed=42):
        self.stock_ticker = stock_ticker
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def _simulate_stock_data(self):
        # Simulate historical stock data for AEL
        dates = pd.date_range(start=self.start_date, end=self.end_date)
        n_days = len(dates)

        # Simulate Open, High, Low, Close, Volume
        close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        open_prices = close_prices * (1 + np.random.randn(n_days) * 0.01)
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.rand(n_days) * 0.005)
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.rand(n_days) * 0.005)
        volume = np.random.randint(1_000_000, 10_000_000, n_days)

        df = pd.DataFrame({
            'Date': dates,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        })
        df.set_index('Date', inplace=True)
        return df

    def _calculate_technical_indicators(self, df):
        # Calculate common technical indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
        df['Upper_BB'], df['Lower_BB'] = self._calculate_bollinger_bands(df['Close'])
        return df

    def _calculate_rsi(self, prices, window=14):
        # Calculate Relative Strength Index (RSI)
        diff = prices.diff(1)
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)
        avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, span_fast=12, span_slow=26, span_signal=9):
        # Calculate Moving Average Convergence Divergence (MACD)
        exp_fast = prices.ewm(span=span_fast, adjust=False).mean()
        exp_slow = prices.ewm(span=span_slow, adjust=False).mean()
        macd = exp_fast - exp_slow
        signal = macd.ewm(span=span_signal, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(self, prices, window=20, num_std_dev=2):
        # Calculate Bollinger Bands
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std_dev)
        lower_band = sma - (std * num_std_dev)
        return upper_band, lower_band

    def _simulate_sentiment_data(self, df):
        # Simulate investor sentiment data
        # For a real scenario, this would involve NLP on textual data
        sentiment = np.random.rand(len(df)) * 2 - 1  # Values between -1 and 1
        df['Sentiment'] = sentiment
        return df

    def _apply_lasso_feature_selection(self, df, target_column='Close', alpha=0.01):
        # Apply LASSO for feature selection
        # Drop rows with NaN values introduced by rolling windows
        df_cleaned = df.dropna()

        if df_cleaned.empty:
            print("Warning: DataFrame is empty after dropping NaNs. Cannot apply LASSO.")
            return pd.DataFrame(), []

        features = df_cleaned.drop(columns=[target_column])
        target = df_cleaned[target_column]

        # It's good practice to scale features before LASSO
        scaler_features = MinMaxScaler()
        scaled_features = scaler_features.fit_transform(features)

        lasso = Lasso(alpha=alpha, random_state=self.random_seed, max_iter=10000)
        lasso.fit(scaled_features, target)

        selected_features_indices = np.where(lasso.coef_ != 0)[0]
        selected_feature_names = features.columns[selected_features_indices].tolist()

        # Include the target column in the returned DataFrame
        return df_cleaned[selected_feature_names + [target_column]], selected_feature_names

    def preprocess(self):
        print(f"Simulating stock data for {self.stock_ticker}...")
        stock_data = self._simulate_stock_data()

        print("Calculating technical indicators...")
        stock_data = self._calculate_technical_indicators(stock_data)

        print("Simulating sentiment data...")
        stock_data = self._simulate_sentiment_data(stock_data)

        print("Applying LASSO feature selection...")
        # For LASSO, we'll predict 'Close' based on other features
        processed_data, selected_features = self._apply_lasso_feature_selection(stock_data.copy(), target_column='Close')

        if processed_data.empty:
            print("Preprocessing failed: No data after feature selection.")
            return pd.DataFrame(), []

        print("Normalizing data...")
        # Normalize all features including the target for LSTM input
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit on all selected features and the target
        scaled_data = scaler.fit_transform(processed_data)
        scaled_df = pd.DataFrame(scaled_data, columns=processed_data.columns, index=processed_data.index)

        print("Data preprocessing complete.")
        return scaled_df, scaler # Return scaler to inverse transform predictions later

# Example Usage (for testing the module)
if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    processed_df, data_scaler = preprocessor.preprocess()
    print("\nProcessed Data Head:")
    print(processed_df.head())
    print("\nProcessed Data Info:")
    processed_df.info()
    print("\nShape of processed data:", processed_df.shape)

    # Example of inverse transformation
    # dummy_data = np.zeros((1, len(processed_df.columns)))
    # dummy_data[0, processed_df.columns.get_loc('Close')] = 0.5 # Example scaled close price
    # original_close = data_scaler.inverse_transform(dummy_data)[0, processed_df.columns.get_loc('Close')]
    # print(f"\nExample: Scaled Close 0.5 inverse transforms to: {original_close}")


