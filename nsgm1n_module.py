import numpy as np
import pandas as pd

# --- Module 3: Cyclic Multidimensional Gray Model (NSGM(1,N)) Module ---

class NSGM1NModel:
    def __init__(self):
        self.model_params = None

    def _ago(self, data):
        # Accumulated Generating Operation (AGO)
        return np.cumsum(data, axis=0)

    def _iago(self, data_ago):
        # Inverse Accumulated Generating Operation (IAGO)
        # The first element remains the same, subsequent elements are differences
        data = np.zeros_like(data_ago)
        data[0] = data_ago[0]
        data[1:] = data_ago[1:] - data_ago[:-1]
        return data

    def train(self, X_train, y_train):
        # X_train: (n_samples, n_features) - input features (related series)
        # y_train: (n_samples,) - target variable (primary series)

        # Combine y_train and X_train to form the multivariate series
        # The first column is the primary series (y_train), rest are related series (X_train)
        y_train_reshaped = y_train.reshape(-1, 1)
        data_combined = np.hstack((y_train_reshaped, X_train))

        n_samples, n_vars = data_combined.shape

        # Step 1: Perform AGO on all series
        data_ago = self._ago(data_combined)

        # Step 2: Construct the B matrix and Y vector for parameter estimation
        # Primary series (y1_ago) is the first column of data_ago
        y1_ago = data_ago[:, 0]

        # Z1 for the 'a' coefficient (average of adjacent AGO values of primary series)
        Z1 = 0.5 * (y1_ago[1:] + y1_ago[:-1])

        # Y vector: -X1(0)(k) (negative of the original differenced primary series)
        # The equation is X1(0)(k) + a*Z1(k) = sum(bi*Xi(1)(k))
        # So, Y = X1(0)(k) and B = [-Z1(k), Xi(1)(k)]
        # Or, for least squares: Y = X1(0)(k) and B = [-Z1(k), X2(1)(k), ..., XN(1)(k)]
        # Let's use the form: Y = X1(0)(k) and B = [-Z1(k), X2(1)(k), ..., XN(1)(k)]

        Y = data_combined[1:, 0].reshape(-1, 1) # Original differenced primary series

        # B matrix: [-Z1(k), X2(1)(k), X3(1)(k), ...]
        B = np.zeros((n_samples - 1, n_vars))
        B[:, 0] = -Z1 # Coefficient for 'a'
        B[:, 1:] = data_ago[1:, 1:] # Coefficients for 'b's (related series AGO values)

        # Step 3: Estimate parameters (a, b2, ..., bN) using least squares
        # beta = (B.T * B)^-1 * B.T * Y
        try:
            # Add a small regularization term to B.T @ B to prevent singularity
            lambda_reg = 1e-6
            beta = np.linalg.inv(B.T @ B + lambda_reg * np.eye(B.shape[1])) @ B.T @ Y
        except np.linalg.LinAlgError:
            print("Singular matrix encountered in NSGM(1,N) parameter estimation. Returning None.")
            self.model_params = None
            return

        self.model_params = beta.flatten()
        print("NSGM(1,N) model training complete.")

    def predict(self, X_test_sequence):
        # X_test_sequence: (timesteps, n_features) - last `timesteps` of data, including primary series.
        # The primary series is assumed to be the first column (index 0).
        # This method predicts the next value of the primary series.

        if self.model_params is None:
            raise ValueError("Model not trained. Call train() first.")

        a = self.model_params[0]
        b_coeffs = self.model_params[1:] # Coefficients for related series

        # Perform AGO on the entire input sequence to get the AGO values up to the current point.
        # This allows the model to be self-contained for prediction given a historical window.
        ago_sequence = self._ago(X_test_sequence)

        # Get the last AGO values from the sequence for prediction
        primary_series_ago_current = ago_sequence[-1, 0]
        related_series_ago_current = ago_sequence[-1, 1:]

        # Calculate the sum(bi*Xi(1)(k)) for related series
        # Ensure b_coeffs and related_series_ago_current have compatible shapes
        if len(b_coeffs) != len(related_series_ago_current):
            raise ValueError("Number of b coefficients does not match number of related series.")

        sum_b_xi = np.sum(b_coeffs * related_series_ago_current)

        # Predict the next AGO value of the primary series (X1(1)(k+1))
        # X1(1)(k+1) = (X1(1)(k) - (sum(bi*Xi(1)(k)) / a)) * exp(-a) + (sum(bi*Xi(1)(k)) / a)
        # Note: if 'a' is very close to 0, this formula can be unstable. Add a small epsilon.
        epsilon = 1e-9
        if abs(a) < epsilon:
            # Handle case where 'a' is close to zero (linear approximation)
            predicted_ago_primary = primary_series_ago_current + sum_b_xi
        else:
            predicted_ago_primary = (primary_series_ago_current - (sum_b_xi / a)) * np.exp(-a) + (sum_b_xi / a)

        # Inverse AGO to get the predicted actual value (X1(0)(k+1))
        # X1(0)(k+1) = X1(1)(k+1) - X1(1)(k)
        # This requires the actual AGO value at time k, which is `primary_series_ago_current`.
        predicted_actual_primary = predicted_ago_primary - primary_series_ago_current

        return predicted_actual_primary

# Example Usage (for testing the module)
if __name__ == '__main__':
    # Simulate some preprocessed data (replace with actual preprocessed_df from Module 1)
    # For demonstration, let's assume preprocessed_df has 6 features (including 'Close')
    # and a shape like (n_samples, n_features)
    n_samples = 1000
    n_features = 6 # e.g., Close, High, Low, SMA_30, EMA_10, RSI (Close is primary)
    # Simulate scaled data between 0 and 1
    simulated_data = np.random.rand(n_samples, n_features)
    simulated_df = pd.DataFrame(simulated_data, columns=['Close'] + [f'feature_{i}' for i in range(n_features-1)])

    # Assuming 'Close' is the target column and is the first column (index 0)
    target_column_index = 0

    look_back = 60

    # Split data for training NSGM(1,N) parameters
    train_size = int(len(simulated_df) * 0.8)
    train_data = simulated_df.iloc[:train_size].values
    test_data = simulated_df.iloc[train_size:].values

    # For NSGM(1,N) training, X_train is all features except the target, y_train is the target.
    # But the model expects the primary series as the first column of combined data.
    nsgm_X_train = train_data[:, 1:] # Related series
    nsgm_y_train = train_data[:, 0]  # Primary series

    nsgm_model = NSGM1NModel()
    nsgm_model.train(nsgm_X_train, nsgm_y_train)

    # Test prediction with a sequence (last 'look_back' steps of test_data)
    if len(test_data) >= look_back + 1: # Need look_back steps for input and 1 for actual next value
        test_sequence = test_data[0:look_back, :]
        predicted_val = nsgm_model.predict(test_sequence)
        print(f"\nNSGM(1,N) Predicted Value: {predicted_val}")
        print(f"Actual Value (next step): {test_data[look_back, 0]}")
    else:
        print("Not enough test data to create a sequence for prediction.")


