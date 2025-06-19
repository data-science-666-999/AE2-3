import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# --- Module 4: Ensemble (Weighted Fusion) Module ---

class EnsembleModel:
    def __init__(self, optimization_method='mse_optimization', random_seed=42):
        self.optimization_method = optimization_method
        self.weights = None
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def _objective_function(self, weights, att_lstm_preds, nsgm_preds, actual_values):
        # Objective function to minimize (e.g., Mean Squared Error)
        # Ensure weights sum to 1 and are non-negative
        weights = np.array(weights)
        weights = np.maximum(0, weights) # Ensure non-negative
        weights = weights / np.sum(weights) # Normalize to sum to 1

        combined_predictions = (weights[0] * att_lstm_preds) + (weights[1] * nsgm_preds)
        return mean_squared_error(actual_values, combined_predictions)

    def train_weights(self, att_lstm_val_preds, nsgm_val_preds, actual_val_targets):
        print(f"Training ensemble weights using {self.optimization_method}...")

        if self.optimization_method == 'mse_optimization':
            # Initial guess for weights (e.g., equal weighting)
            initial_weights = np.array([0.5, 0.5])

            # Bounds for weights (0 to 1)
            bounds = ((0.0, 1.0), (0.0, 1.0))

            # Constraints: weights must sum to 1
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

            # Minimize the objective function
            result = minimize(
                self._objective_function,
                initial_weights,
                args=(att_lstm_val_preds, nsgm_val_preds, actual_val_targets),
                method='SLSQP', # Sequential Least Squares Programming
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                self.weights = np.maximum(0, result.x) # Ensure non-negative after optimization
                self.weights = self.weights / np.sum(self.weights) # Normalize again
                print(f"Ensemble weights optimized: {self.weights}")
            else:
                print(f"Weight optimization failed: {result.message}. Using equal weights.")
                self.weights = np.array([0.5, 0.5])
        elif self.optimization_method == 'fixed':
            self.weights = np.array([0.5, 0.5]) # Default to equal weights
            print(f"Using fixed equal weights: {self.weights}")
        else:
            raise ValueError("Unsupported optimization method.")

        print("Ensemble weight training complete.")

    def predict(self, att_lstm_preds, nsgm_preds):
        if self.weights is None:
            raise ValueError("Ensemble weights not trained. Call train_weights() first.")

        # Ensure predictions are numpy arrays
        att_lstm_preds = np.asarray(att_lstm_preds).flatten()
        nsgm_preds = np.asarray(nsgm_preds).flatten()

        if len(att_lstm_preds) != len(nsgm_preds):
            raise ValueError("ATT-LSTM and NSGM predictions must have the same length.")

        combined_predictions = (self.weights[0] * att_lstm_preds) + (self.weights[1] * nsgm_preds)
        return combined_predictions

# Example Usage (for testing the module)
if __name__ == '__main__':
    # Simulate validation predictions from ATT-LSTM and NSGM
    n_val_samples = 100
    np.random.seed(42)
    att_lstm_val_preds = np.random.rand(n_val_samples) * 100 # Scaled predictions
    nsgm_val_preds = np.random.rand(n_val_samples) * 100 # Scaled predictions
    actual_val_targets = np.random.rand(n_val_samples) * 100 # Scaled actual values

    # Initialize and train the ensemble model
    ensemble_model = EnsembleModel(optimization_method='mse_optimization')
    ensemble_model.train_weights(att_lstm_val_preds, nsgm_val_preds, actual_val_targets)

    # Simulate test predictions from ATT-LSTM and NSGM
    n_test_samples = 50
    att_lstm_test_preds = np.random.rand(n_test_samples) * 100
    nsgm_test_preds = np.random.rand(n_test_samples) * 100

    # Make combined predictions
    combined_predictions = ensemble_model.predict(att_lstm_test_preds, nsgm_test_preds)

    print("\nSample combined predictions:", combined_predictions[:5])
    print("Ensemble weights used:", ensemble_model.weights)

    # Test with fixed weights
    ensemble_model_fixed = EnsembleModel(optimization_method='fixed')
    ensemble_model_fixed.train_weights(att_lstm_val_preds, nsgm_val_preds, actual_val_targets) # val_targets are ignored for fixed method
    combined_predictions_fixed = ensemble_model_fixed.predict(att_lstm_test_preds, nsgm_test_preds)
    print("\nSample combined predictions (fixed weights):", combined_predictions_fixed[:5])
    print("Ensemble weights used (fixed):", ensemble_model_fixed.weights)


