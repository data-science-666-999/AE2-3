import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import modules
from data_preprocessing_module import DataPreprocessor
from att_lstm_module import ATTLSTMModel
# from nsgm1n_module import NSGM1NModel # Removed
# from ensemble_module import EnsembleModel # Removed
import matplotlib.pyplot as plt
import os
import time # Import time module for performance testing

# --- Module 5: Integrate and Test the Full Model ---

class FullStockPredictionModel:
    def __init__(self, stock_ticker="AEL", years_of_data=10, look_back=60, random_seed=42): # Simplified constructor
        self.stock_ticker = stock_ticker
        self.years_of_data = years_of_data
        self.look_back = look_back
        # self.lstm_units, self.dense_units, self.lstm_learning_rate are no longer needed here
        # as ATTLSTMModel will be configured by model_params (hypothetical HPs)
        # self.ensemble_optimization_method = ensemble_optimization_method # Removed
        self.random_seed = random_seed

        self.data_preprocessor = DataPreprocessor(
            stock_ticker=self.stock_ticker,
            years_of_data=self.years_of_data,
            random_seed=self.random_seed
        )
        self.att_lstm_model = None
        # self.nsgm_model = NSGM1NModel() # Removed
        # self.ensemble_model = EnsembleModel(...) # Removed
        self.data_scaler = None
        self.processed_df = None

    def _create_sequences(self, data, target_column_name="Close"):
        # Ensure target_column_name is in data.columns
        if target_column_name not in data.columns:
            raise ValueError(f"Target column '{target_column_name}' not found in data.")

        target_column_index = data.columns.get_loc(target_column_name)
        
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data.iloc[i:(i + self.look_back), :].values)
            y.append(data.iloc[i + self.look_back, target_column_index])
        return np.array(X), np.array(y)

    def train_and_evaluate(self, epochs=50, batch_size=32, test_size=0.2, val_size=0.25):
        print("\n--- Starting Full Model Training and Evaluation ---")

        # 1. Data Acquisition and Preprocessing
        self.processed_df, self.data_scaler = self.data_preprocessor.preprocess()
        if self.processed_df.empty:
            print("Error: Preprocessed data is empty. Aborting training.")
            return

        # Determine target column name dynamically (it's the last column from DataPreprocessor)
        if self.processed_df.empty:
            print("Error: Preprocessed data is empty. Cannot determine target column. Aborting training.")
            return

        target_column_name = self.processed_df.columns[-1]
        print(f"Dynamically determined target column name: {target_column_name}")

        # Create sequences for LSTM and NSGM (if NSGM uses sequences)
        # For NSGM, we need the original data for training, not sequences.
        # For LSTM, we need sequences.
        X_seq, y_seq = self._create_sequences(self.processed_df, target_column_name=target_column_name)

        # Split data into training, validation, and test sets
        # Ensure temporal split for time series data
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
            X_seq, y_seq, test_size=test_size, random_state=self.random_seed, shuffle=False
        )
        X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
            X_train_seq, y_train_seq, test_size=val_size, random_state=self.random_seed, shuffle=False
        ) # val_size of remaining train_size

        print(f"\nData split shapes:")
        print(f"X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
        print(f"X_val_seq: {X_val_seq.shape}, y_val_seq: {y_val_seq.shape}")
        print(f"X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")

        # 2. Train Attention-Enhanced LSTM (ATT-LSTM) Module
        # Define hypothetical best hyperparameters (as if loaded from a completed tuning run)
        # These would replace the default self.lstm_units, self.dense_units, etc.
        hypothetical_best_hps = {
            'num_lstm_layers': 2,
            'lstm_units_1': 256,
            'lstm_units_2': 128,
            'num_dense_layers': 2,
            'dense_units_1': 128,
            'dense_units_2': 64,
            'learning_rate': 0.0005,
            'dropout_rate_lstm': 0.25, # Adjusted for example
            'dropout_rate_dense': 0.35, # Adjusted for example
            'activation_dense': 'relu'
        }
        print(f"\n--- Using Hypothetical Best Hyperparameters for ATT-LSTM ---")
        for key, value in hypothetical_best_hps.items():
            print(f"  {key}: {value}")
        print("------------------------------------------------------------")


        input_shape_lstm = (X_train_seq.shape[1], X_train_seq.shape[2]) # (timesteps, features)

        # Instantiate ATTLSTMModel with the hypothetical best HPs
        self.att_lstm_model = ATTLSTMModel(
            input_shape=input_shape_lstm,
            look_back=self.look_back, # look_back is fixed for data prep, model needs to know it
            random_seed=self.random_seed,
            model_params=hypothetical_best_hps # Pass the dictionary here
        )

        # Build the model using these parameters (since hp=None, it will use model_params)
        self.att_lstm_model.build_model()

        # Train the model
        # Note: epochs and batch_size for this final training could also be part of HPs,
        # but for now, we use the ones passed to train_and_evaluate.
        self.att_lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs, batch_size=batch_size)

        # Predict on test set for LSTM
        att_lstm_test_preds = self.att_lstm_model.predict(X_test_seq).flatten()

        # --- NSGM and Ensemble sections removed ---

        # Inverse transform predictions and actual values to original scale
        # Only ATT-LSTM predictions are relevant now
        dummy_preds_lstm = np.zeros((len(att_lstm_test_preds), self.processed_df.shape[1]))
        dummy_preds_lstm[:, self.processed_df.columns.get_loc(target_column_name)] = att_lstm_test_preds
        original_att_lstm_test_preds = self.data_scaler.inverse_transform(dummy_preds_lstm)[:, self.processed_df.columns.get_loc(target_column_name)]

        # NSGM and Ensemble predictions are no longer generated or transformed
        # original_nsgm_test_preds = None
        # original_ensemble_test_preds = None

        dummy_actuals = np.zeros((len(y_test_seq), self.processed_df.shape[1]))
        dummy_actuals[:, self.processed_df.columns.get_loc(target_column_name)] = y_test_seq
        original_y_test_seq = self.data_scaler.inverse_transform(dummy_actuals)[:, self.processed_df.columns.get_loc(target_column_name)]

        # Evaluate performance
        print("\n--- Model Performance on Test Set (Original Scale) ---")
        
        # ATT-LSTM Metrics
        mse_lstm = mean_squared_error(original_y_test_seq, original_att_lstm_test_preds)
        mae_lstm = mean_absolute_error(original_y_test_seq, original_att_lstm_test_preds)
        rmse_lstm = np.sqrt(mse_lstm)
        mean_actuals_lstm = np.mean(original_y_test_seq)
        if mean_actuals_lstm == 0: # Avoid division by zero
            rmse_perc_mean_actuals_lstm = float('inf')
        else:
            rmse_perc_mean_actuals_lstm = (rmse_lstm / mean_actuals_lstm) * 100
        print(f"ATT-LSTM - MSE: {mse_lstm:.4f}, MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}")
        print(f"ATT-LSTM - RMSE as % of Mean Actuals: {rmse_perc_mean_actuals_lstm:.2f}% (Target: <= 6%)")

        # NSGM Metrics (Removed)
        # Ensemble Metrics (Removed)

        # Store the percentage RMSE in the metrics dictionary as well for ATT-LSTM only
        # The metrics dictionary is populated later in the code, so we'll add it there.

        # Create a directory for plots if it doesn't exist
        plots_dir = "." # Save to root directory for now
        # os.makedirs(plots_dir, exist_ok=True) # Not needed for root
        print(f"Attempting to save plots to current directory: {os.path.abspath(plots_dir)}")


        # Generate and save plots (primarily for ATT-LSTM now)
        test_indices = self.processed_df.index[-len(original_y_test_seq):]

        try:
            # ATT-LSTM Plots
            self._plot_predictions_vs_actuals_timeseries(
                test_indices, original_y_test_seq, original_att_lstm_test_preds,
                "ATT-LSTM Model Predictions vs Actuals",
                os.path.join(plots_dir, "full_run_att_lstm_preds_vs_actuals_timeseries.png")
            )
            self._plot_predictions_vs_actuals_scatter(
                original_y_test_seq, original_att_lstm_test_preds,
                "ATT-LSTM Model Predictions vs Actuals (Scatter)",
                os.path.join(plots_dir, "full_run_att_lstm_preds_vs_actuals_scatter.png")
            )
            residuals_lstm = original_y_test_seq - original_att_lstm_test_preds
            self._plot_residuals_timeseries(
                test_indices, residuals_lstm,
                "ATT-LSTM Model Residuals Over Time",
                os.path.join(plots_dir, "full_run_att_lstm_residuals_timeseries.png")
            )
            self._plot_residuals_histogram(
                residuals_lstm,
                "ATT-LSTM Model Distribution of Residuals",
                os.path.join(plots_dir, "full_run_att_lstm_residuals_histogram.png")
            )

            # NSGM Plots (Removed)
            # Ensemble Plots (Removed)

        except Exception as e:
            print(f"Error during plotting: {e}")

        print(f"\nVisualizations saved to '{plots_dir}' directory.")
        print("\n--- Full Model Training and Evaluation Complete ---")

        return {
            "att_lstm_preds": original_att_lstm_test_preds,
            # "nsgm_preds": None, # Removed
            # "ensemble_preds": None, # Removed
            "actual_values": original_y_test_seq,
            "metrics": {
                "lstm_mse": mse_lstm, "lstm_mae": mae_lstm, "lstm_rmse": rmse_lstm, "lstm_rmse_perc": rmse_perc_mean_actuals_lstm
                # NSGM and Ensemble metrics removed
            }
        }

    # --- Plotting Helper Methods ---
    def _plot_predictions_vs_actuals_timeseries(self, x_values, actuals, predictions, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, actuals, label='Actual Values', color='blue', marker='.', linestyle='-')
        plt.plot(x_values, predictions, label='Predicted Values', color='red', marker='.', linestyle='--')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Stock Price (Original Scale)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _plot_predictions_vs_actuals_scatter(self, actuals, predictions, title, filename):
        plt.figure(figsize=(8, 8))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--') # y=x line
        plt.title(title)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _plot_residuals_timeseries(self, x_values, residuals, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, residuals, label='Residuals (Actual - Predicted)', color='green', linestyle='-')
        plt.axhline(0, color='red', linestyle='--', label='Zero Error')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Residual Value")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _plot_residuals_histogram(self, residuals, title, filename, bins=50):
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=bins, edgecolor='black', alpha=0.7)
        plt.title(title)
        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.axvline(residuals.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {residuals.mean():.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# Example Usage (Run the full model)
if __name__ == '__main__':
    full_model = FullStockPredictionModel(
        stock_ticker='^AEX',    # Using AEX index
        years_of_data=10,       # Using 10 years of data
        look_back=60,           # Common look_back period
        random_seed=42
    )

    # Train with more epochs, relying on EarlyStopping in ATTLSTMModel
    # Epochs and batch_size for the final training run after HPO.
    # These could also be part of the tuned HPs.
    results = full_model.train_and_evaluate(
        epochs=150, # Increased epochs for final training
        batch_size=32
    )

    if results: # Check if results were returned (not empty on error)
        print("\n--- Final Results ---")
        if "att_lstm_preds" in results and results["att_lstm_preds"] is not None:
             print("Final ATT-LSTM Predictions (first 5):", results["att_lstm_preds"][:5])
        # NSGM and Ensemble preds removed

        print("Actual Values (first 5):", results["actual_values"][:5])

        print("\nMetrics from the run:")
        # Only LSTM metrics are now relevant
        if "lstm_mse" in results["metrics"]:
            print(f"  ATT-LSTM Model:")
            print(f"    MSE:  {results['metrics']['lstm_mse']:.4f}")
            print(f"    MAE:  {results['metrics']['lstm_mae']:.4f}")
            print(f"    RMSE: {results['metrics']['lstm_rmse']:.4f}")
            if "lstm_rmse_perc" in results["metrics"]:
                print(f"    RMSE as % of Mean Actuals: {results['metrics']['lstm_rmse_perc']:.2f}%")
        else:
            print("  ATT-LSTM Model: Metrics not available.")

        # --- Production Readiness Testing (Simulated) ---
        if full_model.att_lstm_model and hasattr(full_model.att_lstm_model, 'model') and full_model.att_lstm_model.model is not None:
            print("\n--- Production Readiness Testing (Prediction Speed) ---")

            # Need X_test_seq from the train_and_evaluate scope, or re-generate a sample
            # For simplicity, let's assume train_and_evaluate populates an X_test_seq that can be accessed
            # or we retrieve it from the results if it was returned.
            # However, train_and_evaluate doesn't return X_test_seq.
            # So, we need to get a sample of X_test_seq.
            # We can grab one from the `full_model` instance if it stores it, or re-create one.
            # Let's assume `full_model.processed_df` and `full_model.look_back` are available.

            if full_model.processed_df is not None and not full_model.processed_df.empty:
                # Create a sample sequence for testing prediction speed
                # This is a simplified way; ideally, use an actual X_test_seq sample
                num_features = full_model.processed_df.shape[1]
                sample_raw_data = np.random.rand(full_model.look_back, num_features) # Create dummy data matching shape

                # For single instance prediction, the input needs to be (1, look_back, num_features)
                single_instance_input = np.expand_dims(sample_raw_data, axis=0)

                # Time single instance prediction
                num_single_runs = 10
                single_pred_times = []
                for _ in range(num_single_runs):
                    start_time = time.time()
                    _ = full_model.att_lstm_model.predict(single_instance_input)
                    end_time = time.time()
                    single_pred_times.append(end_time - start_time)

                avg_single_pred_time = np.mean(single_pred_times)
                print(f"Average single instance prediction time: {avg_single_pred_time*1000:.2f} ms (over {num_single_runs} runs)")

                # Time batch instance prediction
                batch_size_test = 32
                # Create a batch of dummy data: (batch_size_test, look_back, num_features)
                batch_input = np.random.rand(batch_size_test, full_model.look_back, num_features)

                num_batch_runs = 5
                batch_pred_times = []
                for _ in range(num_batch_runs):
                    start_time = time.time()
                    _ = full_model.att_lstm_model.predict(batch_input)
                    end_time = time.time()
                    batch_pred_times.append(end_time - start_time)

                avg_batch_pred_time_total = np.mean(batch_pred_times)
                avg_batch_pred_time_per_instance = avg_batch_pred_time_total / batch_size_test
                print(f"Average batch ({batch_size_test} instances) prediction time: {avg_batch_pred_time_total*1000:.2f} ms (total)")
                print(f"Average per-instance prediction time in batch: {avg_batch_pred_time_per_instance*1000:.2f} ms (over {num_batch_runs} runs)")
            else:
                print("Could not perform prediction speed test: processed_df not available.")
        else:
            print("ATT-LSTM model not available for production readiness testing.")

    else:
        print("Model training and evaluation did not complete successfully.")


