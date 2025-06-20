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
    def __init__(self, stock_ticker="AEL", years_of_data=10, look_back=60, random_seed=42, lasso_alpha=0.005): # Added lasso_alpha
        self.stock_ticker = stock_ticker
        self.years_of_data = years_of_data
        self.look_back = look_back
        self.random_seed = random_seed
        self.lasso_alpha = lasso_alpha # Store lasso_alpha

        self.data_preprocessor = DataPreprocessor(
            stock_ticker=self.stock_ticker,
            years_of_data=self.years_of_data,
            random_seed=self.random_seed,
            lasso_alpha=self.lasso_alpha # Pass to DataPreprocessor
        )
        self.att_lstm_model = None
        # self.ensemble_model = EnsembleModel(...) # Removed
        self.data_scaler = None
        self.processed_df = None
        self.selected_features = None # To store selected feature names
        self.df_all_indicators = None # To store dataframe with all indicators for volatility analysis

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
        metrics_log = {} # To store all metrics

        # 1. Data Acquisition and Preprocessing
        print("Starting Data Preprocessing...")
        preprocessing_start_time = time.time()
        # Update to receive all return values from preprocess
        self.processed_df, self.data_scaler, self.selected_features, self.df_all_indicators = \
            self.data_preprocessor.preprocess()
        preprocessing_end_time = time.time()
        metrics_log['preprocessing_time_seconds'] = preprocessing_end_time - preprocessing_start_time
        print(f"Data Preprocessing completed in {metrics_log['preprocessing_time_seconds']:.2f} seconds.")
        print(f"Selected features by LASSO: {self.selected_features}")
        metrics_log['selected_features_count'] = len(self.selected_features)
        metrics_log['selected_features_names'] = self.selected_features


        if self.processed_df.empty:
            print("Error: Preprocessed data is empty. Aborting training.")
            return None # Return None to indicate failure

        target_column_name = self.processed_df.columns[-1] # Target is the last column
        print(f"Dynamically determined target column name for model: {target_column_name}")

        X_seq, y_seq = self._create_sequences(self.processed_df, target_column_name=target_column_name)
        if len(X_seq) == 0:
            print("Error: No sequences created from processed data. Aborting.")
            return None

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
        print(f"--- Final Training Parameters for ATT-LSTM ---")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
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
        print("Starting ATT-LSTM Model Training...")
        model_training_start_time = time.time()
        history = self.att_lstm_model.train(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            epochs=epochs, batch_size=batch_size
        )
        model_training_end_time = time.time()
        metrics_log['model_training_time_seconds'] = model_training_end_time - model_training_start_time
        print(f"ATT-LSTM Model Training completed in {metrics_log['model_training_time_seconds']:.2f} seconds.")
        if history and 'val_loss' in history.history:
            metrics_log['final_val_loss'] = history.history['val_loss'][-1]
            metrics_log['best_val_loss'] = min(history.history['val_loss'])
            metrics_log['epochs_trained'] = len(history.history['val_loss'])


        # Predict on test set for LSTM
        print("Starting predictions on test set...")
        prediction_start_time = time.time()
        att_lstm_test_preds_scaled = self.att_lstm_model.predict(X_test_seq).flatten()
        prediction_end_time = time.time()
        metrics_log['test_set_prediction_time_seconds'] = prediction_end_time - prediction_start_time
        print(f"Test set prediction completed in {metrics_log['test_set_prediction_time_seconds']:.2f} seconds.")

        # Inverse transform predictions and actual values to original scale
        dummy_preds_lstm = np.zeros((len(att_lstm_test_preds_scaled), self.processed_df.shape[1]))
        dummy_preds_lstm[:, self.processed_df.columns.get_loc(target_column_name)] = att_lstm_test_preds_scaled
        original_att_lstm_test_preds = self.data_scaler.inverse_transform(dummy_preds_lstm)[:, self.processed_df.columns.get_loc(target_column_name)]

        dummy_actuals = np.zeros((len(y_test_seq), self.processed_df.shape[1]))
        dummy_actuals[:, self.processed_df.columns.get_loc(target_column_name)] = y_test_seq
        original_y_test_seq = self.data_scaler.inverse_transform(dummy_actuals)[:, self.processed_df.columns.get_loc(target_column_name)]

        # Evaluate performance
        print("\n--- Model Performance on Test Set (Original Scale) ---")
        residuals_lstm = original_y_test_seq - original_att_lstm_test_preds

        # Overall Metrics
        metrics_log["overall_mse"] = mean_squared_error(original_y_test_seq, original_att_lstm_test_preds)
        metrics_log["overall_mae"] = mean_absolute_error(original_y_test_seq, original_att_lstm_test_preds)
        metrics_log["overall_rmse"] = np.sqrt(metrics_log["overall_mse"])

        mean_actuals = np.mean(original_y_test_seq)
        if mean_actuals == 0:
            metrics_log["overall_rmse_perc_mean"] = float('inf')
            metrics_log["overall_mape"] = float('inf') # Mean Absolute Percentage Error
        else:
            metrics_log["overall_rmse_perc_mean"] = (metrics_log["overall_rmse"] / mean_actuals) * 100
            metrics_log["overall_mape"] = np.mean(np.abs(residuals_lstm / original_y_test_seq)) * 100

        # Bias Metrics
        metrics_log["bias_me"] = np.mean(residuals_lstm) # Mean Error
        if mean_actuals == 0:
            metrics_log["bias_mpe"] = float('inf') # Mean Percentage Error
        else:
            metrics_log["bias_mpe"] = np.mean(residuals_lstm / original_y_test_seq) * 100

        print(f"ATT-LSTM - Overall MSE: {metrics_log['overall_mse']:.4f}, MAE: {metrics_log['overall_mae']:.4f}, RMSE: {metrics_log['overall_rmse']:.4f}")
        print(f"ATT-LSTM - Overall RMSE as % of Mean Actuals: {metrics_log['overall_rmse_perc_mean']:.2f}%")
        print(f"ATT-LSTM - Overall MAPE: {metrics_log['overall_mape']:.2f}%")
        print(f"ATT-LSTM - Bias (Mean Error): {metrics_log['bias_me']:.4f}")
        print(f"ATT-LSTM - Bias (Mean Percentage Error): {metrics_log['bias_mpe']:.2f}%")

        # Volatility-Specific Performance
        # Ensure self.df_all_indicators and 'ATR' column exist and are properly aligned with test_indices
        test_indices = self.processed_df.index[-len(original_y_test_seq):] # Get indices for the test set from processed_df

        if self.df_all_indicators is not None and 'ATR' in self.df_all_indicators.columns:
            # Align ATR data with the test set
            atr_series_full = self.df_all_indicators['ATR']
            # Ensure test_indices from processed_df (which is scaled and feature selected) can map back to df_all_indicators
            # This assumes that df_all_indicators (NaN dropped) still covers the range of processed_df
            if not test_indices.isin(atr_series_full.index).all():
                print("Warning: Some test_indices not found in df_all_indicators. Volatility analysis might be incomplete.")
                # Attempt to reindex, filling missing ATRs if any (not ideal, but a fallback)
                aligned_atr = atr_series_full.reindex(test_indices).fillna(method='ffill').fillna(method='bfill')
            else:
                aligned_atr = atr_series_full.loc[test_indices]

            if not aligned_atr.empty and len(aligned_atr) == len(original_y_test_seq):
                low_vol_threshold = aligned_atr.quantile(0.25)
                high_vol_threshold = aligned_atr.quantile(0.75)

                low_vol_mask = aligned_atr <= low_vol_threshold
                high_vol_mask = aligned_atr >= high_vol_threshold
                mid_vol_mask = (~low_vol_mask) & (~high_vol_mask)

                for period_name, mask in zip(["low_vol", "mid_vol", "high_vol"], [low_vol_mask, mid_vol_mask, high_vol_mask]):
                    if np.sum(mask) > 0:
                        metrics_log[f"{period_name}_mse"] = mean_squared_error(original_y_test_seq[mask], original_att_lstm_test_preds[mask])
                        metrics_log[f"{period_name}_mae"] = mean_absolute_error(original_y_test_seq[mask], original_att_lstm_test_preds[mask])
                        metrics_log[f"{period_name}_rmse"] = np.sqrt(metrics_log[f"{period_name}_mse"])
                        print(f"ATT-LSTM - {period_name.replace('_', ' ').title()} - MSE: {metrics_log[f'{period_name}_mse']:.4f}, MAE: {metrics_log[f'{period_name}_mae']:.4f}, RMSE: {metrics_log[f'{period_name}_rmse']:.4f} (Samples: {np.sum(mask)})")
                    else:
                        print(f"ATT-LSTM - No samples for {period_name.replace('_', ' ').title()} period.")
                        metrics_log[f"{period_name}_mse"] = np.nan
                        metrics_log[f"{period_name}_mae"] = np.nan
                        metrics_log[f"{period_name}_rmse"] = np.nan
            else:
                print("Could not perform volatility-specific performance analysis: ATR data alignment issue or insufficient data.")
        else:
            print("Could not perform volatility-specific performance analysis: ATR data not available in df_all_indicators.")


        plots_dir = "."
        print(f"Attempting to save plots to current directory: {os.path.abspath(plots_dir)}")

        try:
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
        except Exception as e:
            print(f"Error during plotting: {e}")

        print(f"\nVisualizations saved to '{plots_dir}' directory.")
        print("\n--- Full Model Training and Evaluation Complete ---")

        # Return all collected metrics along with predictions and actuals
        return {
            "att_lstm_preds": original_att_lstm_test_preds,
            "actual_values": original_y_test_seq,
            "metrics": metrics_log # Return the comprehensive metrics dictionary
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
    # --- Configurable parameters for a single test run ---
    test_stock_ticker = '^AEX'
    test_years_of_data = 3    # Changed to 3 years
    test_look_back = 60
    test_lasso_alpha = 0.005  # Using a default alpha for this run
    test_epochs = 150
    test_batch_size = 32
    results = None # Initialize results to None

    print(f"\n--- Starting Single Test Run: Ticker={test_stock_ticker}, Years={test_years_of_data}, LookBack={test_look_back}, Alpha={test_lasso_alpha} ---")

    full_model = FullStockPredictionModel(
        stock_ticker=test_stock_ticker,
        years_of_data=test_years_of_data,
        look_back=test_look_back,
        random_seed=42,
        lasso_alpha=test_lasso_alpha
    )

    overall_start_time = time.time()
    results = full_model.train_and_evaluate(
        epochs=test_epochs,
        batch_size=test_batch_size
    )
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    print(f"--- Full Model Training and Evaluation Took: {overall_duration:.2f} seconds ---")

    if results: # Check if results were returned (not empty on error)
        print("\n--- Final Results from Test Run ---")
        if results.get("att_lstm_preds") is not None:
             print("Final ATT-LSTM Predictions (first 5):", results["att_lstm_preds"][:5])
        print("Actual Values (first 5):", results["actual_values"][:5])

        print("\nDetailed Metrics from Test Run:")
        if "metrics" in results:
            for key, value in results["metrics"].items():
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                elif isinstance(value, list):
                     print(f"  {key.replace('_', ' ').title()}: {value}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print("  Metrics not available.")

        # --- Production Readiness Testing (Simulated) ---
        if hasattr(full_model, 'att_lstm_model') and full_model.att_lstm_model and \
           hasattr(full_model.att_lstm_model, 'model') and full_model.att_lstm_model.model is not None:
            print("\n--- Production Readiness Testing (Prediction Speed) ---")
            if full_model.processed_df is not None and not full_model.processed_df.empty:
                num_features = full_model.processed_df.shape[1]
                sample_raw_data = np.random.rand(full_model.look_back, num_features).astype(np.float32)
                single_instance_input = np.expand_dims(sample_raw_data, axis=0)

                print("Performing warm-up prediction for single instance...")
                _ = full_model.att_lstm_model.predict(single_instance_input)

                num_single_runs = 10
                single_pred_times = []
                print(f"Timing single instance prediction over {num_single_runs} runs...")
                for i in range(num_single_runs):
                    start_time = time.time()
                    _ = full_model.att_lstm_model.predict(single_instance_input)
                    end_time = time.time()
                    single_pred_times.append(end_time - start_time)

                avg_single_pred_time_ms = np.mean(single_pred_times) * 1000
                print(f"Average single instance prediction time: {avg_single_pred_time_ms:.2f} ms (over {num_single_runs} runs)")
                if results and "metrics" in results:
                    results["metrics"]["latency_single_pred_ms"] = avg_single_pred_time_ms

                batch_size_test = 32
                batch_input = np.random.rand(batch_size_test, full_model.look_back, num_features).astype(np.float32)

                print(f"Performing warm-up prediction for batch (size {batch_size_test})...")
                _ = full_model.att_lstm_model.predict(batch_input)

                num_batch_runs = 10
                batch_pred_times = []
                print(f"Timing batch prediction (size {batch_size_test}) over {num_batch_runs} runs...")
                for i in range(num_batch_runs):
                    start_time = time.time()
                    _ = full_model.att_lstm_model.predict(batch_input)
                    end_time = time.time()
                    batch_pred_times.append(end_time - start_time)

                avg_batch_pred_time_total_ms = np.mean(batch_pred_times) * 1000
                avg_batch_pred_time_per_instance_ms = (np.mean(batch_pred_times) / batch_size_test) * 1000

                print(f"Average batch ({batch_size_test} instances) prediction time: {avg_batch_pred_time_total_ms:.2f} ms (total)")
                print(f"Average per-instance prediction time in batch: {avg_batch_pred_time_per_instance_ms:.2f} ms (over {num_batch_runs} runs)")

                if results and "metrics" in results:
                    results["metrics"]["latency_batch_total_ms"] = avg_batch_pred_time_total_ms
                    results["metrics"]["latency_batch_per_instance_ms"] = avg_batch_pred_time_per_instance_ms
            else:
                print("Could not perform prediction speed test: processed_df not available.")
        else:
            print("ATT-LSTM model not available for production readiness testing.")
    else: # if results is None
        print("Model training and evaluation did not complete successfully for the test run.")

    # --- Previous loop for LASSO alpha (commented out for single test run) ---
    # lasso_alpha_values_to_test = [0.001, 0.005, 0.01, 0.02]
    # all_results_by_alpha = {}
    # for alpha_val in lasso_alpha_values_to_test:
    #     print(f"\n--- Running Full Model Training & Evaluation for LASSO alpha: {alpha_val} ---")
    #     full_model = FullStockPredictionModel(
    #         stock_ticker='^AEX',
    #         years_of_data=10, # This was the original value in the loop
    #         look_back=60,
    #         random_seed=42,
    #         lasso_alpha=alpha_val
    #     )
    #     # ... (rest of the loop as before, including train_and_evaluate, and storing/printing results) ...
    #     # ... and the summary print for all_results_by_alpha ...


