import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import modules
from .data_preprocessing_module import DataPreprocessor
from .att_lstm_module import ATTLSTMModel
# from .nsgm1n_module import NSGM1NModel # Removed
# from .ensemble_module import EnsembleModel # Removed
import matplotlib.pyplot as plt
import os
import time # Import time module for performance testing

# --- Module 5: Integrate and Test the Full Model ---

class FullStockPredictionModel:
    def __init__(self, stock_ticker="AEL", years_of_data=10, look_back=60, random_seed=42, lasso_alpha=0.005, use_differencing=False):
        self.stock_ticker = stock_ticker
        self.years_of_data = years_of_data
        self.look_back = look_back
        self.random_seed = random_seed
        self.lasso_alpha = lasso_alpha
        self.use_differencing = use_differencing # Store differencing choice

        self.data_preprocessor = DataPreprocessor(
            stock_ticker=self.stock_ticker,
            years_of_data=self.years_of_data,
            random_seed=self.random_seed,
            lasso_alpha=self.lasso_alpha,
            use_differencing=self.use_differencing # Pass to DataPreprocessor
        )
        self.att_lstm_model = None
        self.target_scaler = None # Scaler for the target variable
        self.processed_df = None
        self.selected_features = None
        self.df_all_indicators = None
        self.first_price_before_diff = None # For inverse differencing
        # self.plots_dir_path will be set dynamically in train_and_evaluate
        self.plots_dir_path = None
        self.actual_target_column_name_for_inv_transform = None # Store the original target name before it might be changed by differencing suffix

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

        # Store the original target column name from DataPreprocessor instance, before it might get a suffix
        # This assumes the target is 'Close' initially. If DataPreprocessor changes it (e.g. 'Close_AEX'),
        # this needs to be robust. For now, assuming 'Close' is the base.
        # A better way might be to get this from data_preprocessor after it determines the actual name.
        # Let's assume self.data_preprocessor.stock_ticker gives us '^AEX', so target is 'Close'.
        # If yfinance data adds ticker to column name, it becomes 'Close_^AEX'.
        # The data_preprocessor._apply_lasso_feature_selection now handles this flattening.
        # The target_column_name returned by processed_df.columns[-1] is the one to use for sequence creation.
        # However, for inverse differencing, we need the *original* non-differenced values.

        # DataPreprocessor.preprocess() now returns:
        # scaled_df, target_scaler, selected_features_names, df_with_all_indicators_cleaned, first_price_before_diff
        self.processed_df, self.target_scaler, self.selected_features, self.df_all_indicators, self.first_price_before_diff = \
            self.data_preprocessor.preprocess()

        preprocessing_end_time = time.time()
        metrics_log['preprocessing_time_seconds'] = preprocessing_end_time - preprocessing_start_time
        print(f"Data Preprocessing completed in {metrics_log['preprocessing_time_seconds']:.2f} seconds.")
        print(f"Selected features by LASSO: {self.selected_features}")
        metrics_log['selected_features_count'] = len(self.selected_features)
        metrics_log['selected_features_names'] = self.selected_features
        metrics_log['used_differencing'] = self.use_differencing
        if self.use_differencing:
            print(f"Differencing was used. First price before diff: {self.first_price_before_diff}")

        # --- Create unique directory for this run's plots ---
        # Use relevant parameters to name the directory for clarity
        run_specific_plot_dir_name = f"alpha_{self.lasso_alpha}_diff_{self.use_differencing}_lb_{self.look_back}_yrs_{self.years_of_data}"
        self.plots_dir_path = os.path.join("performance_evaluation_report", run_specific_plot_dir_name)
        os.makedirs(self.plots_dir_path, exist_ok=True)
        print(f"Plots for this run will be saved to: {os.path.abspath(self.plots_dir_path)}")
        metrics_log['plot_directory'] = self.plots_dir_path


        if self.processed_df.empty:
            print("Error: Preprocessed data is empty. Aborting training.")
            return None

        # This is the name of the target column in the *final* processed_df (could be differenced)
        current_target_column_name = self.processed_df.columns[-1]
        print(f"Target column name for model training (in processed_df): {current_target_column_name}")

        # Store the actual name of the target column from the *original* (but NaN-cleaned and LASSOed) data
        # This is needed if we have to reconstruct from differenced predictions.
        # df_all_indicators has original values but is already NaN-cleaned and has features.
        # The target column name in df_all_indicators should be the original name.
        # We need the target column name as it appears *before* any potential differencing.
        # This name is consistent in df_with_all_indicators_cleaned (returned by preprocess, now in self.df_all_indicators)
        # and in the original non-differenced processed_data_for_scaling.
        # Let's assume the *last* column of self.df_all_indicators (if it exists and is from before differencing)
        # or rely on a known convention if that's safer.
        # If self.df_all_indicators exists and is not empty, its last column should be the non-differenced target.
        if self.df_all_indicators is not None and not self.df_all_indicators.empty:
            self.actual_target_column_name_for_inv_transform = self.df_all_indicators.columns[-1]
            print(f"Actual target column name for inverse transform (from df_all_indicators): {self.actual_target_column_name_for_inv_transform}")
        else:
            # Fallback if df_all_indicators is not available (should not happen with current flow)
            self.actual_target_column_name_for_inv_transform = "Close" # Default assumption
            print(f"Warning: df_all_indicators not available, falling back to default target name for inv transform: {self.actual_target_column_name_for_inv_transform}")


        X_seq, y_seq = self._create_sequences(self.processed_df, target_column_name=current_target_column_name)
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
        # Define placeholder for best hyperparameters (to be updated after tuning)
        # These are example values and should be replaced by the actual results from tune_att_lstm.py
        # The look_back used here should also correspond to the one that yielded the best HPs.
        # For now, we'll keep the existing look_back from the class instance.
        # A more advanced setup would load these from a file saved by the tuning script.
        # The ATTLSTMModel will now use its internal defaults which have been updated
        # with the best hyperparameters found by the user.
        # No need to pass model_params if we want to use those defaults.
        # If a specific set of HPs for a specific run different from new defaults is needed,
        # then model_params should be populated and passed.
        # For this task, the goal is to use the user-provided HPs as the new standard.
        print(f"\n--- Initializing ATT-LSTM with updated default hyperparameters ---")
        print(f"--- Final Training Parameters for ATT-LSTM ---")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Look_back: {self.look_back}") # Display the look_back being used
        print(f"  Years of Data for training: {self.years_of_data}")
        print("  (Model will use hyperparameters set in att_lstm_module.py)")
        print("------------------------------------------------------------")

        input_shape_lstm = (X_train_seq.shape[1], X_train_seq.shape[2]) # (timesteps, features)

        # Instantiate ATTLSTMModel. It will use the new defaults set in its __init__.
        self.att_lstm_model = ATTLSTMModel(
            input_shape=input_shape_lstm,
            look_back=self.look_back,
            random_seed=self.random_seed
            # No model_params are passed, so it uses the defaults updated in att_lstm_module.py
        )

        # Build the model using its internal (now updated) default parameters
        self.att_lstm_model.build_model() # Since hp=None and no model_params, it will use its internal defaults.

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

        # The target_scaler is now specific to the target column.
        # We need to reshape predictions and actuals to be 2D for the scaler.
        att_lstm_test_preds_scaled_2d = att_lstm_test_preds_scaled.reshape(-1, 1)
        y_test_seq_2d = y_test_seq.reshape(-1, 1)

        # Inverse scale the predictions and actuals (which are potentially differenced and scaled)
        inv_scaled_preds = self.target_scaler.inverse_transform(att_lstm_test_preds_scaled_2d).flatten()
        inv_scaled_actuals = self.target_scaler.inverse_transform(y_test_seq_2d).flatten()

        # Get indices for the test set from processed_df to align with df_all_indicators for inverse differencing
        test_indices = self.processed_df.index[-len(y_test_seq):]


        if self.use_differencing:
            print("Applying inverse differencing to predictions and actuals...")
            if self.first_price_before_diff is None: # This was set at the class level
                # This implies an issue if it's None and use_differencing is True
                # However, the logic for getting price_before_first_pred below is more robust for test set.
                # For inverse differencing the actuals correctly, we need the actual price
                # from original data series just before the first actual difference in y_test_seq.
                # For inverse differencing the predictions, we also need that same price.
                pass


            if self.df_all_indicators is None or self.actual_target_column_name_for_inv_transform not in self.df_all_indicators.columns:
                 raise ValueError("df_all_indicators or the target column for inverse transform is not available for inverse differencing.")

            # Find the index of the data point in df_all_indicators that immediately precedes the first test point.
            # This requires test_indices to be valid for df_all_indicators.
            # df_all_indicators should have been aligned with processed_df (after potential row drop from diff)
            try:
                first_test_date_index_loc = self.df_all_indicators.index.get_loc(test_indices[0])
            except KeyError:
                raise ValueError(f"First test date {test_indices[0]} not found in df_all_indicators. Index alignment issue.")

            if first_test_date_index_loc == 0:
                # This means the first test sample is the very first sample in df_all_indicators.
                # We cannot get a "previous day" price from df_all_indicators.
                # In this specific scenario, self.first_price_before_diff (which is the first price of the *entire series*
                # before any differencing) would be the only reference if the first test point aligns with the first diff.
                # This case is complex and might indicate not enough historical data for the split or look_back.
                # For now, this error is a safeguard. A more robust solution might be needed if this is a common case.
                raise ValueError("Cannot get previous day price for inverse differencing: first test sample is the first data point in aligned df_all_indicators.")

            price_before_first_pred = self.df_all_indicators[self.actual_target_column_name_for_inv_transform].iloc[first_test_date_index_loc - 1]

            original_att_lstm_test_preds = price_before_first_pred + np.cumsum(inv_scaled_preds)
            original_y_test_seq = price_before_first_pred + np.cumsum(inv_scaled_actuals)
            print(f"  Reconstructed first actual price: {original_y_test_seq[0]:.2f} (from {price_before_first_pred:.2f} + {inv_scaled_actuals[0]:.2f})")
            print(f"  Reconstructed first predicted price: {original_att_lstm_test_preds[0]:.2f} (from {price_before_first_pred:.2f} + {inv_scaled_preds[0]:.2f})")

        else: # No differencing was used
            original_att_lstm_test_preds = inv_scaled_preds
            original_y_test_seq = inv_scaled_actuals

        # Evaluate performance
        print("\n--- Model Performance on Test Set (Original Scale) ---")
        residuals_lstm = original_y_test_seq - original_att_lstm_test_preds

        # Overall Metrics
        metrics_log["overall_mse"] = mean_squared_error(original_y_test_seq, original_att_lstm_test_preds)
        metrics_log["overall_mae"] = mean_absolute_error(original_y_test_seq, original_att_lstm_test_preds)
        metrics_log["overall_rmse"] = np.sqrt(metrics_log["overall_mse"])

        mean_actuals = np.mean(original_y_test_seq)
        if mean_actuals == 0: # Avoid division by zero
            metrics_log["overall_rmse_perc_mean"] = float('inf') # Or np.nan, or handle as error
            metrics_log["overall_mape"] = float('inf') if np.any(original_y_test_seq == 0) else np.mean(np.abs(residuals_lstm / original_y_test_seq)) * 100
        else:
            metrics_log["overall_rmse_perc_mean"] = (metrics_log["overall_rmse"] / mean_actuals) * 100
            # MAPE calculation needs care if original_y_test_seq contains zeros
            safe_actuals_for_mape = np.where(original_y_test_seq == 0, 1e-9, original_y_test_seq) # Replace 0s to avoid div by zero
            metrics_log["overall_mape"] = np.mean(np.abs(residuals_lstm / safe_actuals_for_mape)) * 100


        # Bias Metrics
        metrics_log["bias_me"] = np.mean(residuals_lstm) # Mean Error
        if mean_actuals == 0: # Avoid division by zero
            metrics_log["bias_mpe"] = float('inf') # Or np.nan
        else:
            # MPE calculation also needs care if original_y_test_seq contains zeros
            metrics_log["bias_mpe"] = np.mean(residuals_lstm / safe_actuals_for_mape) * 100


        print(f"ATT-LSTM - Overall MSE: {metrics_log['overall_mse']:.4f}, MAE: {metrics_log['overall_mae']:.4f}, RMSE: {metrics_log['overall_rmse']:.4f}")
        print(f"ATT-LSTM - Overall RMSE as % of Mean Actuals: {metrics_log['overall_rmse_perc_mean']:.2f}%")
        print(f"ATT-LSTM - Overall MAPE: {metrics_log['overall_mape']:.2f}%")
        print(f"ATT-LSTM - Bias (Mean Error): {metrics_log['bias_me']:.4f}")
        print(f"ATT-LSTM - Bias (Mean Percentage Error): {metrics_log['bias_mpe']:.2f}%")

        # Volatility-Specific Performance
        # Ensure self.df_all_indicators and 'ATR' column exist and are properly aligned with test_indices
        # test_indices was already defined before the inverse differencing block

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
                # Ensure there's variance in ATR to define quantiles, otherwise thresholds might be identical
                if aligned_atr.nunique() > 1:
                    low_vol_threshold = aligned_atr.quantile(0.25)
                    high_vol_threshold = aligned_atr.quantile(0.75)

                    # Handle cases where thresholds might be equal due to skewed distribution or few unique values
                    if low_vol_threshold == high_vol_threshold: # If all ATR values are similar or quantiles fall on same value
                        # Adjust logic: e.g., split into two halves or use fixed percentage if quantiles are problematic
                        # For simplicity, if they are equal, maybe define low as <= median and high as > median
                        median_vol = aligned_atr.median()
                        low_vol_mask = aligned_atr <= median_vol
                        high_vol_mask = aligned_atr > median_vol
                        mid_vol_mask = pd.Series(False, index=aligned_atr.index) # No mid-vol in this case
                        print(f"Note: ATR quantiles for low/high vol were equal. Using median split: Low <= {median_vol}, High > {median_vol}")

                    else:
                        low_vol_mask = aligned_atr <= low_vol_threshold
                        high_vol_mask = aligned_atr >= high_vol_threshold
                        mid_vol_mask = (~low_vol_mask) & (~high_vol_mask)
                else: # Not enough unique ATR values to form distinct quantiles
                    print("Warning: Not enough unique ATR values to perform robust volatility segmentation. Reporting overall metrics only for volatility.")
                    # Create masks that won't select anything or select all for 'overall' if needed
                    low_vol_mask = pd.Series(False, index=aligned_atr.index)
                    mid_vol_mask = pd.Series(False, index=aligned_atr.index)
                    high_vol_mask = pd.Series(False, index=aligned_atr.index)


                for period_name, mask in zip(["low_vol", "mid_vol", "high_vol"], [low_vol_mask, mid_vol_mask, high_vol_mask]):
                    # Ensure mask is boolean and aligns with data
                    mask_bool = mask.astype(bool)
                    if np.sum(mask_bool) > 0:
                        metrics_log[f"{period_name}_mse"] = mean_squared_error(original_y_test_seq[mask_bool], original_att_lstm_test_preds[mask_bool])
                        metrics_log[f"{period_name}_mae"] = mean_absolute_error(original_y_test_seq[mask_bool], original_att_lstm_test_preds[mask_bool])
                        metrics_log[f"{period_name}_rmse"] = np.sqrt(metrics_log[f"{period_name}_mse"])
                        print(f"ATT-LSTM - {period_name.replace('_', ' ').title()} - MSE: {metrics_log[f'{period_name}_mse']:.4f}, MAE: {metrics_log[f'{period_name}_mae']:.4f}, RMSE: {metrics_log[f'{period_name}_rmse']:.4f} (Samples: {np.sum(mask_bool)})")
                    else:
                        print(f"ATT-LSTM - No samples for {period_name.replace('_', ' ').title()} period.")
                        metrics_log[f"{period_name}_mse"] = np.nan
                        metrics_log[f"{period_name}_mae"] = np.nan
                        metrics_log[f"{period_name}_rmse"] = np.nan
            else:
                print("Could not perform volatility-specific performance analysis: ATR data alignment issue or insufficient data.")
                for period_name in ["low_vol", "mid_vol", "high_vol"]: # Ensure keys exist
                    metrics_log[f"{period_name}_mse"] = np.nan
                    metrics_log[f"{period_name}_mae"] = np.nan
                    metrics_log[f"{period_name}_rmse"] = np.nan

        else:
            print("Could not perform volatility-specific performance analysis: ATR data not available in df_all_indicators.")
            for period_name in ["low_vol", "mid_vol", "high_vol"]: # Ensure keys exist
                metrics_log[f"{period_name}_mse"] = np.nan
                metrics_log[f"{period_name}_mae"] = np.nan
                metrics_log[f"{period_name}_rmse"] = np.nan


        # Ensure the directory exists (using the instance variable)
        os.makedirs(self.plots_dir_path, exist_ok=True)
        print(f"Attempting to save plots to: {os.path.abspath(self.plots_dir_path)}")

        try:
            self._plot_predictions_vs_actuals_timeseries(
                test_indices, original_y_test_seq, original_att_lstm_test_preds,
                "ATT-LSTM Model Predictions vs Actuals",
                os.path.join(self.plots_dir_path, "full_run_att_lstm_preds_vs_actuals_timeseries.png")
            )
            self._plot_predictions_vs_actuals_scatter(
                original_y_test_seq, original_att_lstm_test_preds,
                "ATT-LSTM Model Predictions vs Actuals (Scatter)",
                os.path.join(self.plots_dir_path, "full_run_att_lstm_preds_vs_actuals_scatter.png")
            )
            self._plot_residuals_timeseries(
                test_indices, residuals_lstm,
                "ATT-LSTM Model Residuals Over Time",
                os.path.join(self.plots_dir_path, "full_run_att_lstm_residuals_timeseries.png")
            )
            self._plot_residuals_histogram(
                residuals_lstm,
                "ATT-LSTM Model Distribution of Residuals",
                os.path.join(self.plots_dir_path, "full_run_att_lstm_residuals_histogram.png")
            )
            # Call the new volatility performance plot
            if metrics_log: # Ensure metrics_log is available
                self._plot_volatility_performance_comparison(metrics_log)

        except Exception as e:
            print(f"Error during plotting: {e}")
            import traceback
            print("Traceback for plotting error:")
            traceback.print_exc()


        print(f"\nVisualizations saved to '{self.plots_dir_path}' directory.")
        print("\n--- Full Model Training and Evaluation Complete ---")

        # Return all collected metrics along with predictions and actuals
        return {
            "att_lstm_preds": original_att_lstm_test_preds,
            "actual_values": original_y_test_seq,
            "metrics": metrics_log # Return the comprehensive metrics dictionary
        }

    def _plot_volatility_performance_comparison(self, metrics_log, base_filename="volatility_performance_comparison.png"):
        """Plots a bar chart comparing performance metrics across volatility periods."""
        periods = ["low_vol", "mid_vol", "high_vol"]
        metrics_to_plot = ["rmse", "mae"] # Can extend to other metrics like mse

        for metric_name in metrics_to_plot:
            values = []
            labels = []
            valid_periods_for_metric = 0
            for period in periods:
                key = f"{period}_{metric_name}"
                if key in metrics_log and pd.notna(metrics_log[key]): # Check for np.nan explicitly
                    values.append(metrics_log[key])
                    labels.append(period.replace("_", " ").title())
                    valid_periods_for_metric +=1

            if valid_periods_for_metric < 1: # No data for any period for this metric
                print(f"Skipping {metric_name.upper()} volatility comparison plot: No valid data for any period.")
                continue

            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon'])
            plt.ylabel(metric_name.upper())
            plt.title(f"Model Performance ({metric_name.upper()}) by Volatility Period")
            plt.grid(axis='y', linestyle='--')

            # Add text labels on bars
            for bar_idx, bar in enumerate(bars):
                yval = bar.get_height()
                # Check if yval is NaN before trying to format it
                if pd.notna(yval):
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values, default=0), f'{yval:.2f}', ha='center', va='bottom')
                else: # Handle NaN case if it somehow slips through (should be caught by pd.notna(metrics_log[key]))
                    plt.text(bar.get_x() + bar.get_width()/2.0, 0, 'N/A', ha='center', va='bottom')


            filename = os.path.join(self.plots_dir_path, f"{metric_name}_" + base_filename) # self.plots_dir_path needs to be set
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"Volatility performance comparison plot ({metric_name}) saved to {filename}")


    # --- Plotting Helper Methods ---
    def _plot_predictions_vs_actuals_timeseries(self, x_values, actuals, predictions, title, filename):
        plt.figure(figsize=(12, 6))
        # Ensure x_values are appropriate for plotting (e.g., datetime objects if available and sensible)
        # If x_values are pandas DatetimeIndex, matplotlib handles them well.
        plt.plot(x_values, actuals, label='Actual Values', color='blue', marker='.', linestyle='-')
        plt.plot(x_values, predictions, label='Predicted Values', color='red', marker='.', linestyle='--')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Stock Price (Original Scale)")
        plt.legend()
        plt.grid(True)
        if isinstance(x_values, pd.DatetimeIndex) or (hasattr(x_values, 'dtype') and pd.api.types.is_datetime64_any_dtype(x_values)):
            plt.xticks(rotation=45) # Rotate if datetime
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _plot_predictions_vs_actuals_scatter(self, actuals, predictions, title, filename):
        plt.figure(figsize=(8, 8))
        plt.scatter(actuals, predictions, alpha=0.5)
        # Determine plot limits based on data range for y=x line
        min_val = min(np.min(actuals), np.min(predictions))
        max_val = max(np.max(actuals), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--') # y=x line
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
        if isinstance(x_values, pd.DatetimeIndex) or (hasattr(x_values, 'dtype') and pd.api.types.is_datetime64_any_dtype(x_values)):
            plt.xticks(rotation=45) # Rotate if datetime
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _plot_residuals_histogram(self, residuals, title, filename, bins=50):
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=bins, edgecolor='black', alpha=0.7, density=True) # Use density=True for normalized histogram
        # Fit a normal distribution to the data
        try:
            from scipy.stats import norm
            mu, std = norm.fit(residuals)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'Fit results: mu={mu:.2f}, std={std:.2f}')
        except ImportError:
            print("SciPy not available, skipping normal distribution fit on residual histogram.")
            pass # SciPy might not be installed, or an error during fit

        plt.title(title)
        plt.xlabel("Residual Value")
        plt.ylabel("Density") # Y-axis is density due to density=True
        plt.grid(True)
        # axvline for mean is still useful even with density
        plt.axvline(residuals.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {residuals.mean():.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# Example Usage (Run the full model)
if __name__ == '__main__':
    # --- Configuration for Experimental Runs ---
    # General parameters
    run_stock_ticker = '^AEX'
    run_years_of_data = 3  # Updated to 3 years as per request
    run_look_back = 60     # Set to best look_back from tuning
    run_epochs = 20         # Reduced epochs for quicker tests (can be increased for final run)
    run_batch_size = 32
    run_use_differencing = False # Set to True to test with differencing

    # LASSO alpha values to test
    lasso_alpha_values_to_test = [0.005, 0.01] # Reduced set for quicker tests
    # To run a single test with a specific alpha:
    # lasso_alpha_values_to_test = [0.005]

    all_run_results = {}
    results_df_list = [] # For creating a summary DataFrame

    print(f"--- Starting Experimental Runs ---")
    print(f"Stock: {run_stock_ticker}, Years: {run_years_of_data}, Look_back: {run_look_back}, Epochs: {run_epochs}, Differencing: {run_use_differencing}")

    for alpha_val in lasso_alpha_values_to_test:
        print(f"\n--- Running Experiment with LASSO alpha: {alpha_val}, Differencing: {run_use_differencing} ---")

        # Create a unique ID for this run configuration for storing results
        run_id = f"alpha_{alpha_val}_diff_{run_use_differencing}_lb_{run_look_back}_yrs_{run_years_of_data}"

        full_model_instance = FullStockPredictionModel(
            stock_ticker=run_stock_ticker,
            years_of_data=run_years_of_data,
            look_back=run_look_back,
            random_seed=42,
            lasso_alpha=alpha_val,
            use_differencing=run_use_differencing
        )

        run_start_time = time.time()
        # The train_and_evaluate method uses the HPs defined within its scope
        # (currently placeholders, ideally loaded based on tuned look_back)
        current_run_results = full_model_instance.train_and_evaluate(
            epochs=run_epochs,
            batch_size=run_batch_size
            # test_size and val_size use defaults in train_and_evaluate
        )
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time
        print(f"--- Experiment with {run_id} Took: {run_duration:.2f} seconds ---")

        if current_run_results and "metrics" in current_run_results:
            all_run_results[run_id] = current_run_results["metrics"]

            # Prepare data for DataFrame summary
            summary_data = {
                "run_id": run_id,
                "lasso_alpha": alpha_val,
                "differencing": run_use_differencing,
                "look_back": run_look_back,
                "years_data": run_years_of_data,
                **current_run_results["metrics"] # Unpack all metrics
            }
            results_df_list.append(summary_data)

            print(f"  Key Metrics for {run_id}:")
            print(f"    RMSE: {current_run_results['metrics'].get('overall_rmse', 'N/A'):.4f}")
            print(f"    MAPE: {current_run_results['metrics'].get('overall_mape', 'N/A'):.2f}%")
            print(f"    Bias (ME): {current_run_results['metrics'].get('bias_me', 'N/A'):.4f}")
            print(f"    Selected Features: {current_run_results['metrics'].get('selected_features_count', 'N/A')}")

            # Production readiness testing (prediction speed)
            if hasattr(full_model_instance, 'att_lstm_model') and full_model_instance.att_lstm_model and \
               hasattr(full_model_instance.att_lstm_model, 'model') and full_model_instance.att_lstm_model.model is not None:
                if full_model_instance.processed_df is not None and not full_model_instance.processed_df.empty:
                    # Ensure processed_df has more than 0 columns before accessing shape[1]
                    if full_model_instance.processed_df.shape[1] > 0:
                        num_features = full_model_instance.processed_df.shape[1] -1 # Exclude target for input
                        # Ensure num_features matches model's expected input features.
                        # The model input shape is (look_back, num_features_in_X_seq)
                        # X_seq created by _create_sequences uses all columns of processed_df.
                        # So num_features for sample_raw_data should be processed_df.shape[1]
                        actual_num_features_for_model_input = full_model_instance.att_lstm_model.input_shape[1]

                        sample_raw_data = np.random.rand(full_model_instance.look_back, actual_num_features_for_model_input).astype(np.float32)
                        single_instance_input = np.expand_dims(sample_raw_data, axis=0)

                        # Warm-up prediction
                        try:
                            _ = full_model_instance.att_lstm_model.predict(single_instance_input)
                        except Exception as e:
                            print(f"    Error during warm-up prediction for latency test: {e}")
                            avg_single_pred_time_ms = np.nan # Cannot test latency
                        else:
                            single_pred_times = []
                            for _ in range(10): # Number of iterations for averaging
                                pred_start_time = time.time()
                                _ = full_model_instance.att_lstm_model.predict(single_instance_input)
                                single_pred_times.append(time.time() - pred_start_time)
                            avg_single_pred_time_ms = np.mean(single_pred_times) * 1000

                        all_run_results[run_id]["latency_single_pred_ms"] = avg_single_pred_time_ms
                        # Check if results_df_list is not empty before trying to access its last element
                        if results_df_list:
                            results_df_list[-1]["latency_single_pred_ms"] = avg_single_pred_time_ms
                        print(f"    Avg Single Prediction Time: {avg_single_pred_time_ms:.2f} ms")
                    else:
                        print("    Skipping latency test: processed_df has no columns.")
                        all_run_results[run_id]["latency_single_pred_ms"] = np.nan
                        if results_df_list:
                             results_df_list[-1]["latency_single_pred_ms"] = np.nan
                else:
                    print("    Skipping latency test: processed_df is None or empty.")
                    all_run_results[run_id]["latency_single_pred_ms"] = np.nan
                    if results_df_list:
                        results_df_list[-1]["latency_single_pred_ms"] = np.nan
            else:
                print("    Skipping latency test: ATT-LSTM model not available.")
                all_run_results[run_id]["latency_single_pred_ms"] = np.nan
                if results_df_list:
                    results_df_list[-1]["latency_single_pred_ms"] = np.nan

        else:
            print(f"  Run {run_id} did not produce results or metrics.")
        print(f"--- End of Experiment for {run_id} ---")

    print("\n\n--- Summary of All Experimental Runs (Console) ---")
    if all_run_results:
        for run_id_key, metrics in all_run_results.items():
            print(f"\nResults for Configuration: {run_id_key}")
            print(f"  Plot Directory: {metrics.get('plot_directory', 'N/A')}")
            print(f"  Selected Features Count: {metrics.get('selected_features_count', 'N/A')}")
            overall_rmse_val = metrics.get('overall_rmse')
            overall_rmse_str = f"{overall_rmse_val:.4f}" if pd.notna(overall_rmse_val) else 'N/A'
            print(f"  Overall RMSE: {overall_rmse_str}")
            overall_mape_val = metrics.get('overall_mape')
            overall_mape_str = f"{overall_mape_val:.2f}%" if pd.notna(overall_mape_val) else 'N/A'
            print(f"  Overall MAPE: {overall_mape_str}")
            bias_me_val = metrics.get('bias_me')
            bias_me_str = f"{bias_me_val:.4f}" if pd.notna(bias_me_val) else 'N/A'
            print(f"  Bias (Mean Error): {bias_me_str}")
            latency_val = metrics.get('latency_single_pred_ms')
            latency_str = f"{latency_val:.2f}" if pd.notna(latency_val) else 'N/A'
            print(f"  Latency (ms): {latency_str}")
            # Add more metrics as needed
    else:
        print("No results collected from experimental runs for console summary.")

    # --- Save all_run_results to a JSON file and DataFrame to CSV ---
    if all_run_results:
        # Save to JSON
        results_json_path = os.path.join("performance_evaluation_report", "all_experimental_run_metrics.json")
        os.makedirs("performance_evaluation_report", exist_ok=True)
        try:
            # Convert list of feature names to string for JSON compatibility if they are lists
            serializable_results = {}
            for run_key, metrics_dict in all_run_results.items():
                serializable_metrics = {}
                for k, v in metrics_dict.items():
                    if k == 'selected_features_names' and isinstance(v, list):
                        serializable_metrics[k] = ", ".join(v)
                    elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)): # Ensure numpy types are converted
                        serializable_metrics[k] = float(v) if pd.notna(v) else None # Convert to float or None
                    elif pd.isna(v): # Handle other NaNs (e.g. from empty metrics)
                        serializable_metrics[k] = None
                    else:
                        serializable_metrics[k] = v
                serializable_results[run_key] = serializable_metrics
            with open(results_json_path, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=lambda x: x if x is not None else 'NaN') # Handle None for json
        except Exception as e:
            print(f"Error saving metrics to JSON: {e}")
            import traceback
            print("Traceback for JSON saving error:")
            traceback.print_exc()


        # Save to CSV
        results_csv_path = os.path.join("performance_evaluation_report", "all_experimental_run_summary.csv")
        try:
            summary_df = pd.DataFrame(results_df_list)
            # Convert list of feature names to string for CSV compatibility
            if 'selected_features_names' in summary_df.columns:
                 summary_df['selected_features_names'] = summary_df['selected_features_names'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
            summary_df.to_csv(results_csv_path, index=False)
            print(f"Summary of experimental run metrics saved to CSV: {results_csv_path}")
        except Exception as e:
            print(f"Error saving summary metrics to CSV: {e}")
            import traceback
            print("Traceback for CSV saving error:")
            traceback.print_exc()


    print("\n--- All Experimental Runs Complete ---")
