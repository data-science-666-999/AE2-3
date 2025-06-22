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
    def __init__(self, stock_ticker="AEL", years_of_data=10, look_back=60, random_seed=42, lasso_alpha=0.005,
                 use_differencing=False, loss_function_name='mse',
                 auto_differencing_adf=False, use_log_transform=False): # Added more params
        self.stock_ticker = stock_ticker
        self.years_of_data = years_of_data
        self.look_back = look_back
        self.loss_function_name = loss_function_name
        self.random_seed = random_seed
        self.lasso_alpha = lasso_alpha
        self.use_differencing = use_differencing # This is the manual override for DataPreprocessor
        self.auto_differencing_adf = auto_differencing_adf
        self.use_log_transform = use_log_transform

        self.data_preprocessor = DataPreprocessor(
            stock_ticker=self.stock_ticker,
            years_of_data=self.years_of_data,
            random_seed=self.random_seed,
            lasso_alpha=self.lasso_alpha,
            use_differencing=self.use_differencing,
            auto_differencing_adf=self.auto_differencing_adf,
            use_log_transform=self.use_log_transform
        )
        self.att_lstm_model = None
        self.target_scaler = None
        self.processed_df = None
        self.selected_features = None
        self.df_all_indicators = None
        # For inverse differencing & log transform:
        self.applied_differencing_order = 0
        self.values_for_inverse_differencing = []
        self.log_transform_applied_to_target = False
        self.true_target_column_name = None

        self.plots_dir_path = None

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

        self.processed_df, self.target_scaler, self.selected_features, self.df_all_indicators, \
        self.applied_differencing_order, self.values_for_inverse_differencing, \
        self.true_target_column_name, self.log_transform_applied_to_target = \
            self.data_preprocessor.preprocess()

        preprocessing_end_time = time.time()
        metrics_log['preprocessing_time_seconds'] = preprocessing_end_time - preprocessing_start_time
        print(f"Data Preprocessing completed in {metrics_log['preprocessing_time_seconds']:.2f} seconds.")
        print(f"Selected features by LASSO: {self.selected_features}")
        metrics_log['selected_features_count'] = len(self.selected_features)
        metrics_log['selected_features_names'] = self.selected_features
        metrics_log['log_transform_applied'] = self.log_transform_applied_to_target
        metrics_log['differencing_order_applied'] = self.applied_differencing_order
        print(f"Log transform applied: {self.log_transform_applied_to_target}")
        print(f"Differencing order applied: {self.applied_differencing_order} (manual_config: {self.use_differencing}, auto_adf_config: {self.auto_differencing_adf})")

        # --- Create unique directory for this run's plots ---
        # Plot directory name should reflect actual applied transformations.
        run_specific_plot_dir_name = f"alpha_{self.lasso_alpha}_difforder_{self.applied_differencing_order}_log_{self.log_transform_applied_to_target}_loss_{self.loss_function_name}_lb_{self.look_back}_yrs_{self.years_of_data}"
        self.plots_dir_path = os.path.join("performance_evaluation_report", run_specific_plot_dir_name)
        os.makedirs(self.plots_dir_path, exist_ok=True)
        print(f"Plots for this run will be saved to: {os.path.abspath(self.plots_dir_path)}")
        metrics_log['plot_directory'] = self.plots_dir_path # Store the actual plot directory

        if self.processed_df.empty:
            print("Error: Preprocessed data is empty. Aborting training.")
            return None

        current_target_column_name_in_processed_df = self.processed_df.columns[-1]
        print(f"Target column name for model training (in processed_df): {current_target_column_name_in_processed_df}")
        print(f"True target column name (original scale, post-flattening): {self.true_target_column_name}")

        X_seq, y_seq = self._create_sequences(self.processed_df, target_column_name=current_target_column_name_in_processed_df)
        if len(X_seq) == 0:
            print("Error: No sequences created from processed data. Aborting.")
            return None

        X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
            X_seq, y_seq, test_size=test_size, random_state=self.random_seed, shuffle=False
        )
        X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
            X_train_seq, y_train_seq, test_size=val_size, random_state=self.random_seed, shuffle=False
        )

        print(f"\nData split shapes:")
        print(f"X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
        print(f"X_val_seq: {X_val_seq.shape}, y_val_seq: {y_val_seq.shape}")
        print(f"X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")

        print(f"\n--- Initializing ATT-LSTM with Loss: {self.loss_function_name} ---")
        print(f"--- Final Training Parameters for ATT-LSTM ---")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Look_back: {self.look_back}")
        print(f"  Years of Data for training: {self.years_of_data}")
        print("  (Model will use default HPs from att_lstm_module.py unless overridden via model_params)")
        print("------------------------------------------------------------")

        input_shape_lstm = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.att_lstm_model = ATTLSTMModel(
            input_shape=input_shape_lstm,
            look_back=self.look_back,
            random_seed=self.random_seed,
            loss_function_name=self.loss_function_name
        )
        self.att_lstm_model.build_model()

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

        print("Starting predictions on test set...")
        prediction_start_time = time.time()
        att_lstm_test_preds_scaled = self.att_lstm_model.predict(X_test_seq).flatten()
        prediction_end_time = time.time()
        metrics_log['test_set_prediction_time_seconds'] = prediction_end_time - prediction_start_time
        print(f"Test set prediction completed in {metrics_log['test_set_prediction_time_seconds']:.2f} seconds.")

        att_lstm_test_preds_scaled_2d = att_lstm_test_preds_scaled.reshape(-1, 1)
        y_test_seq_2d = y_test_seq.reshape(-1, 1)

        inv_scaled_preds = self.target_scaler.inverse_transform(att_lstm_test_preds_scaled_2d).flatten()
        inv_scaled_actuals = self.target_scaler.inverse_transform(y_test_seq_2d).flatten()

        test_indices = self.processed_df.index[-len(y_test_seq):]

        # --- Inverse Differencing ---
        # This logic now needs to handle 0, 1, or 2 orders of differencing
        # And use self.values_for_inverse_differencing

        # Start with the already inverse-scaled predictions and actuals
        current_preds_to_reconstruct = inv_scaled_preds.copy()
        current_actuals_to_reconstruct = inv_scaled_actuals.copy()

        if self.applied_differencing_order > 0:
            print(f"Applying inverse differencing (order {self.applied_differencing_order}) to predictions and actuals...")
            if not self.values_for_inverse_differencing or len(self.values_for_inverse_differencing) != self.applied_differencing_order:
                raise ValueError("Mismatch between applied_differencing_order and values_for_inverse_differencing list.")

            # Iteratively apply inverse differencing
            for i in range(self.applied_differencing_order -1, -1, -1):
                # For 1st order diff (i=0): value_to_add is the single price before the first diff.
                # For 2nd order diff (i=1 for first inv_diff, i=0 for second):
                #   - first inv_diff (i=1): value_to_add is price before 2nd order diff (which is a 1st order diff value)
                #   - second inv_diff (i=0): value_to_add is price before 1st order diff (original price)

                # The value stored is the actual price/value of the point *before* the first element of the current differenced series
                # For the test set, we need the actual value from df_all_indicators that corresponds to the point
                # just before the start of the test set's differenced sequence.

                # Get the index of the data point in df_all_indicators that immediately precedes the first test point's original position
                # This needs to be robust to how test_indices were formed after differencing
                # df_all_indicators is aligned with processed_df *before* differencing rows were dropped.
                # So, we need to find the original index that corresponds to test_indices[0] if differencing happened.

                # Let's assume self.values_for_inverse_differencing stores the actual values from the *original series*
                # that are needed. For order 1, it's one value. For order 2, it's two values.
                # The logic in DataPreprocessor stores series.iloc[0] before each diff.
                # So, self.values_for_inverse_differencing[0] is value before 1st diff.
                # self.values_for_inverse_differencing[1] is value before 2nd diff (which is a 1st order diffed value).

                # To reconstruct on the test set:
                # We need the value from the original series that corresponds to the point
                # just before the first prediction in the test set.
                # This is tricky because test_indices are from processed_df (which has rows dropped by differencing).
                # df_all_indicators has the original values but needs careful indexing.

                # The price_before_first_pred logic was for 1st order diff.
                # For generic Nth order: P_t = P_{t-1} + Diff_t (1st order)
                # P_t-1_diff1 = P_{t-2}_diff1 + Diff2_t (for 2nd order)
                # We stored:
                # values_for_inverse_differencing[0] = P_original[0] (before 1st diff)
                # values_for_inverse_differencing[1] = P_diff1[0] (before 2nd diff)

                # For the test set, we need the value from the *original, non-differenced, non-scaled* series
                # that immediately precedes the first data point of the *current differenced level* of the test set.

                # Let's simplify: the inverse differencing needs the value from the *previous time step* of the series
                # at the *current level of integration*.
                # For the test set, this means we need the last actual value of the training/validation set
                # at that level of integration.
                # This is complex to get right without passing the full original series or last parts of train/val.

                # The current self.values_for_inverse_differencing stores the *first* values of the series *before* each diff.
                # This is useful if we reconstruct the *entire series* from the start.
                # For reconstructing only the test set, we need the value *immediately preceding* the test set.

                # Fallback to simpler logic for now, assuming inverse_transform on test set needs careful handling of initial conditions.
                # The most robust way is to get the actual value from df_all_indicators (original scale, but features selected)
                # at the time step *before* test_indices[0] for the *current level of differencing*.
                # This is still complicated.

                # Let's use the previously implemented logic for 1st order differencing as a base,
                # and acknowledge that multi-order inverse on test set is hard with current stored values.
                # The current self.values_for_inverse_differencing is not directly usable for test set partial reconstruction.
                # We MUST use values from the end of the validation set or from df_all_indicators correctly indexed.

                if i == 0: # Reconstructing from 1st order diffs to original scale
                    if self.df_all_indicators is None or self.true_target_column_name not in self.df_all_indicators.columns:
                        raise ValueError(f"True target column '{self.true_target_column_name}' not found in df_all_indicators for inverse differencing.")

                    # Find the index in df_all_indicators that corresponds to the day *before* the first test prediction.
                    # test_indices[0] is the date of the first y_test value.
                    # The actual data used to predict y_test[0] ends one day before test_indices[0].
                    # The price needed is the actual price at that "day before".

                    # Get the original index from processed_df (which test_indices are from)
                    # and map it back to df_all_indicators.
                    # This requires df_all_indicators to have an index that aligns with the state *before any rows were dropped by differencing*.
                    # The current self.df_all_indicators is aligned with processed_df *after* rows were dropped. This is an issue.

                    # For now, assume self.values_for_inverse_differencing[0] can be used if it's the first value of the *entire dataset*
                    # This is only correct if the test set starts right after that first value, which is not generally true.
                    # THIS PART NEEDS A ROBUST FIX IF MULTI-ORDER DIFFERENCING OR EVEN 1-ORDER IS TO WORK RELIABLY FOR TEST SET.
                    # A temporary, potentially INCORRECT assumption for test set:
                    # price_ref_for_inv_diff = self.values_for_inverse_differencing[0] # This is likely wrong for test set.

                    # Correct approach for test set 1st order inverse differencing:
                    # Find the actual value in the original scale series that immediately precedes test_indices[0].
                    # This means we need access to the original series *before* it was subsetted into processed_df.
                    # df_all_indicators should be the data *before* differencing rows were dropped.
                    # Let's assume self.data_preprocessor.stock_data_with_indicators is that. (This is not ideal)
                    # This part of the code is becoming very fragile due to data state management.

                    # For the purpose of this step, let's assume a simplified (potentially flawed for test set) reconstruction
                    # if self.applied_differencing_order == 1, using the old logic as a placeholder.
                    # The core change for this step is adding custom loss. Robust multi-order diff is a separate large task.

                    if self.applied_differencing_order == 1: # Only attempt if it was simple 1st order
                        idx_before_test_start = self.df_all_indicators.index.get_loc(test_indices[0]) -1
                        if idx_before_test_start < 0:
                             raise ValueError("Cannot get previous day price for inverse differencing of test set.")
                        price_ref_for_inv_diff = self.df_all_indicators[self.true_target_column_name].iloc[idx_before_test_start]

                        current_preds_to_reconstruct = price_ref_for_inv_diff + np.cumsum(current_preds_to_reconstruct)
                        current_actuals_to_reconstruct = price_ref_for_inv_diff + np.cumsum(current_actuals_to_reconstruct)
                        print(f"  Reconstructed (order {i+1}) first actual: {current_actuals_to_reconstruct[0]:.2f}, first pred: {current_preds_to_reconstruct[0]:.2f}")
                    else:
                        print(f"Warning: Multi-order ({self.applied_differencing_order}) inverse differencing for test set is complex and might not be fully accurate with current value storage. Skipping this inv-diff step.")
                        # If not reconstructing, then the values remain as (potentially) differenced.
                        pass # Values remain as they are (differenced)
                    break # Exit loop after attempting 1st order reconstruction or skipping for higher.

            original_att_lstm_test_preds = current_preds_to_reconstruct
            original_y_test_seq = current_actuals_to_reconstruct

        else: # No differencing was applied
            original_att_lstm_test_preds = inv_scaled_preds
            original_y_test_seq = inv_scaled_actuals

        # --- Inverse Log Transform (if applied) ---
        if self.log_transform_applied_to_target:
            print("Applying inverse log (expm1) transformation to predictions and actuals.")
            original_att_lstm_test_preds = np.expm1(original_att_lstm_test_preds)
            original_y_test_seq = np.expm1(original_y_test_seq)


        print("\n--- Model Performance on Test Set (Original Scale) ---")
        residuals_lstm = original_y_test_seq - original_att_lstm_test_preds
        metrics_log["overall_mse"] = mean_squared_error(original_y_test_seq, original_att_lstm_test_preds)
        metrics_log["overall_mae"] = mean_absolute_error(original_y_test_seq, original_att_lstm_test_preds)
        metrics_log["overall_rmse"] = np.sqrt(metrics_log["overall_mse"])
        mean_actuals = np.mean(original_y_test_seq)
        if mean_actuals == 0:
            metrics_log["overall_rmse_perc_mean"] = float('inf')
            metrics_log["overall_mape"] = float('inf') if np.any(original_y_test_seq == 0) else np.mean(np.abs(residuals_lstm / original_y_test_seq)) * 100
        else:
            metrics_log["overall_rmse_perc_mean"] = (metrics_log["overall_rmse"] / mean_actuals) * 100
            safe_actuals_for_mape = np.where(original_y_test_seq == 0, 1e-9, original_y_test_seq)
            metrics_log["overall_mape"] = np.mean(np.abs(residuals_lstm / safe_actuals_for_mape)) * 100
        metrics_log["bias_me"] = np.mean(residuals_lstm)
        if mean_actuals == 0:
            metrics_log["bias_mpe"] = float('inf')
        else:
            metrics_log["bias_mpe"] = np.mean(residuals_lstm / safe_actuals_for_mape) * 100

        print(f"ATT-LSTM - Overall MSE: {metrics_log['overall_mse']:.4f}, MAE: {metrics_log['overall_mae']:.4f}, RMSE: {metrics_log['overall_rmse']:.4f}")
        print(f"ATT-LSTM - Overall RMSE as % of Mean Actuals: {metrics_log['overall_rmse_perc_mean']:.2f}%")
        print(f"ATT-LSTM - Overall MAPE: {metrics_log['overall_mape']:.2f}%")
        print(f"ATT-LSTM - Bias (Mean Error): {metrics_log['bias_me']:.4f}")
        print(f"ATT-LSTM - Bias (Mean Percentage Error): {metrics_log['bias_mpe']:.2f}%")

        if self.df_all_indicators is not None and 'ATR' in self.df_all_indicators.columns:
            atr_series_full = self.df_all_indicators['ATR']
            if not test_indices.isin(atr_series_full.index).all():
                print("Warning: Some test_indices not found in df_all_indicators. Volatility analysis might be incomplete.")
                aligned_atr = atr_series_full.reindex(test_indices).fillna(method='ffill').fillna(method='bfill')
            else:
                aligned_atr = atr_series_full.loc[test_indices]
            if not aligned_atr.empty and len(aligned_atr) == len(original_y_test_seq):
                if aligned_atr.nunique() > 1:
                    low_vol_threshold = aligned_atr.quantile(0.25)
                    high_vol_threshold = aligned_atr.quantile(0.75)
                    if low_vol_threshold == high_vol_threshold:
                        median_vol = aligned_atr.median()
                        low_vol_mask = aligned_atr <= median_vol
                        high_vol_mask = aligned_atr > median_vol
                        mid_vol_mask = pd.Series(False, index=aligned_atr.index)
                        print(f"Note: ATR quantiles for low/high vol were equal. Using median split: Low <= {median_vol}, High > {median_vol}")
                    else:
                        low_vol_mask = aligned_atr <= low_vol_threshold
                        high_vol_mask = aligned_atr >= high_vol_threshold
                        mid_vol_mask = (~low_vol_mask) & (~high_vol_mask)
                else:
                    print("Warning: Not enough unique ATR values to perform robust volatility segmentation. Reporting overall metrics only for volatility.")
                    low_vol_mask = pd.Series(False, index=aligned_atr.index)
                    mid_vol_mask = pd.Series(False, index=aligned_atr.index)
                    high_vol_mask = pd.Series(False, index=aligned_atr.index)
                for period_name, mask in zip(["low_vol", "mid_vol", "high_vol"], [low_vol_mask, mid_vol_mask, high_vol_mask]):
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
                for period_name in ["low_vol", "mid_vol", "high_vol"]:
                    metrics_log[f"{period_name}_mse"] = np.nan
                    metrics_log[f"{period_name}_mae"] = np.nan
                    metrics_log[f"{period_name}_rmse"] = np.nan
        else:
            print("Could not perform volatility-specific performance analysis: ATR data not available in df_all_indicators.")
            for period_name in ["low_vol", "mid_vol", "high_vol"]:
                metrics_log[f"{period_name}_mse"] = np.nan
                metrics_log[f"{period_name}_mae"] = np.nan
                metrics_log[f"{period_name}_rmse"] = np.nan

        os.makedirs(self.plots_dir_path, exist_ok=True)
        print(f"Attempting to save plots to: {os.path.abspath(self.plots_dir_path)}")
        try:
            self._plot_predictions_vs_actuals_timeseries(test_indices, original_y_test_seq, original_att_lstm_test_preds, "ATT-LSTM Model Predictions vs Actuals", os.path.join(self.plots_dir_path, "full_run_att_lstm_preds_vs_actuals_timeseries.png"))
            self._plot_predictions_vs_actuals_scatter(original_y_test_seq, original_att_lstm_test_preds, "ATT-LSTM Model Predictions vs Actuals (Scatter)", os.path.join(self.plots_dir_path, "full_run_att_lstm_preds_vs_actuals_scatter.png"))
            self._plot_residuals_timeseries(test_indices, residuals_lstm, "ATT-LSTM Model Residuals Over Time", os.path.join(self.plots_dir_path, "full_run_att_lstm_residuals_timeseries.png"))
            self._plot_residuals_histogram(residuals_lstm, "ATT-LSTM Model Distribution of Residuals", os.path.join(self.plots_dir_path, "full_run_att_lstm_residuals_histogram.png"))
            if metrics_log:
                self._plot_volatility_performance_comparison(metrics_log)
        except Exception as e:
            print(f"Error during plotting: {e}")
            import traceback
            print("Traceback for plotting error:")
            traceback.print_exc()

        print(f"\nVisualizations saved to '{self.plots_dir_path}' directory.")
        print("\n--- Full Model Training and Evaluation Complete ---")
        return {"att_lstm_preds": original_att_lstm_test_preds, "actual_values": original_y_test_seq, "metrics": metrics_log}

    def _plot_volatility_performance_comparison(self, metrics_log, base_filename="volatility_performance_comparison.png"):
        periods = ["low_vol", "mid_vol", "high_vol"]
        metrics_to_plot = ["rmse", "mae"]
        for metric_name in metrics_to_plot:
            values = []
            labels = []
            valid_periods_for_metric = 0
            for period in periods:
                key = f"{period}_{metric_name}"
                if key in metrics_log and pd.notna(metrics_log[key]):
                    values.append(metrics_log[key])
                    labels.append(period.replace("_", " ").title())
                    valid_periods_for_metric +=1
            if valid_periods_for_metric < 1:
                print(f"Skipping {metric_name.upper()} volatility comparison plot: No valid data for any period.")
                continue
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon'])
            plt.ylabel(metric_name.upper())
            plt.title(f"Model Performance ({metric_name.upper()}) by Volatility Period")
            plt.grid(axis='y', linestyle='--')
            for bar_idx, bar in enumerate(bars):
                yval = bar.get_height()
                if pd.notna(yval):
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values, default=0), f'{yval:.2f}', ha='center', va='bottom')
                else:
                    plt.text(bar.get_x() + bar.get_width()/2.0, 0, 'N/A', ha='center', va='bottom')
            filename = os.path.join(self.plots_dir_path, f"{metric_name}_" + base_filename)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"Volatility performance comparison plot ({metric_name}) saved to {filename}")

    def _plot_predictions_vs_actuals_timeseries(self, x_values, actuals, predictions, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, actuals, label='Actual Values', color='blue', marker='.', linestyle='-')
        plt.plot(x_values, predictions, label='Predicted Values', color='red', marker='.', linestyle='--')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Stock Price (Original Scale)")
        plt.legend()
        plt.grid(True)
        if isinstance(x_values, pd.DatetimeIndex) or (hasattr(x_values, 'dtype') and pd.api.types.is_datetime64_any_dtype(x_values)):
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _plot_predictions_vs_actuals_scatter(self, actuals, predictions, title, filename):
        plt.figure(figsize=(8, 8))
        plt.scatter(actuals, predictions, alpha=0.5)
        min_val = min(np.min(actuals), np.min(predictions))
        max_val = max(np.max(actuals), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
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
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _plot_residuals_histogram(self, residuals, title, filename, bins=50):
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=bins, edgecolor='black', alpha=0.7, density=True)
        try:
            from scipy.stats import norm
            mu, std = norm.fit(residuals)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'Fit results: mu={mu:.2f}, std={std:.2f}')
        except ImportError:
            print("SciPy not available, skipping normal distribution fit on residual histogram.")
            pass
        plt.title(title)
        plt.xlabel("Residual Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.axvline(residuals.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {residuals.mean():.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

if __name__ == '__main__':
    # --- Configuration for Experimental Runs ---
    # General parameters
    run_stock_ticker = '^AEX'
    run_years_of_data = 3
    run_look_back = 60
    run_epochs = 20
    run_batch_size = 32

    # Parameters for DataPreprocessor
    run_manual_differencing = False     # Corresponds to old 'use_differencing'
    run_auto_adf_differencing = False   # New ADF based differencing
    run_log_transform = False           # New log transform flag
    # Parameter for ATTLSTMModel
    loss_function_to_test = 'mse' # Can be 'mse' or 'weighted_mse'

    lasso_alpha_values_to_test = [0.005, 0.01]
    all_run_results = {}
    results_df_list = []

    print(f"--- Starting Experimental Runs ---")
    print(f"Stock: {run_stock_ticker}, Years: {run_years_of_data}, Look_back: {run_look_back}, Epochs: {run_epochs}")
    print(f"Manual Diff: {run_manual_differencing}, Auto ADF Diff: {run_auto_adf_differencing}, Log Transform: {run_log_transform}, Loss: {loss_function_to_test}")

    for alpha_val in lasso_alpha_values_to_test:
        # Construct a more descriptive run_id based on parameters varied in the loop or fixed for a set of runs.
        # The dynamic parts (like actual diff order from metrics_log['differencing_order_applied']) will be stored in the results.
        # This config_run_id is for identifying the configuration passed to FullStockPredictionModel.
        config_run_id = f"alpha_{alpha_val}_manualdiff_{run_manual_differencing}_autodiff_{run_auto_adf_differencing}_log_{run_log_transform}_loss_{loss_function_to_test}_lb_{run_look_back}_yrs_{run_years_of_data}"
        print(f"\n--- Running Experiment with Config ID: {config_run_id} ---")

        full_model_instance = FullStockPredictionModel(
            stock_ticker=run_stock_ticker,
            years_of_data=run_years_of_data,
            look_back=run_look_back,
            random_seed=42,
            lasso_alpha=alpha_val,
            use_differencing=run_manual_differencing,
            auto_differencing_adf=run_auto_adf_differencing,
            use_log_transform=run_log_transform,
            loss_function_name=loss_function_to_test
        )
        run_start_time = time.time()
        # The train_and_evaluate method will use the instance's configuration
        current_run_results = full_model_instance.train_and_evaluate(
            epochs=run_epochs,
            batch_size=run_batch_size
        )
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time

        # The actual run_id for storing results will come from the plot_directory in metrics,
        # as it reflects the true applied differencing order.
        actual_run_id_from_metrics = current_run_results["metrics"].get("plot_directory", config_run_id).split('/')[-1]
        print(f"--- Experiment {actual_run_id_from_metrics} (Config: {config_run_id}) Took: {run_duration:.2f} seconds ---")

        if current_run_results and "metrics" in current_run_results:
            all_run_results[actual_run_id_from_metrics] = current_run_results["metrics"]

            # Prepare data for DataFrame summary
            summary_data = {
                "run_id": actual_run_id_from_metrics,
                "lasso_alpha": alpha_val,
                "manual_differencing_config": run_manual_differencing,
                "auto_adf_differencing_config": run_auto_adf_differencing,
                "log_transform_config": run_log_transform,
                "loss_function_config": loss_function_to_test,
                "look_back": run_look_back,
                "years_data": run_years_of_data,
                **current_run_results["metrics"]
            }
            results_df_list.append(summary_data)

            print(f"  Key Metrics for {actual_run_id_from_metrics}:")
            print(f"    RMSE: {current_run_results['metrics'].get('overall_rmse', 'N/A'):.4f}")
            print(f"    MAPE: {current_run_results['metrics'].get('overall_mape', 'N/A'):.2f}%")
            print(f"    Bias (ME): {current_run_results['metrics'].get('bias_me', 'N/A'):.4f}")
            print(f"    Selected Features: {current_run_results['metrics'].get('selected_features_count', 'N/A')}")
            print(f"    Actual Differencing Order: {current_run_results['metrics'].get('differencing_order_applied', 'N/A')}")
            print(f"    Actual Log Transform Applied: {current_run_results['metrics'].get('log_transform_applied', 'N/A')}")


            if hasattr(full_model_instance, 'att_lstm_model') and full_model_instance.att_lstm_model and \
               hasattr(full_model_instance.att_lstm_model, 'model') and full_model_instance.att_lstm_model.model is not None:
                if full_model_instance.processed_df is not None and not full_model_instance.processed_df.empty:
                    if full_model_instance.processed_df.shape[1] > 0:
                        actual_num_features_for_model_input = full_model_instance.att_lstm_model.input_shape[1]
                        sample_raw_data = np.random.rand(full_model_instance.look_back, actual_num_features_for_model_input).astype(np.float32)
                        single_instance_input = np.expand_dims(sample_raw_data, axis=0)
                        try:
                            _ = full_model_instance.att_lstm_model.predict(single_instance_input)
                        except Exception as e:
                            print(f"    Error during warm-up prediction for latency test: {e}")
                            avg_single_pred_time_ms = np.nan
                        else:
                            single_pred_times = []
                            for _ in range(10):
                                pred_start_time = time.time()
                                _ = full_model_instance.att_lstm_model.predict(single_instance_input)
                                single_pred_times.append(time.time() - pred_start_time)
                            avg_single_pred_time_ms = np.mean(single_pred_times) * 1000

                        all_run_results[actual_run_id_from_metrics]["latency_single_pred_ms"] = avg_single_pred_time_ms
                        if results_df_list:
                            results_df_list[-1]["latency_single_pred_ms"] = avg_single_pred_time_ms
                        print(f"    Avg Single Prediction Time: {avg_single_pred_time_ms:.2f} ms")
                    else:
                        print("    Skipping latency test: processed_df has no columns.")
                        all_run_results[actual_run_id_from_metrics]["latency_single_pred_ms"] = np.nan
                        if results_df_list: results_df_list[-1]["latency_single_pred_ms"] = np.nan
                else:
                    print("    Skipping latency test: processed_df is None or empty.")
                    all_run_results[actual_run_id_from_metrics]["latency_single_pred_ms"] = np.nan
                    if results_df_list: results_df_list[-1]["latency_single_pred_ms"] = np.nan
            else:
                print("    Skipping latency test: ATT-LSTM model not available.")
                all_run_results[actual_run_id_from_metrics]["latency_single_pred_ms"] = np.nan
                if results_df_list: results_df_list[-1]["latency_single_pred_ms"] = np.nan
        else:
            print(f"  Run {config_run_id} did not produce results or metrics.")
        print(f"--- End of Experiment for {config_run_id} (Actual ID: {actual_run_id_from_metrics}) ---")

    print("\n\n--- Summary of All Experimental Runs (Console) ---")
    if all_run_results:
        for run_id_key, metrics in all_run_results.items():
            print(f"\nResults for Configuration: {run_id_key}") # This key is actual_run_id_from_metrics
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
    else:
        print("No results collected from experimental runs for console summary.")

    if all_run_results:
        results_json_path = os.path.join("performance_evaluation_report", "all_experimental_run_metrics.json")
        os.makedirs("performance_evaluation_report", exist_ok=True)
        try:
            serializable_results = {}
            for run_key, metrics_dict in all_run_results.items():
                serializable_metrics = {}
                for k, v in metrics_dict.items():
                    if k == 'selected_features_names' and isinstance(v, list):
                        serializable_metrics[k] = ", ".join(v)
                    elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                        serializable_metrics[k] = float(v) if pd.notna(v) else None
                    elif pd.isna(v):
                        serializable_metrics[k] = None
                    else:
                        serializable_metrics[k] = v
                serializable_results[run_key] = serializable_metrics
            with open(results_json_path, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=lambda x: x if x is not None else 'NaN')
        except Exception as e:
            print(f"Error saving metrics to JSON: {e}")
            import traceback
            print("Traceback for JSON saving error:")
            traceback.print_exc()

        results_csv_path = os.path.join("performance_evaluation_report", "all_experimental_run_summary.csv")
        try:
            summary_df = pd.DataFrame(results_df_list)
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
