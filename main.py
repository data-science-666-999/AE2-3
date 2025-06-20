import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import modules
from data_preprocessing_module import DataPreprocessor
from att_lstm_module import ATTLSTMModel
from nsgm1n_module import NSGM1NModel
from ensemble_module import EnsembleModel
import matplotlib.pyplot as plt
import os

# --- Module 5: Integrate and Test the Full Model ---

class FullStockPredictionModel:
    def __init__(self, stock_ticker="AEL", years_of_data=5, look_back=60,
                 lstm_units=64, dense_units=32, lstm_learning_rate=0.001,
                 ensemble_optimization_method="mse_optimization", random_seed=42):
        self.stock_ticker = stock_ticker
        self.years_of_data = years_of_data # New parameter
        self.look_back = look_back
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.lstm_learning_rate = lstm_learning_rate
        self.ensemble_optimization_method = ensemble_optimization_method
        self.random_seed = random_seed

        # Updated DataPreprocessor instantiation
        self.data_preprocessor = DataPreprocessor(
            stock_ticker=self.stock_ticker,
            years_of_data=self.years_of_data,
            random_seed=self.random_seed
        )
        self.att_lstm_model = None # Initialized after data preprocessing to get input_shape
        self.nsgm_model = NSGM1NModel()
        self.ensemble_model = EnsembleModel(optimization_method=self.ensemble_optimization_method, random_seed=self.random_seed)
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
        input_shape_lstm = (X_train_seq.shape[1], X_train_seq.shape[2]) # (timesteps, features)
        self.att_lstm_model = ATTLSTMModel(
            input_shape=input_shape_lstm, 
            lstm_units=self.lstm_units, 
            dense_units=self.dense_units, 
            learning_rate=self.lstm_learning_rate, 
            look_back=self.look_back,
            random_seed=self.random_seed
        )
        self.att_lstm_model.build_model()
        self.att_lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs, batch_size=batch_size)

        # Predict on test set for LSTM
        att_lstm_test_preds = self.att_lstm_model.predict(X_test_seq).flatten()

        # --- NSGM and Ensemble sections are temporarily bypassed for ATT-LSTM focus ---
        # # 3. Train Cyclic Multidimensional Gray Model (NSGM(1,N)) Module
        # nsgm_train_data_end_idx = len(self.processed_df) - len(y_seq) + len(y_train_seq)
        # nsgm_train_df = self.processed_df.iloc[:nsgm_train_data_end_idx]
        # nsgm_X_train = nsgm_train_df.drop(columns=[target_column_name]).values
        # nsgm_y_train = nsgm_train_df[target_column_name].values
        # self.nsgm_model.train(nsgm_X_train, nsgm_y_train)
        #
        # # Predict on validation and test sets for NSGM
        # att_lstm_val_preds = self.att_lstm_model.predict(X_val_seq).flatten() # Needed for ensemble training
        # nsgm_val_preds = []
        # for i in range(len(X_val_seq)):
        #     start_idx_in_processed_df = len(self.processed_df) - len(y_seq) + len(y_train_seq) + i - self.look_back
        #     end_idx_in_processed_df = start_idx_in_processed_df + self.look_back
        #     current_nsgm_sequence = self.processed_df.iloc[start_idx_in_processed_df:end_idx_in_processed_df].values
        #     nsgm_val_preds.append(self.nsgm_model.predict(current_nsgm_sequence))
        # nsgm_val_preds = np.array(nsgm_val_preds).flatten()
        #
        # nsgm_test_preds = []
        # for i in range(len(X_test_seq)):
        #     start_idx_in_processed_df = len(self.processed_df) - len(y_seq) + len(y_train_seq) + len(y_val_seq) + i - self.look_back
        #     end_idx_in_processed_df = start_idx_in_processed_df + self.look_back
        #     current_nsgm_sequence = self.processed_df.iloc[start_idx_in_processed_df:end_idx_in_processed_df].values
        #     nsgm_test_preds.append(self.nsgm_model.predict(current_nsgm_sequence))
        # nsgm_test_preds = np.array(nsgm_test_preds).flatten()
        #
        # # 4. Train Ensemble (Weighted Fusion) Module
        # self.ensemble_model.train_weights(att_lstm_val_preds, nsgm_val_preds, y_val_seq)
        #
        # # 5. Make Final Ensemble Predictions on Test Set
        # ensemble_test_preds = self.ensemble_model.predict(att_lstm_test_preds, nsgm_test_preds)
        # --- End of bypassed NSGM and Ensemble sections ---

        # Inverse transform predictions and actual values to original scale
        dummy_preds_lstm = np.zeros((len(att_lstm_test_preds), self.processed_df.shape[1]))
        dummy_preds_lstm[:, self.processed_df.columns.get_loc(target_column_name)] = att_lstm_test_preds
        original_att_lstm_test_preds = self.data_scaler.inverse_transform(dummy_preds_lstm)[:, self.processed_df.columns.get_loc(target_column_name)]

        # Placeholder for NSGM and Ensemble if they were active
        original_nsgm_test_preds = np.full_like(original_att_lstm_test_preds, np.nan)
        original_ensemble_test_preds = original_att_lstm_test_preds # Default to LSTM if ensemble is off

        dummy_actuals = np.zeros((len(y_test_seq), self.processed_df.shape[1]))
        dummy_actuals[:, self.processed_df.columns.get_loc(target_column_name)] = y_test_seq
        original_y_test_seq = self.data_scaler.inverse_transform(dummy_actuals)[:, self.processed_df.columns.get_loc(target_column_name)]

        # Evaluate performance
        print("\n--- Model Performance on Test Set (Original Scale) ---")
        
        mse_lstm = mean_squared_error(original_y_test_seq, original_att_lstm_test_preds)
        mae_lstm = mean_absolute_error(original_y_test_seq, original_att_lstm_test_preds)
        rmse_lstm = np.sqrt(mse_lstm)
        print(f"ATT-LSTM - MSE: {mse_lstm:.4f}, MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}")

        # Bypassed NSGM and Ensemble metrics
        # mse_nsgm = mean_squared_error(original_y_test_seq, original_nsgm_test_preds)
        # mae_nsgm = mean_absolute_error(original_y_test_seq, original_nsgm_test_preds)
        # rmse_nsgm = np.sqrt(mse_nsgm)
        # print(f"NSGM(1,N) - MSE: {mse_nsgm:.4f}, MAE: {mae_nsgm:.4f}, RMSE: {rmse_nsgm:.4f}")
        #
        # mse_ensemble = mean_squared_error(original_y_test_seq, original_ensemble_test_preds)
        # mae_ensemble = mean_absolute_error(original_y_test_seq, original_ensemble_test_preds)
        # rmse_ensemble = np.sqrt(mse_ensemble)
        # print(f"Ensemble Model - MSE: {mse_ensemble:.4f}, MAE: {mae_ensemble:.4f}, RMSE: {rmse_ensemble:.4f}")
        print("NSGM and Ensemble models are currently bypassed for ATT-LSTM focus.")


        # Create a directory for plots if it doesn't exist
        plots_dir = "prediction_plots"
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Ensured '{plots_dir}' directory exists at: {os.path.abspath(plots_dir)}")


        # Generate and save plots (primarily for ATT-LSTM now)
        test_indices = self.processed_df.index[-len(original_y_test_seq):]

        try: # Simplified plotting section
            self._plot_predictions_vs_actuals_timeseries(
                test_indices, original_y_test_seq, original_att_lstm_test_preds,
                "ATT-LSTM Model Predictions vs Actuals",
                os.path.join(plots_dir, "att_lstm_preds_vs_actuals_timeseries.png")
            )
            self._plot_predictions_vs_actuals_scatter(
                original_y_test_seq, original_att_lstm_test_preds,
                "ATT-LSTM Model Predictions vs Actuals",
                os.path.join(plots_dir, "att_lstm_preds_vs_actuals_scatter.png")
            )
            residuals_lstm = original_y_test_seq - original_att_lstm_test_preds
            self._plot_residuals_timeseries(
                test_indices, residuals_lstm,
                "ATT-LSTM Model Residuals Over Time",
                os.path.join(plots_dir, "att_lstm_residuals_timeseries.png")
            )
            self._plot_residuals_histogram(
                residuals_lstm,
                "ATT-LSTM Model Distribution of Residuals",
                os.path.join(plots_dir, "att_lstm_residuals_histogram.png")
            )
        except Exception as e:
            print(f"Error during plotting: {e}")

        print(f"\nVisualizations saved to '{plots_dir}' directory.")
        print("\n--- ATT-LSTM Focused Training and Evaluation Complete ---")

        return {
            "att_lstm_preds": original_att_lstm_test_preds,
            # "nsgm_preds": original_nsgm_test_preds, # Bypassed
            # "ensemble_preds": original_ensemble_test_preds, # Bypassed
            "actual_values": original_y_test_seq,
            "metrics": {
                "lstm_mse": mse_lstm, "lstm_mae": mae_lstm, "lstm_rmse": rmse_lstm,
                # "nsgm_mse": mse_nsgm, "nsgm_mae": mae_nsgm, "nsgm_rmse": rmse_nsgm, # Bypassed
                # "ensemble_mse": mse_ensemble, "ensemble_mae": mae_ensemble, "ensemble_rmse": rmse_ensemble # Bypassed
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
        stock_ticker='^AEX',     # Using AEX index
        years_of_data=5,         # As requested
        look_back=60,            # Using a common look_back period
        lstm_units=100,          # Increased LSTM units
        dense_units=50,          # Increased Dense units
        lstm_learning_rate=0.001,# Default learning rate
        ensemble_optimization_method='mse_optimization',
        random_seed=42
    )

    # Train with more epochs, relying on EarlyStopping in ATTLSTMModel
    # The ATTLSTMModel.train() method now defaults to 100 epochs and has early stopping.
    # We can override epochs here if needed, or pass specific patience values for callbacks.
    # For now, let's use the new defaults in ATTLSTMModel by not overriding `epochs` here,
    # or explicitly set a higher number.
    results = full_model.train_and_evaluate(
        epochs=100, # Max epochs, EarlyStopping will likely trigger sooner.
        batch_size=32
    )

    if results: # Check if results were returned (not empty on error)
        print("\nFinal Ensemble Predictions (first 5):", results["ensemble_preds"][:5])
        print("Actual Values (first 5):", results["actual_values"][:5])
        print("\nMetrics from the run:")
        for model_name, metrics in results["metrics"].items():
            if isinstance(metrics, float): # Handles individual metric entries like lstm_mse etc.
                 print(f"  {model_name}: {metrics:.4f}")
            else: # Should not happen with current structure but good for robustness
                 print(f"  {model_name}: {metrics}")

    else:
        print("Model training and evaluation did not complete successfully.")


