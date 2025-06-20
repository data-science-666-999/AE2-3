import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
import matplotlib.pyplot as plt
import os

# Import DataPreprocessor from the sibling module
from data_preprocessing_module import DataPreprocessor


# --- Module 2: Attention-Enhanced LSTM (ATT-LSTM) Module ---

class ATTLSTMModel:
    def __init__(self, input_shape, lstm_units=64, dense_units=32, learning_rate=0.001, look_back=60, random_seed=42):
        self.input_shape = input_shape  # (timesteps, features)
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.look_back = look_back
        self.model = None
        self.random_seed = random_seed
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)
        # self.hp = None # hp object will be passed to build_model

    def _create_sequences(self, data, target_column_index):
        # This method might be better placed in DataPreprocessor or main script
        # if look_back is tuned, as ATTLSTMModel instance might not know the tuned look_back
        # For now, assuming look_back is fixed for a given ATTLSTMModel instance.
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back), :])
            y.append(data[i + self.look_back, target_column_index])
        return np.array(X), np.array(y)

    def build_model(self, hp=None): # Accept hp object
        if hp: # Use hyperparameters from tuner if provided
            lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
            dense_units = hp.Int('dense_units', min_value=16, max_value=128, step=16)
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
            dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        else: # Use instance attributes if hp not provided (for direct instantiation and training)
            lstm_units = self.lstm_units
            dense_units = self.dense_units
            learning_rate = self.learning_rate
            dropout_rate = 0.2 # Default if not tuning

        inputs = Input(shape=self.input_shape)

        # LSTM layer to process sequences
        # return_sequences=True is crucial for attention mechanism to attend over the sequence
        lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)

        # --- Attention Mechanism (using keras.ops for compatibility with KerasTensors) ---
        # Query: last hidden state of LSTM
        query = lstm_out[:, -1, :]
        # Value and Key: all hidden states of LSTM
        value = lstm_out
        key = lstm_out

        # Calculate attention scores (dot product between query and each key in the sequence)
        # query_reshaped (batch, 1, units) * key (batch, timesteps, units) -> need to transpose key
        query_reshaped = keras.ops.expand_dims(query, axis=1) # Shape: (batch_size, 1, units)

        # Transpose key for matrix multiplication: (batch_size, units, timesteps)
        key_transposed = keras.ops.transpose(key, axes=(0, 2, 1))

        # keras.ops.matmul handles batch dimensions correctly
        scores = keras.ops.matmul(query_reshaped, key_transposed) # Shape: (batch_size, 1, timesteps)
        scores = keras.ops.squeeze(scores, axis=1) # Shape: (batch_size, timesteps)

        # Apply softmax to get attention weights
        attention_weights = keras.ops.softmax(scores, axis=-1) # Shape: (batch_size, timesteps)

        # Reshape attention_weights to (batch_size, timesteps, 1) for element-wise multiplication
        attention_weights_reshaped = keras.ops.expand_dims(attention_weights, axis=-1) # Shape: (batch_size, timesteps, 1)

        # Calculate context vector: weighted sum of values
        # value (batch_size, timesteps, units) * attention_weights_reshaped (batch_size, timesteps, 1)
        context_vector = keras.ops.multiply(value, attention_weights_reshaped) # Shape: (batch_size, timesteps, units)
        context_vector = keras.ops.sum(context_vector, axis=1) # Sum over timesteps -> (batch_size, units)

        # Concatenate the last LSTM output (query) with the context vector
        # Both query and context_vector are now 2D tensors (batch_size, units)
        merged_output = Concatenate()([query, context_vector])

        # Dense layers for prediction
        # Use dense_units from hp or self, and dropout_rate from hp or default
        x = Dense(dense_units, activation="relu")(merged_output)
        x = Dropout(dropout_rate)(x) # Use tuned dropout rate or default
        outputs = Dense(1)(x) # Output a single value for stock price prediction

        # Create the model object
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model with the potentially tuned learning rate
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

        # If build_model is called directly (not by KerasTuner),
        # it should still assign the created model to self.model
        if not hp:
            self.model = model

        return model # Crucial: return the compiled model for KerasTuner

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, early_stopping_patience=10, reduce_lr_patience=5, reduce_lr_factor=0.2):
        # If self.model is not set (e.g. if only build_model(hp) was called by tuner),
        # this method might not work as expected unless it receives a model or self.model is set by caller after tuning.
        # For direct training of an ATTLSTMModel instance, ensure build_model() (no hp) is called if model is None.
        if self.model is None:
            print("Warning: self.model is None in train(). Calling build_model() without hp to initialize.")
            self.build_model() # This will use instance attributes and assign to self.model

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            verbose=1,
            restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            verbose=1,
            min_lr=1e-6 # Do not reduce LR below this value
        )

        print("Training ATT-LSTM model with Early Stopping and ReduceLROnPlateau...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
            shuffle=False # Important for time series data
        )
        print("ATT-LSTM model training complete.")
        return history

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not built or trained. Call build_model() and train() first.")
        return self.model.predict(X_test)

# Example Usage (for testing the module)
if __name__ == '__main__':
    print("\n--- ATTLSTMModel Module Test ---")
    # 1. Get preprocessed data
    # Using AAPL for 2 years to get a bit more data for LSTM training than 1 year.
    # look_back is 60, so we need at least 60+ data points for one sequence.
    # train/val/test split will further reduce this.
    # (0.8 * 0.75) * (N - 60) for X_train. If N=250 (1 year), (250-60)*0.6 = 114 samples for training.
    # If N=500 (2 years), (500-60)*0.6 = 264 samples for training. This is better.
    data_preprocessor = DataPreprocessor(stock_ticker='AAPL', years_of_data=2, random_seed=42)
    processed_df, data_scaler = data_preprocessor.preprocess()

    if processed_df is None or processed_df.empty:
        print("Failed to preprocess data. Aborting ATTLSTMModel test.")
    else:
        print(f"Shape of preprocessed_df: {processed_df.shape}")
        # Assuming target column is the last one, as per DataPreprocessor output
        target_column_name = processed_df.columns[-1]
        target_column_index = processed_df.columns.get_loc(target_column_name)
        print(f"Target column for LSTM: {target_column_name} at index {target_column_index}")

        look_back = 60 # Standard look_back

        # Create sequences (using the model's own _create_sequences if it's suitable, or a local one)
        # The model's _create_sequences is simple, let's use a local version for clarity here.
        def create_sequences_for_test(data_df, target_idx, lb):
            X_data, y_data = [], []
            raw_data = data_df.values # Convert to numpy array for faster iloc-like access
            for i in range(len(raw_data) - lb):
                X_data.append(raw_data[i:(i + lb), :]) # All features
                y_data.append(raw_data[i + lb, target_idx]) # Target feature
            return np.array(X_data), np.array(y_data)

        X, y = create_sequences_for_test(processed_df, target_column_index, look_back)

        if X.shape[0] == 0:
            print("Not enough data to create sequences. Aborting.")
        else:
            # Split data: 80% train, 20% temp -> then 20% of temp becomes 25% of train for val
            # Temporal split is important: shuffle=False
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False) # Split temp into 50% val, 50% test

            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
                print("Not enough data after splitting for training/validation/testing. Aborting.")
            else:
                input_shape = (X_train.shape[1], X_train.shape[2]) # (timesteps, features)

                # Initialize and build model
                # Using default units from the class for this test
                att_lstm_test_model = ATTLSTMModel(
                    input_shape=input_shape,
                    lstm_units=50, # Smaller units for faster test
                    dense_units=25,
                    look_back=look_back,
                    random_seed=42
                )
                att_lstm_test_model.build_model() # Build with default params
                print("\nATTLSTM Model Summary (for test):")
                att_lstm_test_model.model.summary()

                # Train model
                print("\nTraining ATTLSTM model for test...")
                # Train for fewer epochs for a module test, relying on EarlyStopping
                history = att_lstm_test_model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=10, # Reduced epochs for test
                    batch_size=16, # Smaller batch size for smaller dataset
                    early_stopping_patience=3,
                    reduce_lr_patience=2
                )

                # Evaluate model
                print("\nEvaluating ATTLSTM model on test set...")
                predictions_scaled = att_lstm_test_model.predict(X_test).flatten()

                # Since y_test is already scaled (it came from processed_df),
                # we can calculate metrics directly on scaled data for this module test.
                from sklearn.metrics import mean_squared_error, mean_absolute_error

                mse_scaled = mean_squared_error(y_test, predictions_scaled)
                mae_scaled = mean_absolute_error(y_test, predictions_scaled)
                rmse_scaled = np.sqrt(mse_scaled)

                print("\n--- ATTLSTM Model Performance (Scaled Data) ---")
                print(f"Test MSE (scaled): {mse_scaled:.6f}")
                print(f"Test MAE (scaled): {mae_scaled:.6f}")
                print(f"Test RMSE (scaled): {rmse_scaled:.6f}")

                # Visualizations
                print("\nGenerating visualizations for ATTLSTM module test...")

                # 1. Plot training & validation loss
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('ATTLSTM Model Training and Validation Loss (Module Test)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True)
                loss_plot_filename = "att_lstm_module_loss_curve.png"
                plt.savefig(loss_plot_filename)
                print(f"Loss curve plot saved as {loss_plot_filename} in {os.path.abspath('.')}")
                plt.close()

                # 2. Plot predictions vs. actuals for a sample from the test set
                sample_size = min(50, len(y_test)) # Plot up to 50 points
                plt.figure(figsize=(12, 6))
                plt.plot(y_test[:sample_size], label='Actual Values (Scaled)', marker='.')
                plt.plot(predictions_scaled[:sample_size], label='Predicted Values (Scaled)', linestyle='--')
                plt.title(f'ATTLSTM Predictions vs Actuals (First {sample_size} Test Samples - Scaled)')
                plt.xlabel('Sample Index')
                plt.ylabel('Value (Scaled)')
                plt.legend()
                plt.grid(True)
                preds_plot_filename = "att_lstm_module_preds_vs_actuals.png"
                plt.savefig(preds_plot_filename)
                print(f"Predictions vs actuals plot saved as {preds_plot_filename} in {os.path.abspath('.')}")
                plt.close()

    print("\n--- End of ATTLSTMModel Module Test ---")


