import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras

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

    def _create_sequences(self, data, target_column_index):
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back), :])
            y.append(data[i + self.look_back, target_column_index])
        return np.array(X), np.array(y)

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # LSTM layer to process sequences
        # return_sequences=True is crucial for attention mechanism to attend over the sequence
        lstm_out = LSTM(self.lstm_units, return_sequences=True)(inputs)

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
        x = Dense(self.dense_units, activation="relu")(merged_output)
        outputs = Dense(1)(x) # Output a single value for stock price prediction

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            self.build_model()

        print("Training ATT-LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
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
    # Simulate some preprocessed data (replace with actual preprocessed_df from Module 1)
    # For demonstration, let's assume preprocessed_df has 6 features (including 'Close')
    # and a shape like (n_samples, n_features)
    n_samples = 1000
    n_features = 6 # e.g., High, Low, SMA_30, EMA_10, RSI, Close
    # Simulate scaled data between 0 and 1
    simulated_data = np.random.rand(n_samples, n_features)
    simulated_df = pd.DataFrame(simulated_data, columns=[f'feature_{i}' for i in range(n_features-1)] + ['Close'])

    # Assuming 'Close' is the target column and is the last column (index n_features-1)
    target_column_index = simulated_df.columns.get_loc('Close')

    look_back = 60
    # Create sequences
    X, y = [], []
    for i in range(len(simulated_df) - look_back):
        X.append(simulated_df.iloc[i:(i + look_back), :].values)
        y.append(simulated_df.iloc[i + look_back, target_column_index])
    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=False) # 0.25 of 0.8 is 0.2

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2]) # (timesteps, features)
    att_lstm = ATTLSTMModel(input_shape=input_shape, look_back=look_back)
    att_lstm.build_model()
    att_lstm.model.summary()

    history = att_lstm.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)

    predictions = att_lstm.predict(X_test)
    print("\nSample predictions:", predictions[:5].flatten())
    print("Sample true values:", y_test[:5])


