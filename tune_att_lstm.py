import keras_tuner as kt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from data_preprocessing_module import DataPreprocessor
from att_lstm_module import ATTLSTMModel

def build_hypermodel(hp):
    """
    Builds a hypermodel for KerasTuner.
    This function is called by the tuner to create a model with a given set of hyperparameters.
    """
    # --- Data Preprocessing ---
    # For tuning, we might fix look_back or make it a hyperparameter.
    # If look_back is a hyperparameter, data preprocessing needs to be efficient per trial.
    # For this first pass, let's assume a fixed look_back for data generation.
    # The input_shape required by ATTLSTMModel will be determined by the data preprocessed
    # in main_tuner() using a fixed look_back.

    # The `hp` object is passed to ATTLSTMModel's build_model method, which defines
    # its own HPs (lstm_units, dense_units, learning_rate, dropout_rate).
    # We do not need to define them again here in this wrapper, nor pass them to constructor.

    # The `input_shape` is now passed directly to this function when the tuner calls it.
    # No, that's not how kt.HyperModel works. The build_hypermodel should define how input_shape is derived or fixed.
    # The `current_input_shape` is passed to `build_hypermodel_with_shape` which then passes it to ATTLSTMModel.
    # The `fixed_look_back` used here is for the ATTLSTMModel's own `look_back` attribute, which is mainly
    # for its internal _create_sequences method (not used if data is prepared externally).
    # The critical `input_shape` for the model layers comes from `current_input_shape`.

    # The `hp.Choice('look_back', ...)` was removed as it's not effectively used without
    # regenerating data sequences per trial based on the chosen look_back.
    # `input_shape` is now determined by `fixed_look_back_data_prep` in `main_tuner`.

    # This function is not directly called by the tuner if using the closure `build_hypermodel_with_shape`.
    # The actual hyperparameter definitions are in ATTLSTMModel.build_model(hp).
    # This function's body can be empty or raise an error if accidentally called.
    # For clarity, let's ensure it's not causing confusion. It's effectively replaced by the closure.
    raise NotImplementedError("This build_hypermodel(hp) should not be called directly by tuner if using the closure; use ATTLSTMModel.build_model(hp) via the closure.")

    # Note: Preprocessing data inside build_hypermodel for each trial can be slow if look_back changes.
    # A more efficient approach for tuning look_back would be to prepare datasets for each look_back
    # value beforehand and select the appropriate one.
    # For now, this is a simplified approach to demonstrate tuning other parameters primarily.
    # If look_back is fixed, data_preprocessor can be initialized once outside the tuner loop.

    # Simplified: We need input_shape. This is tricky if look_back itself is tuned.
    # Let's assume a fixed number of features for now after preprocessing for a default look_back,
    # and the actual look_back for sequence creation is tuned.
    # This means data_preprocessor.preprocess() needs to be called to determine num_features.

    # This function is now a wrapper around ATTLSTMModel.build_model.
    # The actual hyperparameter definitions (lstm_units, dense_units, learning_rate, dropout_rate)
    # are inside ATTLSTMModel.build_model(hp).
    # We need to ensure this function receives input_shape.
    # This function will be replaced by build_hypermodel_with_shape in main_tuner.
    # The original build_hypermodel(hp) is not used directly by RandomSearch anymore.
    raise NotImplementedError("This build_hypermodel(hp) should not be called directly by tuner if using the closure.")


def main_tuner(stock_ticker="^AEX", years_of_data=10, project_name="stock_att_lstm_hyperband_10y"):
    """
    Main function to run KerasTuner with Hyperband.
    """
    print(f"Starting hyperparameter tuning for {stock_ticker} using {years_of_data} years of data.")

    # --- 1. Data Preparation ---
    # We need to decide on how to handle look_back here.
    # If look_back is tuned by hp.Choice in build_hypermodel, then data splitting and sequence creation
    # should ideally happen *after* a look_back value is chosen for a trial.
    # This makes the tuner setup more complex.

    # Simpler approach for first pass: Fix look_back for data generation.
    # The 'look_back' hp.Choice in build_hypermodel will then primarily inform the Input layer shape.
    # This assumes the number of features remains constant regardless of look_back choice.

    # Let's fix data generation look_back for now, and tune other params.
    # The hp.Choice for look_back in build_hypermodel will define the sequence length for that trial.
    # This means `_create_sequences` needs to be called with the tuned look_back for each trial's data.

    # This is still not ideal as it re-processes data for each trial if look_back is tuned.
    # For a truly robust look_back tuning, one would typically create datasets for each look_back
    # and have the tuner select from these datasets or pass look_back to a data generation callback.

    # Let's refine: preprocess data once, then create sequences per trial based on tuned look_back.
    data_preprocessor = DataPreprocessor(stock_ticker=stock_ticker, years_of_data=years_of_data, random_seed=42)
    processed_df, _ = data_preprocessor.preprocess()

    if processed_df.empty:
        print("Error: Preprocessed data is empty. Aborting tuning.")
        return

    # The target column is the last one
    target_column_name = processed_df.columns[-1]

    # Data for tuning will be created dynamically if look_back is tuned.
    # For now, let's assume a fixed look_back for data generation outside the tuner,
    # and the tuner will adjust model architecture.
    # We will use a fixed look_back for data splitting here, and the build_hypermodel
    # will use hp.get('look_back') to set the input_shape.

    # This requires X_train, X_val to be lists of sequences if look_back is tuned by KerasTuner.
    # For simplicity in this first pass, I will NOT tune look_back with KerasTuner directly in build_hypermodel.
    # Instead, I will fix look_back for the data preparation, and KerasTuner will tune model architecture params.
    # The `hp.Choice('look_back', ...)` in build_hypermodel was illustrative and will be removed for now.
    # The model will get its input_shape based on this fixed look_back.

    fixed_look_back_data_prep = 60 # Example fixed look_back for data preparation.
                               # This could be iterated upon manually or in a higher-level loop.

    print(f"Using fixed look_back = {fixed_look_back_data_prep} for data preparation during tuning.")

    # Create sequences function (adapted from FullStockPredictionModel)
    def create_sequences_for_tuning(data, look_back, target_col_name):
        target_col_idx = data.columns.get_loc(target_col_name)
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data.iloc[i:(i + look_back), :].values)
            y.append(data.iloc[i + look_back, target_col_idx])
        return np.array(X), np.array(y)

    X_seq, y_seq = create_sequences_for_tuning(processed_df, fixed_look_back_data_prep, target_column_name)

    if len(X_seq) == 0:
        print("Error: No sequences created. Check data length and look_back period. Aborting tuning.")
        return

    # Split data: 70% train, 15% validation (for tuner), 15% test (final holdout, not used by tuner)
    # Temporal split is important.
    train_size = int(len(X_seq) * 0.70)
    val_size = int(len(X_seq) * 0.15)

    X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(
        X_seq, y_seq, train_size=train_size, shuffle=False
    )
    X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(
        X_temp_seq, y_temp_seq, train_size=val_size, shuffle=False # val_size from remaining
    )

    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        print("Error: Not enough data for train/validation split after sequencing. Aborting tuning.")
        return

    print(f"X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
    print(f"X_val_seq shape: {X_val_seq.shape}, y_val_seq shape: {y_val_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")

    # Update build_hypermodel to remove look_back tuning and use fixed_look_back_data_prep for input_shape
    # This is a workaround for the complexity of tuning look_back directly with KerasTuner's default flow.
    # The ATTLSTMModel's internal look_back is less critical if input_shape is correctly set.

    # We need to pass the actual input_shape to build_hypermodel, or it needs to derive it.
    # Let's make build_hypermodel accept input_shape.

    current_input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    def build_hypermodel_with_shape(hp):
        # This inner function captures current_input_shape
        model_instance = ATTLSTMModel(
            input_shape=current_input_shape,
            look_back=fixed_look_back_data_prep # For consistency if model uses it internally
        )
        # ATTLSTMModel.build_model uses hp to get units, lr, dropout
        return model_instance.build_model(hp)


    # --- 2. KerasTuner Setup ---
    # Using Hyperband tuner
    # max_epochs is the max epochs a single model can be trained for.
    # factor is the reduction factor for the number of models and epochs per bracket.
    # hyperband_iterations controls how many times the Hyperband algorithm is run.
    # More iterations can lead to better results but take longer.
    tuner = kt.Hyperband(
        hypermodel=build_hypermodel_with_shape,
        objective='val_loss',
        max_epochs=81, # Max epochs for the best models (e.g., factor=3, iterations=1 -> 3^0*X, 3^1*X, 3^2*X, 3^3*X, 3^4*X. If X=1, then 1,3,9,27,81)
                       # A common setup: max_epochs=81, factor=3. This implies configurations are trained for 1, 3, 9, 27, 81 epochs.
                       # Or, if max_epochs is total for one config, then for Hyperband it's max_epochs for the full training of one version.
                       # KerasTuner Hyperband: max_epochs is the number of epochs to train a model for in the last bracket.
        factor=3,
        hyperband_iterations=2, # Run the Hyperband algorithm twice.
        directory='keras_tuner_dir',
        project_name=project_name, # Updated project name
        overwrite=True
    )

    tuner.search_space_summary()

    # --- 3. Run Search ---
    print("Starting KerasTuner Hyperband search...")
    # Early stopping for each trial within Hyperband
    # Patience should be appropriate for the number of epochs in each Hyperband round.
    # Smallest number of epochs could be max_epochs / factor^log_factor(max_epochs) approx.
    # For max_epochs=81, factor=3 => 81/3^4 = 1.  Smallest epochs = 1.
    # Next round: 3 epochs, then 9, 27, 81.
    # Patience of 5-10 might be reasonable for rounds with more epochs.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # The `epochs` parameter in search() for Hyperband is the number of epochs to train configurations for in the first bracket.
    # This is a bit confusing. KerasTuner's Hyperband `epochs` in `search` is more like an overall training budget indicator.
    # The actual epochs per trial are managed by Hyperband's successive halving.
    # Let's rely on max_epochs in Hyperband constructor and a high number for search epochs.
    # From KerasTuner docs: "epochs: Number of epochs to train each model.
    # This parameter is overridden by the `max_epochs` argument of the `Hyperband` Tuner."
    # So, the `epochs` in `search` might not be strictly necessary here if `max_epochs` is set in Hyperband.
    # However, it's often provided. Let's set it to `max_epochs`.
    tuner.search(
        X_train_seq, y_train_seq,
        epochs=81, # Corresponds to max_epochs for Hyperband
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stopping_cb],
        batch_size=32 # Batch size can also be a hyperparameter if desired
    )

    # --- 4. Results ---
    print("\nTuning complete.")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest Hyperparameters Found:")
    for param, value in best_hps.values.items():
        print(f"- {param}: {value}")

    # Build the best model with the best hyperparameters
    best_model = tuner.hypermodel.build(best_hps) # or tuner.get_best_models(num_models=1)[0]

    # Optional: Train the best model on combined training and validation data for a few more epochs
    # Or evaluate it directly on the test set
    print("\nEvaluating the best model found by KerasTuner on the test set...")
    loss = best_model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f"Best model test loss (MSE): {loss:.4f}")
    # To get other metrics like MAE, RMSE, you'd predict and calculate manually.

if __name__ == '__main__':
    # Set random seeds for reproducibility as KerasTuner can also have randomness
    np.random.seed(42)
    tf.random.set_seed(42)
    # Add any other library-specific random seed settings if necessary

    main_tuner()
    print("KerasTuner script finished.")

# Note on look_back tuning:
# If 'look_back' were to be tuned by KerasTuner (e.g., hp.Choice in build_hypermodel),
# the data (X_train_seq, y_train_seq, X_val_seq, y_val_seq) would need to be regenerated
# for each trial *before* model.fit is called by the tuner. This typically requires
# subclassing keras_tuner.Tuner and overriding `run_trial`, or using a custom training loop.
# The current setup fixes look_back for data generation and tunes model architecture parameters.
# The `fixed_look_back` in ATTLSTMModel constructor is mostly for its internal sequence creation
# method, which isn't used when data is prepared externally. The `input_shape` passed to it is key.
