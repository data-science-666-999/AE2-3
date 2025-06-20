# Performance Evaluation Report

This report summarizes the performance testing of the stock prediction model and its individual modules.

## Test Runs Overview

1.  **Initial Full Model (ATT-LSTM Focus):**
    *   Ticker: `^AEX` (as per `main.py` default at the time of this specific test modification).
    *   ATT-LSTM MSE: 447.8876, MAE: 13.5030, RMSE: 21.1634
    *   Visuals:
        *   `att_lstm_preds_vs_actuals_scatter.png`
        *   `att_lstm_preds_vs_actuals_timeseries.png`
        *   `att_lstm_residuals_histogram.png`
        *   `att_lstm_residuals_timeseries.png`
    *(Note: These were from an earlier run before standardizing on `^AEX` for the final full model test and before NSGM/Ensemble were enabled. The primary purpose was to confirm plotting and basic execution.)*

2.  **DataPreprocessor Module Test:**
    *   Ticker: `AAPL` (changed from `AEL` due to download issues).
    *   Years of data: 1.
    *   LASSO selected 14 out of 32 features for `AAPL` data.
    *   Output: Scaled DataFrame (250 rows, 15 columns for `AAPL`).
    *   Visual: `lasso_feature_coefficients.png`

3.  **ATTLSTMModel Module Test:**
    *   Ticker: `AAPL`, 2 years of data used.
    *   Model: 50 LSTM units, 25 dense units, trained for max 10 epochs.
    *   Performance (Scaled Test Data for `AAPL`):
        *   MSE: 0.004182
        *   MAE: 0.049657
        *   RMSE: 0.064668
    *   Visuals:
        *   `att_lstm_module_loss_curve.png`
        *   `att_lstm_module_preds_vs_actuals.png`

4.  **NSGM1NModel Module Test:**
    *   Ticker: `AAPL`, 2 years of data used.
    *   Performance (Scaled Test Data for `AAPL`):
        *   MSE: 0.495382
        *   MAE: 0.692221
        *   RMSE: 0.703834
    *   Visual: `nsgm_module_preds_vs_actuals.png`
    *   Observation: NSGM performance was notably poor in this isolated test with scaled data.

5.  **EnsembleModel Module Test:**
    *   Method: Tested with `mse_optimization` and `fixed` (0.5, 0.5) weights using simulated data.
    *   Optimized Weights (simulated data): `[0.5736, 0.4264]` (approx.) for ATT-LSTM and NSGM respectively.
    *   Visual: `ensemble_module_optimized_weights.png`

6.  **Full Model Re-evaluation (All Modules Active):**
    *   Ticker: `^AEX`
    *   Years of data: 5
    *   **ATT-LSTM Performance (Original Scale):**
        *   MSE: 300.0773
        *   MAE: 13.6202
        *   RMSE: 17.3227
    *   **NSGM(1,N) Performance (Original Scale):**
        *   MSE: 134157.8997
        *   MAE: 365.0352
        *   RMSE: 366.2757
    *   **Ensemble Model Performance (Original Scale):**
        *   Learned Weights (ATT-LSTM, NSGM): `[1.0, 0.0]`
        *   MSE: 300.0773
        *   MAE: 13.6202
        *   RMSE: 17.3227
    *   **Key Observation:** For the `^AEX` ticker over 5 years, the NSGM model performed very poorly. Consequently, the ensemble model assigned it a weight of 0, making the ensemble's performance identical to that of the ATT-LSTM model alone.
    *   Visuals (prefixed with `full_run_`):
        *   `full_run_att_lstm_preds_vs_actuals_scatter.png`
        *   `full_run_att_lstm_preds_vs_actuals_timeseries.png`
        *   `full_run_att_lstm_residuals_histogram.png`
        *   `full_run_att_lstm_residuals_timeseries.png`
        *   `full_run_nsgm_preds_vs_actuals_timeseries.png`
        *   `full_run_ensemble_preds_vs_actuals_scatter.png`
        *   `full_run_ensemble_preds_vs_actuals_timeseries.png`
        *   `full_run_ensemble_residuals_histogram.png`
        *   `full_run_ensemble_residuals_timeseries.png`

## Summary of Visuals

All generated plots are located in this directory (`performance_evaluation_report/`). They include:

*   **Module-specific tests:**
    *   `lasso_feature_coefficients.png`: Importance of features selected by LASSO in `DataPreprocessor`.
    *   `att_lstm_module_loss_curve.png`: Training/validation loss for the ATT-LSTM module test.
    *   `att_lstm_module_preds_vs_actuals.png`: Predictions vs actuals for the ATT-LSTM module test.
    *   `nsgm_module_preds_vs_actuals.png`: Predictions vs actuals for the NSGM module test.
    *   `ensemble_module_optimized_weights.png`: Learned weights for the Ensemble module test.
*   **Initial ATT-LSTM focused run (main.py, modified for plotting):**
    *   `att_lstm_preds_vs_actuals_scatter.png`
    *   `att_lstm_preds_vs_actuals_timeseries.png`
    *   `att_lstm_residuals_histogram.png`
    *   `att_lstm_residuals_timeseries.png`
*   **Full Model run (main.py with all modules enabled for `^AEX`):**
    *   `full_run_att_lstm_preds_vs_actuals_scatter.png`
    *   `full_run_att_lstm_preds_vs_actuals_timeseries.png`
    *   `full_run_att_lstm_residuals_histogram.png`
    *   `full_run_att_lstm_residuals_timeseries.png`
    *   `full_run_nsgm_preds_vs_actuals_timeseries.png`
    *   `full_run_ensemble_preds_vs_actuals_scatter.png`
    *   `full_run_ensemble_preds_vs_actuals_timeseries.png`
    *   `full_run_ensemble_residuals_histogram.png`
    *   `full_run_ensemble_residuals_timeseries.png`

## Conclusion

The ATT-LSTM model demonstrates reasonable performance for the `^AEX` ticker. The NSGM(1,N) model, in its current implementation and tested configuration, did not perform well and was effectively excluded by the ensemble weighting mechanism. Further tuning or re-evaluation of the NSGM model might be necessary if it's intended to be a contributing part of the ensemble. The data preprocessing and ensemble modules function as expected.

---

## Methodology for Achieving MSE <= 6% (Target: RMSE as % of Mean Actuals <= 6%)

The primary goal of the latest development cycle was to rigorously test and improve the model to achieve an RMSE that is less than or equal to 6% of the mean of the actual test values. This involved several key strategies:

1.  **Extended Dataset:**
    *   The model training and evaluation process was configured to use **10 years of historical stock data** (`^AEX` ticker) to provide a more comprehensive basis for learning market dynamics.

2.  **Advanced Hyperparameter Optimization (ATT-LSTM):**
    *   The `ATTLSTMModel` was subjected to an extensive hyperparameter optimization process using `KerasTuner` (specifically, the `Hyperband` algorithm).
    *   The search space was expanded to include:
        *   Number of LSTM layers (1 or 2).
        *   Units in each LSTM layer.
        *   Number of Dense layers after attention (1 or 2).
        *   Units in each Dense layer.
        *   Learning rate.
        *   Dropout rates for LSTM and Dense layers.
        *   Activation functions for Dense layers.
    *   The `tune_att_lstm.py` script was configured to run this search over the 10-year dataset with increased trials and epochs. The best hyperparameters found by this process were then intended to be used for the final model. (For this exercise, placeholder 'best HPs' were used in `main.py`).

3.  **Feature Engineering:**
    *   The `DataPreprocessor` module was enhanced:
        *   **New Technical Indicators:** Added the Average Directional Index (ADX, ADX_Pos, ADX_Neg).
        *   **Time-Based Features:** Incorporated 'DayOfWeek' and 'Month' derived from the data's timestamp.
        *   **LASSO Adjustment:** The `alpha` for LASSO feature selection was reduced from `0.01` to `0.005` to potentially retain more relevant features.

4.  **Model Simplification (Focus on ATT-LSTM):**
    *   Given the historically poor performance of the `NSGM1NModel` and its zero weighting in the ensemble, a decision was made to remove both the NSGM and the `EnsembleModel` from the main pipeline.
    *   This simplifies the system to a standalone `ATTLSTMModel`, allowing focused efforts on its optimization. All evaluation metrics and efforts are now centered on this single model.

5.  **Target Metric Definition:**
    *   The primary success metric was clarified as: `(RMSE / mean(original_y_test_seq)) * 100 <= 6%`. This "RMSE as % of Mean Actuals" is calculated and reported.

6.  **Production Readiness (Simulated):**
    *   Basic tests for prediction latency (single instance and batch) were added to `main.py` to provide an initial estimate of the trained model's inference speed.

**Expected Outcome (To be filled after actual run):**

*   **Final ATT-LSTM RMSE as % of Mean Actuals:** [Insert Value Here]%
*   **Achieved Target (<= 6%):** Yes/No
*   **Key Hyperparameters Used:** [List key HPs from the tuning process or the placeholder HPs used]
*   **Prediction Latency:**
    *   Single Instance: [Value] ms
    *   Batch (e.g., 32 instances): [Value] ms total, [Value] ms per instance.

All generated plots related to this final model run (e.g., `full_run_att_lstm_preds_vs_actuals_timeseries.png`, `full_run_att_lstm_residuals_histogram.png`, etc.) would reflect the performance of this optimized, standalone ATT-LSTM model trained on 10 years of data with the new features.
