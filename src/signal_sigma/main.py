# %%
# -------------------------------------------------------------------------------------
# 📊 TFT Time Series Forecasting Pipeline Configuration for a Target Stock
# -------------------------------------------------------------------------------------

# ===============================
# 📁 Custom Project Modules
# ===============================
import signal_sigma.config.cfg as cfg

from signal_sigma.data_gathering import DataGathering  # Fetch stock & Yahoo macro data
from signal_sigma.data_preparator import DataPreparator  # Prepare dataset for modeling
from signal_sigma.feature_engineering import (
    FeatureEngineering,
)  # Generate technical indicators
from signal_sigma.fred_macro import (
    FredMacroProcessor,
)  # Download & compress FRED macroeconomic indicators
from signal_sigma.market_macro_compressor import (
    MarketMacroCompressor,
)  # Compress Yahoo macroeconomic signals
from signal_sigma.data_engineering_pipeline import DataEngineeringPipeline  # Main Class
from signal_sigma.loss_history import LossHistory  # Loss function for the model

# ===============================
# 🧪 Core Python Libraries
# ===============================
import os
import sys
import copy
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Timestamp


# ===============================
# ⏳ Time Series Libraries (Darts)
# ===============================
from darts import TimeSeries  # Main time series object
from darts.models import TFTModel  # Temporal Fusion Transformer model
from darts.utils.likelihood_models import (
    QuantileRegression,
)  # For quantile loss functions
from darts.dataprocessing.transformers import Scaler  # For scaling input data
from darts.metrics import (
    mae,
    rmse,
    smape,
    mape,
    r2_score,
    mse,
    rmsle,
)  # Evaluation metrics

# ===============================
# ⚙️ Machine Learning (Auxiliary)
# ===============================
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)  # Early stopping callback
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)  # Sklearn metrics

# ===============================
# 📋 Utility Tools
# ===============================
from tabulate import tabulate  # Nice tabular printing for reporting

# ===============================
# ❗ Clean Up Output
# ===============================
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner logs

# ===============================
# 🧭 Configuration Parameters
# ===============================
target_stock = "NVDA"  # Set the stock you want to analyze
path_stock = os.path.join(cfg.DATA_PATH, "Stock_market_data")
# path_stock = "../../data/Stock_market_data"        # Path to your preprocessed datasets
start_date = "2014-01-01"  # Start of historical data
end_date = "2025-05-09"  # End date of your data
output_len = 15  # Forecast horizon (e.g., predict 15 days ahead)

# Disable actual showing of plots
plt.show = lambda: None

# %%

# # ===============================
# # 📁 Custom Project Modules
# # ===============================
# from signal_sigma.data_gathering import DataGathering                  # Fetch stock & Yahoo macro data
# from signal_sigma.data_preparator import DataPreparator            # Prepare dataset for modeling
# from signal_sigma.feature_engineering import FeatureEngineering           # Generate technical indicators
# from signal_sigma.fred_macro import FredMacroProcessor                    # Download & compress FRED macroeconomic indicators
# from signal_sigma.market_macro_compressor import MarketMacroCompressor      # Compress Yahoo macroeconomic signals
# from signal_sigma.data_engineering_pipeline import DataEngineeringPipeline   # Main Class
# from signal_sigma.data_engineering_pipeline import DataEngineeringPipeline
# #path_stock = "../data/Stock_market_data"
# # Disable actual showing of plots
# plt.show = lambda: None
# ===============================
# 🧭 Configuration Parameters
# ===============================
target_stock = "NVDA"  # Set the stock you want to analyze
path_stock = "../data/Stock_market_data"  # Path to your preprocessed datasets
start_date = "2014-01-01"  # Start of historical data
end_date = "2025-05-09"  # End date of your data

pipeline = DataEngineeringPipeline(
    path_stock="../data/Stock_market_data",
    start_date=start_date,
    end_date=end_date,
    top_n_feature_important=10,
)

features, columns = pipeline.run()

# %% [markdown]
# # worked with the reduced feratured engineering df

# %%
# Load datasets for each stock                         # Forecast horizon (e.g., predict 15 days ahead)

TSLA_df = pd.read_csv(
    f"{path_stock}/TSLA_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
NVDA_df = pd.read_csv(
    f"{path_stock}/NVDA_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
MSFT_df = pd.read_csv(
    f"{path_stock}/MSFT_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
GOOGL_df = pd.read_csv(
    f"{path_stock}/GOOGL_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
AMZN_df = pd.read_csv(
    f"{path_stock}/AMZN_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()
AAPL_df = pd.read_csv(
    f"{path_stock}/AAPL_reduced_dataset_{start_date}_{end_date}.csv",
    parse_dates=["date"],
    index_col="date",
).sort_index()

# Map stock tickers to their respective DataFrames
stock_dataframes = {
    "TSLA": TSLA_df,
    "NVDA": NVDA_df,
    "MSFT": MSFT_df,
    "GOOGL": GOOGL_df,
    "AMZN": AMZN_df,
    "AAPL": AAPL_df,
}

# Select the correct DataFrame based on the target stock
model_df = stock_dataframes.get(target_stock)


# XXX: On Apple Silicon, Darts may require float32 instead of float64
# to avoid issues with PyTorch. This is a workaround for compatibility.
cols = model_df.select_dtypes(include=["float64"]).columns
model_df[cols] = model_df[cols].astype(np.float32)
# # Or alternatively, you can set the device to CPU

model_df_copy = model_df.copy(deep=True)


print("model_df_copy.columns :", model_df_copy.columns)
print("model_df_copy.index :", model_df_copy.index)
# Extract all feature column names, excluding the 'target' column
features_list_model = [col for col in model_df.columns if col != "target"]

# Print out the number of features and their names
print("\n[INFO] Number of features for model:", len(features_list_model))
print("\n[INFO] Feature list for model:", features_list_model)


# %% [markdown]
# # Step 2-4:Reindexing to daily frequency

# %%
# ------------------------------------------------------------------------------
# 🧾 Step 2: Inspect & Standardize Time Index (Reindexing to Daily Frequency)
# ------------------------------------------------------------------------------
full_index = pd.date_range(
    start=model_df.index.min(), end=model_df.index.max(), freq="D"
)
if model_df.index.inferred_freq is None or not model_df.index.is_monotonic_increasing:
    model_df = model_df.reindex(full_index)
    model_df.interpolate(method="linear", limit_direction="both", inplace=True)
print("[INFO] Loaded model_dataset shaped: ", model_df.shape)
print("[INFO] Loaded model_dataset NaN: ", model_df.isnull().sum().sum())
print("[INFO] Loaded model_dataset type: ", type(model_df))
print(
    "[INFO] Loaded model_dataset index:",
    model_df.index.min(),
    "--- to ---",
    model_df.index.max(),
)


# %% [markdown]
# # 🔁 Step 3: Convert to Darts TimeSeries format
#

# %%
# ---------------------------------------------
# 🎯 Step 3-1: Convert Target Column to Darts TimeSeries
# ---------------------------------------------
raw_target_df = model_df[["target"]]
raw_target = TimeSeries.from_dataframe(
    raw_target_df, value_cols="target", freq="D", fill_missing_dates=True
)
raw_covariates_df = model_df[features_list_model]
raw_covariates = TimeSeries.from_dataframe(
    raw_covariates_df, value_cols=features_list_model, freq="D", fill_missing_dates=True
)


# %% [markdown]
# # Step 4: Time-based Split for Train/Val/Test

# %%
# -------------------------------------------------------------------------------------
# ✂️ Step 4: Time-based Split for Train/Val/Test
# -------------------------------------------------------------------------------------
from datetime import timedelta  # Used to offset time ranges for slicing covariates

output_len = output_len  # Number of days we want to predict into the future
input_len = (
    output_len * 5
)  # Number of past days (lookback window) to feed into the model

val_size = min(((input_len + output_len) + 5), int(0.1 * len(raw_target)))
train_target_raw = raw_target[: -val_size - output_len]
val_target_raw = raw_target[-val_size - output_len : -output_len]
test_target_raw = raw_target[-output_len:]

train_covariates_raw = raw_covariates.slice(
    train_target_raw.start_time(),  # Start of training period
    train_target_raw.end_time()
    + timedelta(days=output_len),  # Extend to include future forecast period
)

val_covariates_raw = raw_covariates.slice(
    val_target_raw.start_time(),  # Start of validation period
    val_target_raw.end_time()
    + timedelta(days=output_len),  # Extend to include forecast period
)

test_covariates_raw = raw_covariates.slice(
    test_target_raw.start_time()
    - timedelta(days=input_len),  # Start: back-projected from test start
    test_target_raw.end_time()
    + timedelta(days=output_len),  # End: includes forecast window
)


# %% [markdown]
# # Step 5: Scale Target and Covariates

# %%
# --------------------------------------------------------------
# 🧮 Step 5: Normalize TimeSeries Data Using Darts Scaler
# --------------------------------------------------------------
t_scaler = Scaler()  # Target scaler
f_scaler = Scaler()  # Feature/covariates scaler

# Fit the target scaler on the training target only and transform all splits
train_target = t_scaler.fit_transform(
    train_target_raw
)  # Learn scaling on training target
val_target = t_scaler.transform(val_target_raw)  # Apply same scale to validation target
test_target = t_scaler.transform(test_target_raw)  # Apply same scale to test target
print("\n[INFO] Step 5🧾 target scaler Time Index Ranges after Scaling:")
print(
    f"[INFO] Train Target         : {train_target.time_index.min()} → {train_target.time_index.max()}"
)
print(
    f"[INFO] Validation Target    : {val_target.time_index.min()} → {val_target.time_index.max()}"
)
print(
    f"[INFO] Test Target          : {test_target.time_index.min()} → {test_target.time_index.max()}"
)

train_covariates = f_scaler.fit_transform(
    train_covariates_raw
)  # Learn scaling on training covariates
val_covariates = f_scaler.transform(
    val_covariates_raw
)  # Apply same scale to validation covariates
test_covariates = f_scaler.transform(
    test_covariates_raw
)  # Apply same scale to test covariates

print("\n[INFO] Step 5🧾 covariate scaler Time Index Ranges after Scaling:")
print(
    f"[INFO] Train Covariates    : {train_covariates.time_index.min()} → {train_covariates.time_index.max()}"
)
print(
    f"[INFO] Validation Covariates: {val_covariates.time_index.min()} → {val_covariates.time_index.max()}"
)
print(
    f"[INFO] Test Covariates      : {test_covariates.time_index.min()} → {test_covariates.time_index.max()}"
)


# %% [markdown]
# # Step 6: Initialize and Train TFT Model
#

# %%
# -------------------------------------------------------------------------------------
# 🧠 Step 6: Initialize and Train TFT Model
# -------------------------------------------------------------------------------------

# Initialize the Temporal Fusion Transformer (TFT) model from Darts
from signal_sigma.loss_history import LossHistory

loss_logger = LossHistory()  # Instantiate the callback
model = TFTModel(
    input_chunk_length=input_len,  # 🔁 Number of historical time steps used as input
    output_chunk_length=output_len,  # 🔮 Number of future time steps to predict
    hidden_size=64,  # 💡 Number of hidden units in LSTM layers (feature extractor size)
    lstm_layers=1,  # 🔄 Number of LSTM layers used in the encoder-decoder architecture
    dropout=0.1,  # 🕳 Dropout rate for regularization (to prevent overfitting)
    batch_size=64,  # 📦 Number of samples per training batch
    n_epochs=2,  # 🔁 Number of training epochs
    num_attention_heads=1,  # 🎯 Heads in multi-head attention layer for learning temporal patterns
    force_reset=True,  # 🧽 Force fresh training, resetting previous weights and checkpoints
    save_checkpoints=True,  # 💾 Save intermediate models automatically during training
    # 🎯 Quantile regression for probabilistic forecasting (predicts intervals)
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
    # 🗓️ Add time-based encoders to give model temporal awareness
    add_encoders={
        "cyclic": {
            "past": [
                "month",
                "day",
                "weekday",
            ]  # Cyclical encodings (e.g., sin/cos for months, days)
        },
        "datetime_attribute": {
            "past": ["year", "month", "weekday"]  # Raw datetime attributes
        },
        "position": {
            "past": ["relative"]  # Relative time encoding (position in sequence)
        },
    },
    # ⚙️ Pass advanced settings to PyTorch Lightning trainer
    # ⚙️ Pass advanced settings to PyTorch Lightning trainer
    pl_trainer_kwargs={
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=5),  # ⏸ Early stopping callback
            loss_logger,  # 📊 Our custom loss tracking callback
        ],
        "log_every_n_steps": 10,  # 📝 How often to log training steps
        "enable_model_summary": True,  # 📋 Show model layer summary
        "enable_progress_bar": True,  # 📊 Show progress bar during training
        "logger": True,  # 🧾 Enable default logger (e.g., TensorBoard)
    },
)


# ---------------------------------------------------------------
# 🚀 Fit (Train) the Model
# ---------------------------------------------------------------

model.fit(
    series=train_target,  # 🎯 Target series for training
    future_covariates=train_covariates,  # 🔮 Associated features for training (aligned with future steps)
    val_series=val_target,  # 🎯 Validation target series (to monitor overfitting)
    val_future_covariates=val_covariates,  # 🔮 Validation covariates aligned with val_target
    verbose=True,  # 📣 Print training progress to console
)


# %% [markdown]
# # Step 7: Plot Train vs Validation Loss Over Epochs

# %%
# -------------------------------------------------------------------------------------
# Step 7: Plot Train vs Validation Loss Over Epochs
# -------------------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(loss_logger.epochs, loss_logger.train_losses, label="Train Loss", marker="o")
plt.plot(
    loss_logger.epochs, loss_logger.val_losses, label="Validation Loss", marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.title(F"Train vs Validation Loss Over Epochs -\n Feature:{features_list_model}")
plt.title("Train vs Validation Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# #  Single-Shot Forecast for Singel Point and Quantiles

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# === Step 1: Predict quantile forecasts (probabilistic output with uncertainty) ===
quantil_forecast = model.predict(
    n=output_len,  # Number of time steps to forecast
    series=val_target,  # Past target values to condition the model
    future_covariates=test_covariates,  # Known future inputs (calendar features, macro, etc.)
    num_samples=100,  # Enables Monte Carlo sampling for quantiles
)

# === Step 2: Predict deterministic (point) forecast separately ===
point_forecast = model.predict(
    n=output_len, series=val_target, future_covariates=test_covariates
)

# === Step 3: Inverse transform all forecast outputs back to original scale ===
quantil_forecast_p10 = t_scaler.inverse_transform(
    quantil_forecast.quantile_timeseries(0.1)
)
quantil_forecast_p50 = t_scaler.inverse_transform(
    quantil_forecast.quantile_timeseries(0.5)
)
quantil_forecast_p90 = t_scaler.inverse_transform(
    quantil_forecast.quantile_timeseries(0.9)
)
point_forecast_inv = t_scaler.inverse_transform(point_forecast)

# === Step 4: Align forecasts and ground truth by shared date index ===
common_index = quantil_forecast_p50.time_index.intersection(test_target_raw.time_index)

# Extract aligned values as 1D arrays
true_vals = test_target_raw[common_index].values().squeeze()
p10_vals = quantil_forecast_p10[common_index].values().squeeze()
p50_vals = quantil_forecast_p50[common_index].values().squeeze()
p90_vals = quantil_forecast_p90[common_index].values().squeeze()
point_vals = point_forecast_inv[common_index].values().squeeze()
time_index = common_index


# === Step 5: Define evaluation metrics as helper functions ===
def smape(y_true, y_pred):
    return 200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)


def mape(y_true, y_pred):
    return 100.0 * np.abs((y_true - y_pred) / (y_true + 1e-9))


def pinball_loss(y_true, y_pred, q):
    delta = y_true - y_pred
    return np.maximum(q * delta, (q - 1) * delta)


# === Step 6: Initialize evaluation DataFrame ===
forecast_df = pd.DataFrame(
    {
        "date": time_index,
        "true": true_vals,  # Ground truth
        "p10": p10_vals,  # 10% lower quantile
        "p50": p50_vals,  # Median forecast
        "p90": p90_vals,  # 90% upper quantile
        "point_forecast": point_vals,  # Point (mean or deterministic) prediction
    }
)

# === Step 7: Compute residuals and interval width ===
forecast_df["residual_point"] = forecast_df["true"] - forecast_df["point_forecast"]
forecast_df["residual_p50"] = forecast_df["true"] - forecast_df["p50"]
forecast_df["quantile_width_80"] = forecast_df["p90"] - forecast_df["p10"]
forecast_df["z_residual_point"] = forecast_df["residual_point"] / (
    forecast_df["quantile_width_80"] + 1e-9
)

# === Step 8A: Point forecast metrics (per day) ===
forecast_df["mae_point"] = np.abs(forecast_df["residual_point"])
forecast_df["mape_point"] = mape(forecast_df["true"], forecast_df["point_forecast"])
forecast_df["smape_point"] = smape(forecast_df["true"], forecast_df["point_forecast"])
forecast_df["rmse_point"] = np.sqrt(forecast_df["residual_point"] ** 2)
forecast_df["r2_point"] = r2_score(forecast_df["true"], forecast_df["point_forecast"])

# === Step 8B: Quantile p50 forecast metrics (for comparison) ===
forecast_df["mae_p50"] = np.abs(forecast_df["residual_p50"])
forecast_df["mape_p50"] = mape(forecast_df["true"], forecast_df["p50"])
forecast_df["smape_p50"] = smape(forecast_df["true"], forecast_df["p50"])
forecast_df["rmse_p50"] = np.sqrt(forecast_df["residual_p50"] ** 2)
forecast_df["r2_p50"] = r2_score(forecast_df["true"], forecast_df["p50"])

# === Step 9: Pinball loss per quantile (measures sharpness + calibration) ===
forecast_df["pinball_p10"] = pinball_loss(forecast_df["true"], forecast_df["p10"], 0.1)
forecast_df["pinball_p50"] = pinball_loss(forecast_df["true"], forecast_df["p50"], 0.5)
forecast_df["pinball_p90"] = pinball_loss(forecast_df["true"], forecast_df["p90"], 0.9)

# === Step 10: Interval coverage (binary flag per day) ===
forecast_df["interval_covered_80"] = (
    (forecast_df["true"] >= forecast_df["p10"])
    & (forecast_df["true"] <= forecast_df["p90"])
).astype(int)

# === Step 11: Quantile calibration columns — 1 if true <= quantile prediction ===
quantiles = np.linspace(0.05, 0.95, 19)
empirical_coverage_dict = {
    f"coverage_q{int(q*100)}": (
        forecast_df["true"]
        <= quantil_forecast.quantile_timeseries(q).values().squeeze()
    ).astype(int)
    for q in quantiles
}
# Add each column to the main DataFrame
for col, values in empirical_coverage_dict.items():
    forecast_df[col] = values

# === Step 12: Final formatting and export ===
# Step 1: Make sure forecast_df index is set to date
forecast_df.set_index("date", inplace=True)
forecast_df.to_csv(
    f"{path_stock}/{target_stock}_forecast_eval_metrics_{start_date}_{end_date}.csv"
)

# Step 2: Perform a left join on time index
model_result_df = model_df_copy.join(forecast_df, how="left")

# Step 3: Fill missing predictions (non-forecasted rows) with 0
model_result_df.fillna(0, inplace=True)
# Step 4: Save to CSV
model_result_df.to_csv(
    f"{path_stock}/{target_stock}_Model_Initial_Result_{start_date}_{end_date}.csv"
)

print("forecast_df.columns :", forecast_df.columns)
print("model_result_df.columns :", model_result_df.columns)

# Optionally inspect
model_result_df.tail(5)


# %% [markdown]
# # Step 11: Plot Forecast vs Actual (Quantiles + True)

# %%


# === Step 1: Set start of plot range for history context ===
plot_start = Timestamp("2024-10-01")
raw_target_slice = raw_target.slice(plot_start, raw_target.end_time())
full_target_index = raw_target_slice.time_index
full_target_vals = raw_target_slice.values().squeeze()

# === Step 2: Extract forecast and truth values from forecast_df ===
forecast_index = forecast_df.index
y_true = forecast_df["true"].astype(float).values
y_p10 = forecast_df["p10"].astype(float).values
y_p50 = forecast_df["p50"].astype(float).values
y_p90 = forecast_df["p90"].astype(float).values
y_point = forecast_df["point_forecast"].astype(float).values

# === Step 3: Define split markers ===
# Assuming you have already split your raw_target like this:

train_end = train_target_raw.end_time()
val_end = val_target_raw.end_time()
test_start = test_target_raw.start_time()


train_target_raw = raw_target[:train_end]
val_target_raw = raw_target[train_end:val_end]
test_target_raw = raw_target[val_end:]

train_end = train_target_raw.end_time()
val_end = val_target_raw.end_time()
test_start = forecast_index[0]

# === Step 4: Begin plot ===
plt.figure(figsize=(14, 6))

# Plot full historical target
plt.plot(
    full_target_index,
    full_target_vals,
    label="Historical Target",
    color="black",
    linewidth=2,
)

# Plot true values in test set
plt.plot(forecast_index, y_true, label="True (Test)", color="blue", linewidth=2)

# Plot forecasts
plt.plot(
    forecast_index, y_p50, label="Forecast Median (p50)", color="magenta", linewidth=2
)
plt.plot(
    forecast_index, y_p10, label="Forecast Lower (p10)", linestyle="--", color="skyblue"
)
plt.plot(
    forecast_index, y_p90, label="Forecast Upper (p90)", linestyle="--", color="green"
)
plt.plot(forecast_index, y_point, label="Forecast Point", linestyle=":", color="orange")

# Shaded p10–p90 interval
plt.fill_between(
    forecast_index,
    y_p10,
    y_p90,
    color="lightgray",
    alpha=0.4,
    label="Confidence Interval (p10–p90)",
)

# === Step 5: Add vertical split markers ===
plt.axvline(train_end, color="gray", linestyle="--", linewidth=1.5, label="Train End")
plt.axvline(
    val_end, color="purple", linestyle="--", linewidth=1.5, label="Validation End"
)
plt.axvline(
    test_start, color="red", linestyle="--", linewidth=2, label="Forecast Start"
)

# === Style ===
plt.title(f"TFT Forecast vs. True - Full Signal View ({target_stock})")
plt.xlabel("Date")
plt.ylabel("Target Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
