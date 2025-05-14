import os
import streamlit as st
import pandas as pd
import torch

# custom pipeline modules (local imports)
from data_gathering import DataGathering
from fred_macro import FredMacroProcessor
from market_macro_compressor import MarketMacroCompressor
from temporal_feature_combiner import TemporalFeatureCombiner
from data_preparator import DataPreparator
from feature_engineering import FeatureEngineering
from features_selection import ReducedFeatureSelector
from loss_history import LossHistory

# Darts TFT imports
import warnings
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse

warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(page_title="Capstone Stock-Macro ML Pipeline", layout="wide")
st.title("🔗 Integrated Stock & Macro ML Capstone Application (TFT Only)")

# Sidebar: settings
st.sidebar.header("Data & Model Configuration")
stock_list = st.sidebar.multiselect(
    "Select stock tickers", ['AAPL', 'MSFT', 'AMZN'], default=['AAPL', 'MSFT']
)
macro_map = {'^GSPC':'sp500', 'BTC-USD':'bitcoin', 'ETH-USD':'ethereum'}
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2014-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
target_stock = st.sidebar.selectbox("Target Stock", stock_list)
seq_len = st.sidebar.number_input("Sequence Length", 1, 200, 60)
epochs = st.sidebar.number_input("Epochs", 1, 100, 10)
batch_size = st.sidebar.number_input("Batch Size", 1, 256, 32)
predict_days = st.sidebar.number_input("Days to Predict", 1, 30, 5)

if st.sidebar.button("🚀 Run Full Pipeline (TFT)"):
    # 1) Gather raw data
    dg = DataGathering(stock_list, macro_map, start_date.isoformat(), end_date.isoformat())
    df_raw = dg.run()
    st.success("✅ Raw data gathered")

    # 2) FRED composites
    fmp = FredMacroProcessor(start_date=start_date.isoformat())
    df_fred = fmp.run_pipeline()
    st.success("✅ FRED macro features generated")

    # 3) Market macro composites
    mmc = MarketMacroCompressor(start=start_date.isoformat(), end=end_date.isoformat())
    df_market = mmc.generate_macro_features()
    st.success("✅ Market macro features generated")

    # 4) Combine temporal features
    tfc = TemporalFeatureCombiner(
        df_raw.join(df_fred).join(df_market).reset_index(drop=False)
    )
    df_combined = tfc.combine()
    st.success("✅ Temporal features combined")

    # 5) Prepare dataset
    preparator = DataPreparator(df_combined, target_stock, stock_list)
    df_ready = preparator.prepare()
    st.success("✅ Data prepared for modeling")

    # 6) Feature selection
    selector = ReducedFeatureSelector(data=df_ready, target_col='target')
    selected_feats, _ = selector.select_features()
    st.write("Selected features:", selected_feats)

    # 7) Darts TFT Training & Forecasting
    # Convert to TimeSeries
    df_tft = df_ready.copy()
    df_tft.index = pd.DatetimeIndex(df_tft.index)
    target_series = TimeSeries.from_series(df_tft['target'])
    covariates = TimeSeries.from_dataframe(df_tft[selected_feats])

    # Scale series
    scaler = Scaler()
    target_transformed = scaler.fit_transform(target_series)
    cov_transformed = covariates

    # Initialize TFTModel
    tft = TFTModel(
        input_chunk_length=seq_len,
        output_chunk_length=predict_days,
        hidden_size=16,
        lstm_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=batch_size,
        n_epochs=epochs,
        likelihood=QuantileRegression([0.1, 0.5, 0.9]),
        random_state=42
    )

    with st.spinner("Training Darts TFT model..."):
        tft.fit(
            series=target_transformed,
            past_covariates=cov_transformed,
            verbose=False,
            val_past_covariates=cov_transformed
        )
    st.success("✅ Darts TFT model trained")

    # Metric evaluation
    val_pred = tft.predict(
        n=predict_days,
        past_covariates=cov_transformed
    )
    val = target_transformed.slice_intersect(val_pred)
    st.write(f"MAPE: {mape(val, val_pred):.2f}%, RMSE: {rmse(val, val_pred):.2f}")

    # Forecast future
    forecast = tft.predict(n=predict_days, past_covariates=cov_transformed)
    forecast = scaler.inverse_transform(forecast)
    df_forecast = forecast.pd_dataframe()['target']
    st.subheader("Forecast")
    st.line_chart(df_forecast)
    st.write(df_forecast)
else:
    st.info("Konfiguriere die Parameter und klicke 'Run Full Pipeline (TFT)' um zu starten.")
