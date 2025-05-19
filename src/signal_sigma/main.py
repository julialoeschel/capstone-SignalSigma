#!/usr/bin/env python3
"""
Signal Sigma Stock Forecasting CLI

Provides options to load data and train the Temporal Fusion Transformer model.
"""
import argparse
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Timestamp

# Custom modules
import signal_sigma.config.cfg as cfg
from signal_sigma.data_engineering_pipeline import DataEngineeringPipeline

# Darts and training
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from signal_sigma.loss_history import LossHistory

# Suppress warnings and plot auto-show
warnings.filterwarnings("ignore")
plt.show = lambda: None


def load_data(path_stock: str, start_date: str, end_date: str, stock_list: list) -> dict:
    """
    Load reduced datasets for each stock ticker in stock_list.
    Returns a dict of DataFrames keyed by ticker.
    """
    dfs = {}
    for ticker in stock_list:
        file_path = os.path.join(
            path_stock,
            f"{ticker}_reduced_dataset_{start_date}_{end_date}.csv"
        )
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue
        df = pd.read_csv(
            file_path,
            parse_dates=["date"],
            index_col="date"
        ).sort_index()
        dfs[ticker] = df
    return dfs


def train_model(
    path_stock: str,
    target_stock: str,
    start_date: str,
    end_date: str,
    output_len: int = 15,
    top_n_feature_important: int = 10
):
    """
    Run DataEngineeringPipeline and train a Darts TFT model for the given target_stock.
    """
    # 1. Data engineering pipeline
    pipeline = DataEngineeringPipeline(
        path_stock=path_stock,
        start_date=start_date,
        end_date=end_date,
        top_n_feature_important=top_n_feature_important
    )
    features, columns = pipeline.run()
    print(f"[INFO] Top {top_n_feature_important} features: {features}\n")

    # 2. Load the prepared dataset for target_stock
    df = pd.read_csv(
        os.path.join(
            path_stock,
            f"{target_stock}_reduced_dataset_{start_date}_{end_date}.csv"
        ),
        parse_dates=["date"],
        index_col="date"
    ).sort_index()

    # Ensure compatibility on float types
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # TODO: Insert the full training routine here:
    #       - convert df to TimeSeries
    #       - split into train/val/test
    #       - scale with Scaler()
    #       - configure TFTModel and fit
    #       - log and plot losses
    # For now, just confirm the DataFrame
    print(f"[INFO] Loaded {target_stock} data: {df.shape[0]} rows, {df.shape[1]} columns")


def main():
    parser = argparse.ArgumentParser(
        description="Signal Sigma Stock Forecasting CLI"
    )
    parser.add_argument(
        '--load-data',
        action='store_true',
        help='Load and preview datasets for all supported stocks'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run the data pipeline and train the TFT model for a target stock'
    )
    parser.add_argument(
        '--stock',
        type=str,
        default='NVDA',
        help='Target stock ticker to train (default: NVDA)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2014-01-01',
        help='Start date in YYYY-MM-DD format (default: 2014-01-01)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-05-16',
        help='End date in YYYY-MM-DD format (default: 2025-05-16)'
    )
    parser.add_argument(
        '--path-stock',
        type=str,
        default=os.path.join(cfg.DATA_PATH, "Stock_market_data"),
        help='Path to stock data directory'
    )
    parser.add_argument(
        '--output-len',
        type=int,
        default=15,
        help='Forecast horizon length (default: 15)'
    )
    parser.add_argument(
        '--top-features',
        type=int,
        default=10,
        help='Number of top features for pipeline (default: 10)'
    )
    args = parser.parse_args()

    stock_list = ['TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'AAPL']

    if args.load_data:
        dfs = load_data(
            args.path_stock,
            args.start_date,
            args.end_date,
            stock_list
        )
        for ticker, df in dfs.items():
            print(f"{ticker}: {df.shape[0]} rows, {df.shape[1]} cols")

    if args.train:
        train_model(
            path_stock=args.path_stock,
            target_stock=args.stock,
            start_date=args.start_date,
            end_date=args.end_date,
            output_len=args.output_len,
            top_n_feature_important=args.top_features
        )


if __name__ == '__main__':
    main()
