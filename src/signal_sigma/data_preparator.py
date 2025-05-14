import os
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineering

class DataPreparator:
    """
    Prepares a modeling-ready dataset for stock forecasting with technical and market-wide features.
    Includes relative investment analysis and trend tracking across stocks.
    """

    def __init__(self, data, target_stock, stock_list, time_cutoff="2014-01-01"):
        self.target_stock = target_stock.upper()
        self.stock_list = stock_list
        self.time_cutoff = pd.to_datetime(time_cutoff)

        self.df = None
        self.source_df = None
        self.feature_df = None
        self.merged_df = None

        # Load data
        if isinstance(data, pd.DataFrame):
            print("[INFO] Using provided DataFrame as source data.")
            self.df = data.sort_index().copy(deep=True)
        elif isinstance(data, str) and os.path.isfile(data):
            print(f"[INFO] Loading dataset from file: {data}")
            self.df = pd.read_csv(data, parse_dates=["date"], index_col="date").sort_index()
        else:
            raise ValueError("❌ data must be a DataFrame or valid CSV path.")

        self.source_df = self.df.copy(deep=True)

    def prepare(self):
        target_df = self.prepare_target_columns()
        self.compute_market_investment()
        self.analyze_investment_trend_features()  # 👈 NEW: Analyze relative market dynamics
        self.drop_raw_columns()
        self.generate_features(target_df)
        self.clean_feature_df()
        self.merge_with_aux()
        self.report_summary()
        return self.merged_df.copy(deep=True)

    def prepare_target_columns(self):
        print(f"[INFO] Preparing target stock: {self.target_stock}")
        cols = [f"{col}_{self.target_stock}" for col in ['close', 'open', 'low', 'high', 'volume']]
        return self.df[cols]

    def compute_market_investment(self):
        print("[INFO] Calculating total average daily investment across all tracked stocks...")
        self.source_df['total_ave_daily_invest'] = 0.0
        for stock in self.stock_list:
            avg_price = (
                self.df[f'close_{stock}'] + self.df[f'open_{stock}'] +
                self.df[f'low_{stock}'] + self.df[f'high_{stock}']
            ) / 4
            self.source_df[f'ave_daily_invest_{stock}'] = avg_price * self.df[f'volume_{stock}']
            self.source_df['total_ave_daily_invest'] += self.source_df[f'ave_daily_invest_{stock}']

        for stock in self.stock_list:
            self.source_df[f'ratio_daily_invest_{stock}'] = (
                self.source_df[f'ave_daily_invest_{stock}'] / (self.source_df['total_ave_daily_invest'] + 1e-9)
            )

    def analyze_investment_trend_features(self):
        """
        🔍 Analyzes only the target stock's investment ratio trend:
        - Daily change, percent change
        - Rolling mean and std
        - Z-score (standardized position in distribution)
        """
        print(f"[INFO] Analyzing investment trend features for {self.target_stock} only...")

        col = f'ratio_daily_invest_{self.target_stock}'
        if col in self.source_df.columns:
            # Calculate recent changes in relative investment
            self.source_df[f'{col}_change_7d'] = self.source_df[col].diff(periods=7)
            self.source_df[f'{col}_pct_change_7d'] = self.source_df[col].pct_change(periods=7).clip(-1, 1)

            self.source_df[f'{col}_change_30d'] = self.source_df[col].diff(periods=30)
            self.source_df[f'{col}_pct_change_30d'] = self.source_df[col].pct_change(periods=30).clip(-1, 1)

            # Rolling stats to assess trend and volatility
            self.source_df[f'{col}_rolling_mean_15d'] = self.source_df[col].rolling(window=15).mean()
            self.source_df[f'{col}_rolling_std_7d'] = self.source_df[col].rolling(window=7).std()

            # Z-score of current value vs rolling distribution
            mean = self.source_df[f'{col}_rolling_mean_15d']
            std = self.source_df[f'{col}_rolling_std_7d'] + 1e-9
            self.source_df[f'{col}_zscore'] = (self.source_df[col] - mean) / std

            print(f"  ↳ Features added for {self.target_stock}: Δ7d, %Δ7d, Δ30d, %Δ30d, mean, std, z-score")
        else:
            print(f"⚠️ {col} not found in source_df — skipping investment trend analysis.")

    
    def drop_raw_columns(self):
        print("[INFO] Dropping raw OHLCV and investment columns (except target)...")
        drop_cols = []
        for stock in self.stock_list:
            drop_cols.extend([f'{p}_{stock}' for p in ['close', 'open', 'low', 'high', 'volume']])
            drop_cols.append(f'ave_daily_invest_{stock}')
            if stock != self.target_stock:
                drop_cols.append(f'ratio_daily_invest_{stock}')
        self.source_df.drop(columns=[col for col in drop_cols if col in self.source_df.columns], inplace=True)

    def generate_features(self, target_df):
        print("[INFO] Generating technical indicators and engineered features...")
        fe = FeatureEngineering(df=target_df, target_stock=self.target_stock)
        self.feature_df = fe.generate_all_features().copy()
        print("[INFO] NaNs in feature_df:", self.feature_df.isnull().sum().sum())

    def clean_feature_df(self):
        print("[INFO] Cleaning feature_df: filtering date and removing NaNs...")
        self.feature_df = self.feature_df[self.feature_df.index > self.time_cutoff]
        self.feature_df.dropna(inplace=True)

        drop_cols = [f"{c}_{self.target_stock}" for c in ['open', 'low', 'high', 'volume']]
        self.feature_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    def merge_with_aux(self):
        print("[INFO] Merging engineered features with market-wide data...")
        self.merged_df = self.feature_df.join(self.source_df, how='inner').sort_index()
        self.merged_df.drop(columns=[c for c in ['is_real_date', 'price_avg_ohlc'] if c in self.merged_df.columns], inplace=True)

        # Rename target close column
        target_col = f"close_{self.target_stock}"
        if target_col in self.merged_df.columns:
            self.merged_df.rename(columns={target_col: "target"}, inplace=True)

    def report_summary(self):
        print("[INFO] Final merged dataset summary:")
        print("→ Shape      :", self.merged_df.shape)
        print("→ NaNs       :", self.merged_df.isnull().sum().sum())
        print("→ Index Range:", self.merged_df.index.min(), "→", self.merged_df.index.max())
        print("→ Columns    :", list(self.merged_df.columns)[:10], "...")
        print("[INFO] ✅ Dataset ready for modeling!")
