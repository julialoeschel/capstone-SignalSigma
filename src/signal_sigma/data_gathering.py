# ---
# description: Provides a data gathering utility for stock and macroeconomic data.
# ---

import os
import yfinance as yf
import pandas as pd
from datetime import datetime


class DataGathering:
    """
    tft_DataGathering is a utility class to download, merge, and clean stock price data
    and macroeconomic indicators using the Yahoo Finance API.

    Features:
    - Downloads historical stock data (Open, High, Low, Close, Volume).
    - Downloads macroeconomic indicators (e.g., indices, crypto, ETFs).
    - Automatically renames and merges data by date.
    - Outputs a clean CSV file with a datetime index.

    Parameters:
    ----------
    stock_tickers : list
        List of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
    macro_tickers : dict
        Dictionary mapping macro/crypto tickers to readable names
        (e.g., {"^GSPC": "S&P500_Index"}).
    start_date : str, default="2014-01-01"
        The start date for downloading data.
    end_date : str or None, default=None
        The end date for downloading data (defaults to today).
    save_path : str, default="../data/Stock_market_data"
        Directory path to save the final CSV.
    filename : str, default="clean_stock_data_with_time_index.csv"
        Filename for the final merged CSV.

    Usage:
    ------
    >>> tickers = ['AAPL', 'MSFT']
    >>> macro = {'^GSPC': 'S&P500_Index', 'BTC-USD': 'Bitcoin'}
    >>> gatherer = tft_DataGathering(tickers, macro)
    >>> df = gatherer.run()
    """

    def __init__(
        self,
        stock_tickers,
        macro_tickers,
        start_date="2014-01-01",
        end_date=None,
        save_path="../data/Stock_market_data",
        filename="clean_stock_data_with_time_index.csv",
    ):
        self.stock_tickers = stock_tickers
        self.macro_tickers = macro_tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.save_path = save_path
        self.filename = filename

    def download_stocks(self):
        print(f"[INFO] Downloading stock tickers: {self.stock_tickers}")
        all_dfs = []

        for ticker in self.stock_tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                )
                df.dropna(inplace=True)
                df.reset_index(inplace=True)

                column_map = {
                    "Open": f"open_{ticker}",
                    "High": f"high_{ticker}",
                    "Low": f"low_{ticker}",
                    "Close": f"close_{ticker}",
                    "Volume": f"volume_{ticker}",
                }

                existing_rename_map = {
                    k: v for k, v in column_map.items() if k in df.columns
                }
                df.rename(columns=existing_rename_map, inplace=True)
                df = df[["Date"] + list(existing_rename_map.values())]
                df.rename(columns={"Date": "date"}, inplace=True)

                all_dfs.append(df)
                print(f"[✓] Downloaded: {ticker}")
            except Exception as e:
                print(f"[✗] Error downloading {ticker}: {e}")

        if not all_dfs:
            raise RuntimeError("No stock data was successfully downloaded.")

        print("[INFO] Merging downloaded stock data...")
        merged_df = (
            all_dfs[0].rename(columns={"Date": "date"})
            if "Date" in all_dfs[0].columns
            else all_dfs[0]
        )
        for i, df in enumerate(all_dfs[1:], start=1):
            df.rename(columns={"Date": "date"}, inplace=True)
            merged_df = pd.merge(merged_df, df, on="date", how="outer")

        merged_df["date"] = pd.to_datetime(merged_df["date"])
        merged_df.set_index("date", inplace=True)
        merged_df.sort_index(inplace=True)

        # Clean up MultiIndex and duplicate column name patterns
        if isinstance(merged_df.columns, pd.MultiIndex):
            merged_df.columns = [
                "_".join(filter(None, col)).strip() for col in merged_df.columns
            ]
        merged_df.columns = [col.replace("__", "_") for col in merged_df.columns]
        merged_df.columns = [
            "_".join(dict.fromkeys(col.split("_"))) if "_" in col else col
            for col in merged_df.columns
        ]

        print(f"[INFO] Final stock data shape: {merged_df.shape}")
        return merged_df

    def download_macro(self):
        print("[INFO] Downloading macroeconomic & crypto data...")
        macro_df = pd.DataFrame()

        for ticker, label in self.macro_tickers.items():
            try:
                df = yf.download(
                    ticker, start=self.start_date, end=self.end_date, progress=False
                )
                macro_df[label] = df["Close"]
                print(f"[✓] Downloaded: {label}")
            except Exception as e:
                print(f"[✗] Error downloading {label}: {e}")

        macro_df.dropna(axis=1, how="all", inplace=True)
        macro_df.ffill(inplace=True)
        macro_df.bfill(inplace=True)
        macro_df.index.name = "date"
        print(f"[INFO] Final macro data shape: {macro_df.shape}")
        return macro_df

    def run(self):
        stock_df = self.download_stocks()
        macro_df = self.download_macro()

        print("[INFO] Merging stock and macro data...")
        full_df = stock_df.merge(
            macro_df, left_index=True, right_index=True, how="left"
        )
        full_df.index.name = "date"

        os.makedirs(self.save_path, exist_ok=True)
        full_path = os.path.join(self.save_path, self.filename)
        full_df.to_csv(full_path, date_format="%Y-%m-%d")

        print(f"[✓] Final dataset saved to: {full_path}")
        print(f"[INFO] Final merged shape: {full_df.shape}")
        return full_df
