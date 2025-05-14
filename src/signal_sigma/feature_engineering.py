# ---
# description: Feature engineering for stock data.
# ---

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal


class FeatureEngineering:
    def __init__(self, df, target_stock):
        """
        Initialize the FeatureEngineering class.

        Parameters:
        - df: DataFrame containing stock data with DatetimeIndex.
        - target_stock: The stock symbol (e.g., 'AAPL') to engineer features for.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise KeyError(
                "The DataFrame index must be a DatetimeIndex for time series processing."
            )

        self.target_stock = target_stock.upper()

        # Align the DataFrame index to the official NYSE trading calendar
        print(
            "[TRACE] --- class FeatureEngineering: Reindexing to NYSE trading calendar..."
        )
        nyse = mcal.get_calendar("XNYS")
        trading_days = nyse.schedule(
            start_date=df.index.min().strftime("%Y-%m-%d"),
            end_date=df.index.max().strftime("%Y-%m-%d"),
        )
        full_index = pd.DatetimeIndex(trading_days.index)

        df = df.copy().reindex(full_index)
        df.interpolate(method="linear", limit_direction="both", inplace=True)
        df["is_real_date"] = df.index.isin(df.dropna(how="all").index).astype(int)
        self.df = df.sort_index()

        # Define column names for target stock
        self.close_col = f"close_{self.target_stock}"
        self.open_col = f"open_{self.target_stock}"
        self.high_col = f"high_{self.target_stock}"
        self.low_col = f"low_{self.target_stock}"
        self.volume_col = f"volume_{self.target_stock}"

    def generate_all_features(self):
        """
        Generate all engineered features for the target stock.
        Returns a DataFrame with added feature columns.
        """
        print(
            "[TRACE] --- class FeatureEngineering: Starting full feature generation..."
        )
        self._add_trend_indicators()
        self._add_momentum_indicators()
        self._add_volatility_indicators()
        self._add_volume_indicators()
        self._add_seasonal_features()
        self._add_return_features()
        self._calculate_risk_metrics()
        self._dily_price_features()
        self.df.dropna(inplace=True)  # ← Drop only after all features are calculated
        self.df = self.df.copy()  # Defragment DataFrame for performance
        print(
            "[TRACE] --- class FeatureEngineering: Feature generation complete. Returning final DataFrame."
        )
        return self.df

    def _add_trend_indicators(self):
        """
        Add trend-based technical indicators: SMA, EMA, MACD.
        """
        print("[TRACE] --- class FeatureEngineering: add_trend_indicators")
        close = self.df[self.close_col]
        high = self.df[self.high_col]
        low = self.df[self.low_col]

        for window in [7, 100, 300]:
            self.df[f"trend_sma_{window}"] = close.rolling(window=window).mean()

        for window in [50, 100, 250, 1000]:
            self.df[f"trend_ema_{window}"] = close.ewm(span=window, adjust=False).mean()

        short_ema = close.ewm(span=12, adjust=False).mean()
        long_ema = close.ewm(span=26, adjust=False).mean()
        self.df["trend_macd"] = short_ema - long_ema
        self.df["trend_macd_signal"] = (
            self.df["trend_macd"].ewm(span=9, adjust=False).mean()
        )

    def _add_momentum_indicators(self):
        """
        Add momentum-based indicators: RSI, Momentum (diff), ROC, and Stochastic.
        """
        print("[TRACE] --- class FeatureEngineering: add_momentum_indicators")
        close = self.df[self.close_col]
        high = self.df[self.high_col]
        low = self.df[self.low_col]

        for period in [3, 75, 150]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            self.df[f"momentum_rsi_{period}"] = rsi

        for period in [7, 50, 100]:
            self.df[f"momentum_diff_{period}"] = close.diff(period)
            self.df[f"momentum_roc_{period}"] = close.pct_change(periods=period)

        self.df["momentum_stoch_k"], self.df["momentum_stoch_d"] = (
            self._calculate_stochastic(high, low, close)
        )

    def _calculate_stochastic(self, high, low, close, period=14):
        """
        Calculate stochastic oscillator components: %K and %D.
        """
        print("[TRACE] --- class FeatureEngineering: calculate_stochastic")
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        percent_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        percent_d = percent_k.rolling(window=3).mean()
        return percent_k, percent_d

    def _add_volatility_indicators(self):
        """
        Add volatility indicators: ATR and Donchian Channels.
        """
        print("[TRACE] --- class FeatureEngineering: add_volatility_indicators")

        close = self.df[self.close_col]
        high = self.df[self.high_col]
        low = self.df[self.low_col]

        for window in [3, 14, 45]:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.df[f"volatility_atr_{window}"] = tr.rolling(window=window).mean()
            self.df[f"volatility_donchian_high_{window}"] = high.rolling(
                window=window
            ).max()
            self.df[f"volatility_donchian_low_{window}"] = low.rolling(
                window=window
            ).min()

    def _add_volume_indicators(self):
        """
        Add volume-based indicators: OBV, CMF, A/D, VPT.
        """
        print("[TRACE] --- class FeatureEngineering: add_volume_indicators")
        close = self.df[self.close_col]
        high = self.df[self.high_col]
        low = self.df[self.low_col]
        volume = self.df[self.volume_col]

        prev_close = close.shift(1)
        self.df["volume_obv"] = (
            volume.where(close > prev_close, -volume).fillna(0).cumsum()
        )
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
        self.df["volume_cmf"] = (mf_multiplier * volume).rolling(
            window=20
        ).sum() / volume.rolling(window=20).sum()
        self.df["volume_ad"] = (
            ((close - low) - (high - close)) / (high - low + 1e-9) * volume
        )
        self.df["volume_vpt"] = (
            ((close - prev_close) / prev_close * volume).fillna(0).cumsum()
        )

    def _add_seasonal_features(self):
        """
        Add seasonal and cyclical time features.
        Includes lags, diffs, and sin/cos encodings for monthly, weekly, and quarterly cycles.
        """
        print("[TRACE] --- class FeatureEngineering: add_seasonal_features")
        close = self.df[self.close_col]
        for lag in [14, 35, 100]:
            self.df[f"lag_{lag}"] = close.shift(lag)
            self.df[f"diff_{lag}"] = close.diff(lag)

        self.df["season_year"] = self.df.index.year
        self.df["season_month"] = self.df.index.month
        self.df["season_day_of_week"] = self.df.index.dayofweek
        self.df["season_day_of_month"] = self.df.index.day
        self.df["season_week"] = self.df.index.isocalendar().week.astype(int)
        self.df["season_weekday"] = self.df.index.weekday
        self.df["season_quarter"] = self.df.index.quarter

        # Encode cyclical seasonality
        def encode_cyclical(df, col, max_val):
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
            df.drop(columns=[col], inplace=True)

        cyclical_mappings = {
            "season_month": 12,
            "season_day_of_week": 7,
            "season_day_of_month": 31,
            "season_week": 53,
            "season_weekday": 7,
            "season_quarter": 4,
        }
        for col, max_val in cyclical_mappings.items():
            encode_cyclical(self.df, col, max_val)

    def _add_return_features(self):
        """
        Add percentage and log return features over multiple time horizons.
        """
        print("[TRACE] --- class FeatureEngineering: add_return_features")
        close = self.df[self.close_col]
        self.df["return_1d"] = close.pct_change()
        for window in [5, 25, 45]:
            self.df[f"return_{window}d"] = close.pct_change(periods=window)

        log_close = np.log(close + 1e-9)
        self.df["log_return_1d"] = log_close.diff()
        for window in [15, 75, 125]:
            self.df[f"log_return_{window}d"] = log_close.diff(window)

    def _calculate_risk_metrics(self):
        """
        Compute risk-related metrics including drawdowns and Fibonacci levels.
        """
        print("[TRACE] --- class FeatureEngineering: calculate_risk_metrics")
        close = self.df[self.close_col]
        self.df["risk_drawdown"] = (close - close.cummax()) / close.cummax()

        windows = [21, 150, 600]
        for window in windows:
            recent_max = close.rolling(window=window).max()
            recent_min = close.rolling(window=window).min()
            diff = recent_max - recent_min
            self.df[f"risk_fib_0.236_{window}"] = recent_max - diff * 0.236
            self.df[f"risk_fib_0.382_{window}"] = recent_max - diff * 0.382
            self.df[f"risk_fib_0.618_{window}"] = recent_max - diff * 0.618

        for window in windows:
            for level in ["0.236", "0.382", "0.618"]:
                col = f"risk_fib_{level}_{window}"
                self.df[f"risk_diff_from_fib_{level}_{window}"] = close - self.df[col]
                self.df[f"risk_ratio_to_fib_{level}_{window}"] = close / (
                    self.df[col] + 1e-9
                )

    def _dily_price_features(self):
        """
        Add price structure and liquidity-adjusted investment behavior features.
        """
        print("[TRACE] --- class FeatureEngineering: dily_price_features")
        close = self.df[self.close_col]
        high = self.df[self.high_col]
        low = self.df[self.low_col]
        open = self.df[self.open_col]
        volume = self.df[self.volume_col]

        for window in [30, 250, 650]:
            self.df[f"trend_sma_{window}"] = close.rolling(window=window).mean()

        self.df["price_avg_ohlc"] = (close + high + low + open) / 4
        self.df["price_investment"] = self.df["price_avg_ohlc"] * volume
        self.df["price_investment_trend_7d"] = self.df["price_investment"].diff(7)
        self.df["price_investment_trend_50d"] = self.df["price_investment"].diff(50)
        self.df["price_investment_trend_150d"] = self.df["price_investment"].diff(150)
        self.df["price_investment_ma_30d"] = (
            self.df["price_investment"].rolling(window=30).mean()
        )
        self.df["price_investment_deviation_30d"] = (
            self.df["price_investment_ma_30d"] - self.df["price_investment_trend_7d"]
        ) / (self.df["price_investment"] + 1e-9)

        for window in [30, 250, 650]:
            self.df[f"price_investment_vs_indicators_{window}"] = (
                self.df["price_investment"] / self.df[f"trend_sma_{window}"]
            )
            self.df[f"price_investment_ma_{window}"] = (
                self.df["price_investment"].rolling(window).mean()
            )
