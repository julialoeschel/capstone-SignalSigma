# ---
# description: Provides a class to fetch, normalize, and compress macro-market indicators into composite features.
# ---

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from signal_sigma.config import cfg


class MarketMacroCompressor:
    """
    🧠 MarketMacroCompressor - A class to fetch, normalize, and compress macro-market indicators
    into 3 interpretable, weighted composite features suitable for machine learning models.

    These features reflect:
    - Market stress & liquidity conditions
    - Innovation & growth appetite
    - Demand for real assets & inflation hedges

    Use this class to reduce dimensionality while retaining economic insight.
    """

    DEFAULT_MACROS = cfg.MACROS_ALT

    def __init__(self, start="2000-01-01", end=None):
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.indicator_symbols = self.DEFAULT_MACROS
        # XXX: Why this selection

    def fetch_data(self):
        """
        📡 Download daily closing prices for all macro indicators.
        """
        print("📥 Fetching macro indicators from Yahoo Finance...")
        data = yf.download(
            list(self.indicator_symbols.keys()), start=self.start, end=self.end
        )["Close"]
        data.rename(columns=self.indicator_symbols, inplace=True)
        return data

    def preprocess_and_scale(self, df):
        """
        🧼 Clean missing values and apply Z-score scaling to normalize features.
        This ensures fair weighting during composite calculation.
        """
        print("🧪 Preprocessing: Filling missing values...")
        df = df.ffill().bfill()
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df), index=df.index, columns=df.columns
        )
        return df_scaled

    def create_composite_features(self, df_scaled):
        """
        🏗️ Construct 3 macro feature composites using economic logic and impact weighting.

        Composite features:
        - market_stress
        - growth_innovation_sentiment
        - real_asset_confidence
        """

        # -----------------------------------
        # 🔥 1. Market Stress
        # Captures fear, tightening liquidity, and macro instability.
        # Higher values => higher risk aversion and economic stress.
        # -----------------------------------
        df_scaled["Yahoo_mcro_market_stress"] = (
            0.35 * df_scaled["vix"]  # fear index
            + 0.25 * df_scaled["10y_yield"]  # long-term rate
            + 0.20 * df_scaled["3mo_yield"]  # short-term monetary policy signal
            + 0.20 * df_scaled["dxy"]  # strong dollar = tight global liquidity
        )

        # -----------------------------------
        # 🚀 2. Growth & Innovation Sentiment
        # Represents optimism about future innovation and risk-taking.
        # Higher values => bullish tech/speculative appetite.
        # -----------------------------------
        df_scaled["Yahoo_mcro_growth_innovation_sentiment"] = (
            0.30 * df_scaled["nasdaq"]  # growth-heavy index
            + 0.25 * df_scaled["qqq"]  # tech ETF
            + 0.20 * df_scaled["bitcoin"]  # risk-on/speculative barometer
            + 0.15 * df_scaled["arkk"]  # highly speculative
            + 0.10 * df_scaled["tech_etf"]  # broad tech optimism
        )

        # -----------------------------------
        # 🛢️ 3. Real Asset Confidence
        # Demand for hard/tangible assets reflecting inflation pressure and real growth.
        # Higher values => investor preference for real/physical assets.
        # -----------------------------------
        df_scaled["Yahoo_mcro_real_asset_confidence"] = (
            0.30 * df_scaled["oil"]  # inflation / demand proxy
            + 0.25 * df_scaled["gold"]  # inflation + safe haven
            + 0.20 * df_scaled["energy_etf"]  # commodity earnings exposure
            + 0.15 * df_scaled["longbond_etf"]  # rate hedging
            + 0.10 * df_scaled["bond_market_etf"]  # broad fixed income
        )

        return df_scaled[
            [
                "Yahoo_mcro_market_stress",
                "Yahoo_mcro_growth_innovation_sentiment",
                "Yahoo_mcro_real_asset_confidence",
            ]
        ]

    def generate_macro_features(self):
        """
        🔄 Main entry point: fetch, clean, scale, and return composite macro features.
        Returns:
            pd.DataFrame with 3 columns and date index
        """
        raw = self.fetch_data()
        scaled = self.preprocess_and_scale(raw)
        final_df = self.create_composite_features(scaled)
        print("✅ Macro composite features created.")
        return final_df
