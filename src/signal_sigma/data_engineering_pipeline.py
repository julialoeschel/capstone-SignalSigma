# ---
# description: Provides a data engineering pipeline for stock and macroeconomic data.
# ---

from typing import List
import signal_sigma.config.cfg as cfg
from signal_sigma.data_gathering import DataGathering
from signal_sigma.data_preparator import DataPreparator
from signal_sigma.fred_macro import FredMacroProcessor
from signal_sigma.market_macro_compressor import MarketMacroCompressor
from signal_sigma.temporal_feature_combiner import TemporalFeatureCombiner
from signal_sigma.features_selection import ReducedFeatureSelector


class DataEngineeringPipeline:
    def __init__(
        self,
        path_stock: str,
        start_date: str,
        end_date: str,
        top_n_feature_important: int,
        tickers: List[str] = cfg.STOCK_TICKERS,
        macro_tickers: List[str] = cfg.MACRO_TICKERS,
    ):
        self.path_stock = path_stock
        self.start_date = start_date
        self.end_date = end_date
        self.top_n = top_n_feature_important

        # Define stock tickers and macroeconomic indicators
        self.tickers = tickers
        self.macro_tickers = macro_tickers

        self.model_dataset_combined_cols = {}
        self.final_featured_selected = {}

    def run(self):
        # === Step 1: Gather Raw Stock and Macro Data === #
        gatherer = DataGathering(
            stock_tickers=self.tickers,
            macro_tickers=self.macro_tickers,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        raw_data_stock_yahoo = gatherer.run()
        raw_data_stock_yahoo.index.name = "date"
        raw_data_stock_yahoo.to_csv(
            f"{self.path_stock}/stock_yahoo_{self.start_date}_{self.end_date}.csv"
        )

        # === Step 2: FRED Macro Indicators === #
        processor = FredMacroProcessor(
            start_date=self.start_date, save_path=self.path_stock
        )
        macro_FRED_df = processor.run_pipeline()
        macro_FRED_df.index.name = "date"
        macro_FRED_df.to_csv(
            f"{self.path_stock}/macro_FRED_{self.start_date}_{self.end_date}.csv"
        )

        # === Step 3: Yahoo Macro Compression === #
        compressor = MarketMacroCompressor(start=self.start_date)
        macro_Yahoo_df = compressor.generate_macro_features()
        macro_Yahoo_df.index.name = "date"
        macro_Yahoo_df.to_csv(
            f"{self.path_stock}/macro_Yahoo_{self.start_date}_{self.end_date}.csv"
        )

        # === Step 4: Merge All Datasets === #
        all_raw_data_merge_economy = raw_data_stock_yahoo.merge(
            macro_FRED_df, left_index=True, right_index=True, how="left"
        ).merge(macro_Yahoo_df, left_index=True, right_index=True, how="left")
        all_raw_data_merge_economy.ffill(inplace=True)
        all_raw_data_merge_economy.index.name = "date"
        all_raw_data_merge_economy.to_csv(
            f"{self.path_stock}/all_raw_data_merge_economy_{self.start_date}_{self.end_date}.csv"
        )

        # === Step 5: Feature Engineering and Selection Per Stock === #
        # XXX: Which which values to replace this hardcoded list?
        stock_list_top_targets = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
        ]

        # Optional: Drop seasonal columns if present
        seasonal_cols = [
            col
            for col in all_raw_data_merge_economy.columns
            if col.startswith("season_")
        ]
        all_raw_data_merge_economy = all_raw_data_merge_economy.drop(
            columns=seasonal_cols
        )

        for stock in stock_list_top_targets:
            # Stage 1: Feature Engineering
            preparator = DataPreparator(
                data=all_raw_data_merge_economy,
                target_stock=stock,
                stock_list=stock_list_top_targets,
                time_cutoff="2017-01-01",
            )
            model_dataset = preparator.prepare()
            model_dataset.index.name = "date"
            model_dataset.to_csv(
                f"{self.path_stock}/{stock}_model_dataset_featured_{self.start_date}_{self.end_date}.csv"
            )

            # Stage 1.5: Combine temporal features
            combiner = TemporalFeatureCombiner(model_dataset)
            model_dataset_combined = combiner.combine()

            # Drop common OHLCV columns
            prefixes_to_drop = (
                "close_",
                "open_",
                "low_",
                "high_",
                "volume_",
            )
            
            cols_to_drop = [
                col
                for col in model_dataset_combined.columns
                if col.startswith(prefixes_to_drop)
            ]
            model_dataset_combined.drop(columns=cols_to_drop, inplace=True)

            self.model_dataset_combined_cols[stock] = (
                model_dataset_combined.columns.tolist()
            )

            # Stage 2: ML Feature Selection
            selector = ReducedFeatureSelector(
                data=model_dataset_combined,
                target_col="target",
                top_n=self.top_n,
                stock_name=stock,
            )
            automatic_selected_features, model_scores = selector.select_features()
            important_features = automatic_selected_features.head(self.top_n)

            selected_feature_list = important_features.index.tolist()
            self.final_featured_selected[stock] = selected_feature_list

            # Stage 3: Save Reduced Dataset
            ready_reduced_features_dataset = model_dataset_combined[
                selected_feature_list + ["target"]
            ]
            ready_reduced_features_dataset.index.name = "date"
            reduced_path = f"{self.path_stock}/{stock}_reduced_dataset_{self.start_date}_{self.end_date}.csv"
            ready_reduced_features_dataset.to_csv(
                reduced_path, index=True, date_format="%Y-%m-%d"
            )

        print("\n✅ All data engineering steps completed successfully.")
        return self.final_featured_selected, self.model_dataset_combined_cols
