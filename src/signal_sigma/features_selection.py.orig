import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor


class ReducedFeatureSelector:
    def __init__(self, data: pd.DataFrame, target_col: str, top_n: int = 15):
        self.df = data.dropna().copy()
        self.target_col = target_col
        self.top_n = top_n
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]

    def calculate_metrics(self, y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
        }

    def plot_importances(self, importances, title):
        top_features = importances.sort_values(ascending=False).head(self.top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f"{title} (Top {self.top_n})")
        plt.tight_layout()
        plt.show()

    def select_features(self):
        metrics_dict = {}
        all_top_features = []

        # --- Pearson Correlation ---
        corr = self.X.corrwith(self.y).abs()
        top_corr = corr.sort_values(ascending=False).head(self.top_n)
        self.plot_importances(top_corr, "Pearson Correlation")
        all_top_features.extend(top_corr.index.tolist())

        # --- Mutual Information ---
        mi = mutual_info_regression(self.X, self.y, random_state=42)
        mi_series = pd.Series(mi, index=self.X.columns).sort_values(ascending=False)
        self.plot_importances(mi_series, "Mutual Information")
        all_top_features.extend(mi_series.head(self.top_n).index.tolist())

        # --- XGBoost ---
        xgb = XGBRegressor(n_estimators=300, random_state=42)
        xgb.fit(self.X, self.y)
        xgb_importances = pd.Series(xgb.feature_importances_, index=self.X.columns)
        self.plot_importances(xgb_importances, "XGBoost Feature Importance")
        metrics_dict["XGBoost"] = self.calculate_metrics(self.y, xgb.predict(self.X))
        all_top_features.extend(xgb_importances.nlargest(self.top_n).index.tolist())

        # --- Decision Tree ---
        dt = DecisionTreeRegressor(max_depth=5, random_state=42)
        dt.fit(self.X, self.y)
        dt_importances = pd.Series(dt.feature_importances_, index=self.X.columns)
        self.plot_importances(dt_importances, "Decision Tree Feature Importance")
        metrics_dict["Decision Tree"] = self.calculate_metrics(
            self.y, dt.predict(self.X)
        )
        all_top_features.extend(dt_importances.nlargest(self.top_n).index.tolist())

        # --- Random Forest ---
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(self.X, self.y)
        rf_importances = pd.Series(rf.feature_importances_, index=self.X.columns)
        self.plot_importances(rf_importances, "Random Forest Feature Importance")
        metrics_dict["Random Forest"] = self.calculate_metrics(
            self.y, rf.predict(self.X)
        )
        all_top_features.extend(rf_importances.nlargest(self.top_n).index.tolist())

        # --- Linear Regression ---
        lr = LinearRegression()
        lr.fit(self.X, self.y)
        lr_coef = pd.Series(np.abs(lr.coef_), index=self.X.columns).sort_values(
            ascending=False
        )
        self.plot_importances(lr_coef, "Linear Regression Coefficients")
        metrics_dict["Linear Regression"] = self.calculate_metrics(
            self.y, lr.predict(self.X)
        )
        all_top_features.extend(lr_coef.head(self.top_n).index.tolist())

        # --- Lasso Regression ---
        lasso = LassoCV(cv=5)
        lasso.fit(self.X, self.y)
        lasso_coef = pd.Series(np.abs(lasso.coef_), index=self.X.columns).sort_values(
            ascending=False
        )
        self.plot_importances(lasso_coef, "Lasso Regression Coefficients")
        metrics_dict["Lasso"] = self.calculate_metrics(self.y, lasso.predict(self.X))
        all_top_features.extend(lasso_coef.head(self.top_n).index.tolist())

        # --- Permutation Importance ---
        perm = permutation_importance(rf, self.X, self.y, n_repeats=10, random_state=42)
        perm_importance = pd.Series(
            perm.importances_mean, index=self.X.columns
        ).sort_values(ascending=False)
        self.plot_importances(perm_importance, "Permutation Importance")
        all_top_features.extend(perm_importance.head(self.top_n).index.tolist())

        # ✅ Deduplicate features while preserving order
        seen = set()
        final_features = [f for f in all_top_features if not (f in seen or seen.add(f))]

        print(
            f"\n✅ Final Selected Features (Deduplicated from all models): {len(final_features)}"
        )
        print(final_features)

        print("\n📊 Performance Metrics by Model:")
        for name, metrics in metrics_dict.items():
            print(f"{name}: {metrics}")

        return final_features, metrics_dict
