
import warnings
=======
# ---
# description: Provides a feature selection utility for stock forecasting.
# ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

# Optional XGBoost import, fallback to RandomForest if missing
try:
    from xgboost import XGBRegressor
except ImportError:
    warnings.warn("XGBoost nicht gefunden; verwende RandomForestRegressor als Fallback.")
    from sklearn.ensemble import RandomForestRegressor as XGBRegressor

class ReducedFeatureSelector:
    """
    Wählt die Top-K-Features basierend auf einem Regressionsmodell.
    Falls XGBoost installiert ist, wird XGBRegressor genutzt, sonst RandomForestRegressor.

    Methoden:
        select_features() -> (selected_features: list, metrics: dict)
    """
    def __init__(self, data: pd.DataFrame, target_col: str):
        self.data = data
        self.target_col = target_col

    def select_features(self):
        """
        Führt Features-Selection durch und gibt gewählte Features sowie Performance-Metriken.

        Returns:
            selected (list): Top 10 Feature-Namen nach Wichtigkeit.
            metrics (dict): Wörterbuch mit {'rmse', 'mae', 'mape'} auf dem Test-Set.
        """
        # Input-Daten trennen
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]

        # Trainings-/Test-Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Modell initialisieren und trainieren
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Kreuzvalidierte RMSE
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
        )
        rmse_cv = np.sqrt(-cv_scores)

        # Feature-Importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            imp = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            importances = imp.importances_mean

        # Top-10 Features auswählen
        feat_imp = pd.Series(importances, index=X.columns)
        selected = feat_imp.nlargest(10).index.tolist()

        # Metriken auf Test-Set
        y_pred = model.predict(X_test)
        metrics = {
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred)
        }

        return selected, metrics
