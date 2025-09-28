"""Machine learning extensions for crypto price prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

from .processing import add_features, prepare_price_frame

logger = logging.getLogger(__name__)


class DirectionPredictor:
    """Light gradient boosting classifier for next-day price direction."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 3, random_state: int = 42):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            learning_rate=0.1,
        )
        self.feature_columns: list[str] = []
        self.is_fitted = False

    def prepare_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable for ML training."""
        enriched = add_features(frame, rolling_windows=(7, 14, 30))
        
        # Create target: 1 if next day price is higher, 0 otherwise
        enriched["target"] = (enriched["price"].shift(-1) > enriched["price"]).astype(int)
        
        # Select features that are available at prediction time
        feature_columns = [
            "return",
            "log_return",
            "rolling_mean_7",
            "rolling_mean_14", 
            "rolling_mean_30",
            "rolling_std_7",
            "rolling_std_14",
            "rolling_std_30",
            "rolling_volatility_7",
            "rolling_volatility_14",
            "rolling_volatility_30",
            "cum_return",
            "drawdown",
        ]
        
        # Add day-of-week features
        enriched["day_of_week"] = enriched.index.dayofweek
        enriched["is_weekend"] = (enriched.index.dayofweek >= 5).astype(int)
        feature_columns.extend(["day_of_week", "is_weekend"])
        
        # Add price momentum features
        enriched["price_vs_sma7"] = enriched["price"] / enriched["rolling_mean_7"] - 1
        enriched["price_vs_sma30"] = enriched["price"] / enriched["rolling_mean_30"] - 1
        feature_columns.extend(["price_vs_sma7", "price_vs_sma30"])
        
        # Remove rows with NaN values and the last row (no target)
        clean_data = enriched[feature_columns + ["target"]].dropna()
        
        X = clean_data[feature_columns]
        y = clean_data["target"]
        
        self.feature_columns = feature_columns
        return X, y

    def train(self, frame: pd.DataFrame, test_size: float = 0.25) -> dict[str, float]:
        """Train the classifier and return performance metrics."""
        X, y = self.prepare_features(frame)
        
        if len(X) < 50:
            raise ValueError("Insufficient data for training (need at least 50 samples)")
        
        # Split data chronologically (no shuffle for time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        logger.info("Training with %d samples, testing with %d samples", len(X_train), len(X_test))
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate performance
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        metrics = {
            "test_accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "baseline_accuracy": max(y_test.mean(), 1 - y_test.mean()),  # Always predict majority class
        }
        
        logger.info("Model performance: Test accuracy %.3f, CV %.3f±%.3f", 
                   accuracy, cv_scores.mean(), cv_scores.std())
        
        return {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def predict(self, frame: pd.DataFrame) -> tuple[int, float]:
        """Predict direction for the next day."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        enriched = add_features(frame, rolling_windows=(7, 14, 30))
        
        # Prepare the same features as training
        latest_data = enriched.iloc[[-1]].copy()  # Get last row
        latest_data["day_of_week"] = latest_data.index.dayofweek
        latest_data["is_weekend"] = (latest_data.index.dayofweek >= 5).astype(int)
        latest_data["price_vs_sma7"] = latest_data["price"] / latest_data["rolling_mean_7"] - 1
        latest_data["price_vs_sma30"] = latest_data["price"] / latest_data["rolling_mean_30"] - 1
        
        X = latest_data[self.feature_columns]
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0].max()
        
        return int(prediction), float(probability)

    def save_model(self, path: Path) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load_model(cls, path: Path) -> "DirectionPredictor":
        """Load a trained model from disk."""
        model_data = joblib.load(path)
        
        predictor = cls()
        predictor.model = model_data["model"]
        predictor.feature_columns = model_data["feature_columns"]
        predictor.is_fitted = model_data["is_fitted"]
        
        logger.info("Model loaded from %s", path)
        return predictor


def train_direction_predictor(
    price_data: pd.DataFrame,
    model_path: Optional[Path] = None,
) -> tuple[DirectionPredictor, dict]:
    """Train a direction predictor and optionally save it."""
    predictor = DirectionPredictor()
    
    # Prepare and validate data
    prepared_data = prepare_price_frame(price_data)
    
    # Train the model
    results = predictor.train(prepared_data)
    
    # Save model if path provided
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(model_path)
    
    return predictor, results