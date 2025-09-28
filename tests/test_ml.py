"""Test suite for ML functionality."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
import tempfile
import numpy as np

from src.crypto_dashboard.ml import DirectionPredictor, train_direction_predictor
from src.crypto_dashboard.processing import prepare_price_frame


def _sample_extended_prices() -> pd.DataFrame:
    """Create a larger sample dataset for ML testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    
    # Generate synthetic price data with some trend and volatility
    prices = []
    price = 40000
    for _ in range(100):
        price += (np.random.randn() * 500) + (np.random.randn() * 50)
        prices.append(max(price, 1000))  # Prevent negative prices
    
    frame = pd.DataFrame({
        "timestamp": dates,
        "price": prices
    }).set_index("timestamp")
    
    return frame


def test_direction_predictor_initialization():
    """Test DirectionPredictor initialization."""
    predictor = DirectionPredictor()
    assert not predictor.is_fitted
    assert len(predictor.feature_columns) == 0


def test_prepare_features():
    """Test feature preparation for ML."""
    frame = prepare_price_frame(_sample_extended_prices())
    predictor = DirectionPredictor()
    
    X, y = predictor.prepare_features(frame)
    
    # Check that features and target are created
    assert len(X) > 0
    assert len(y) > 0
    assert len(X) == len(y)
    
    # Check feature columns are set
    assert len(predictor.feature_columns) > 0
    assert "return" in predictor.feature_columns
    assert "is_weekend" in predictor.feature_columns


def test_model_training():
    """Test ML model training."""
    frame = prepare_price_frame(_sample_extended_prices())
    predictor = DirectionPredictor()
    
    results = predictor.train(frame)
    
    # Check that model is trained
    assert predictor.is_fitted
    
    # Check results structure
    assert "metrics" in results
    assert "feature_importance" in results
    assert "classification_report" in results
    
    # Check metrics
    metrics = results["metrics"]
    assert "test_accuracy" in metrics
    assert "cv_mean" in metrics
    assert 0 <= metrics["test_accuracy"] <= 1


def test_model_prediction():
    """Test making predictions with trained model."""
    frame = prepare_price_frame(_sample_extended_prices())
    predictor = DirectionPredictor()
    
    # Train the model
    predictor.train(frame)
    
    # Make prediction
    direction, confidence = predictor.predict(frame)
    
    # Check prediction format
    assert direction in [0, 1]
    assert 0 <= confidence <= 1


def test_model_save_load():
    """Test saving and loading trained models."""
    frame = prepare_price_frame(_sample_extended_prices())
    predictor = DirectionPredictor()
    
    # Train the model
    predictor.train(frame)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        predictor.save_model(temp_path)
        
        # Load the model
        loaded_predictor = DirectionPredictor.load_model(temp_path)
        
        # Check that loaded model works
        assert loaded_predictor.is_fitted
        assert loaded_predictor.feature_columns == predictor.feature_columns
        
        # Test prediction with loaded model
        direction, confidence = loaded_predictor.predict(frame)
        assert direction in [0, 1]
        
    finally:
        temp_path.unlink(missing_ok=True)


def test_train_direction_predictor_function():
    """Test the high-level training function."""
    frame = _sample_extended_prices()
    
    predictor, results = train_direction_predictor(frame)
    
    # Check that we get a trained predictor
    assert predictor.is_fitted
    assert "metrics" in results
    assert results["metrics"]["test_accuracy"] > 0


def test_insufficient_data_error():
    """Test error handling for insufficient data."""
    # Create tiny dataset
    small_frame = _sample_extended_prices().head(10)
    frame = prepare_price_frame(small_frame)
    
    predictor = DirectionPredictor()
    
    with pytest.raises(ValueError, match="Insufficient data"):
        predictor.train(frame)


def test_prediction_before_training_error():
    """Test error when trying to predict before training."""
    frame = prepare_price_frame(_sample_extended_prices())
    predictor = DirectionPredictor()
    
    with pytest.raises(ValueError, match="Model must be trained"):
        predictor.predict(frame)


def test_save_untrained_model_error():
    """Test error when trying to save untrained model."""
    predictor = DirectionPredictor()
    
    with tempfile.NamedTemporaryFile(suffix='.joblib') as f:
        temp_path = Path(f.name)
        
        with pytest.raises(ValueError, match="Cannot save untrained model"):
            predictor.save_model(temp_path)