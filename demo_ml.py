"""Simple ML demonstration script."""

from src.crypto_dashboard.api import CoinGeckoClient, MarketDataRequest
from src.crypto_dashboard.processing import prepare_price_frame, add_features
from src.crypto_dashboard.ml import train_direction_predictor


def demo_ml_pipeline(coin_id: str = "bitcoin", days: int = 30) -> None:
    """Demo the complete ML pipeline."""
    print(f"ML Pipeline Demo for {coin_id} ({days} days)")
    print("=" * 50)
    
    # Fetch fresh data
    print("Fetching data...")
    try:
        client = CoinGeckoClient()
        request = MarketDataRequest(coin_id=coin_id, days=days)
        data = client.market_chart(request)
        print(f"   Fetched {len(data)} data points")
    except Exception as e:
        print(f"Failed to fetch data for {coin_id}: {e}")
        return
    
    # Prepare the data
    print("Preparing features...")
    prepared = prepare_price_frame(data)
    price_frame = add_features(prepared)
    print(f"   Data shape: {price_frame.shape}")
    print(f"   Date range: {price_frame.index.min().date()} to {price_frame.index.max().date()}")
    
    # Train the model
    print("Training ML model...")
    try:
        predictor, results = train_direction_predictor(price_frame)
        
        # Display results
        metrics = results["metrics"]
        print(f"   Model trained successfully!")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"   CV Mean Score: {metrics['cv_mean']:.3f} (+/-{metrics['cv_std']:.3f})")
        
        # Show feature importance
        print("\nTop 5 Most Important Features:")
        importance_data = results["feature_importance"]
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            print(f"   {i}. {feature}: {importance:.3f}")
        
        # Make a prediction for tomorrow
        print("\nNext-Day Prediction:")
        direction, confidence = predictor.predict(price_frame)
        direction_text = "UP" if direction == 1 else "DOWN"
        print(f"   Direction: {direction_text}")
        print(f"   Confidence: {confidence:.1%}")
        
        print("\nClassification Report:")
        print(results["classification_report"])
        
    except ValueError as e:
        print(f"   Training failed: {e}")
        
    print("\n" + "=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    # Demo with Bitcoin
    demo_ml_pipeline("bitcoin", days=60)
    
    print("\n" + "=" * 70)
    
    # Demo with Ethereum  
    demo_ml_pipeline("ethereum", days=60)