#!/usr/bin/env python3
"""
Setup script for Streamlit deployment
Generates sample data and trains models for the dashboard
"""

import os
import sys
from data_generator import generate_subscriber_data
from feature_engineering import FeatureEngineer
from churn_model import ChurnPredictor

def setup_for_streamlit():
    """Generate data and train models for Streamlit deployment."""
    print("Setting up data for Streamlit deployment...")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('exports', exist_ok=True)
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_subscriber_data(n_subscribers=500, days_history=180)
    df.to_csv('data/subscriber_data.csv', index=False)
    print(f"Generated {len(df)} records for {df['subscriber_id'].nunique()} subscribers")
    
    # Feature engineering
    print("Creating features...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_features(df)
    feature_engineer.save_to_database(features_df)
    
    # Prepare model data
    X, y, feature_names = feature_engineer.prepare_model_data(features_df)
    
    # Train models
    print("Training models...")
    predictor = ChurnPredictor()
    best_model = predictor.train_models(X, y, feature_names)
    
    # Generate predictions
    print("Generating predictions...")
    predictions, probabilities = predictor.predict_churn(X)
    predictor.save_predictions_to_db(
        features_df['subscriber_id'], 
        predictions, 
        probabilities
    )
    
    # Save model
    predictor.save_model('models/churn_model.pkl')
    
    print("Setup complete! You can now run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    setup_for_streamlit() 