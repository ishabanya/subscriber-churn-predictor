#!/usr/bin/env python3
"""
Subscriber Churn Prediction Pipeline

Main script to run the complete churn prediction workflow.
Handles data generation, feature engineering, model training, and dashboard launch.

Usage:
    python main_pipeline.py [--generate-data] [--train-model] [--launch-dashboard]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_generator import generate_subscriber_data
from feature_engineering import FeatureEngineer
from churn_model import ChurnPredictor

def setup_directories():
    """Create necessary directories for outputs."""
    directories = ['data', 'models', 'reports', 'exports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directories created")

def generate_sample_data(n_subscribers=1000, days_history=365):
    """Generate sample subscriber data."""
    print("\n" + "="*60)
    print("STEP 1: GENERATING SAMPLE SUBSCRIBER DATA")
    print("="*60)
    
    try:
        # Generate data
        df = generate_subscriber_data(n_subscribers=n_subscribers, days_history=days_history)
        
        # Save to CSV
        output_path = 'data/subscriber_data.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✓ Generated {len(df)} records for {df['subscriber_id'].nunique()} subscribers")
        print(f"✓ Churn rate: {df['churned'].mean():.2%}")
        print(f"✓ Data saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return None

def run_feature_engineering(data_path='data/subscriber_data.csv'):
    """Run feature engineering pipeline."""
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    try:
        # Load data
        print("Loading subscriber data...")
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} records")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Create features
        features_df = feature_engineer.create_features(df)
        
        # Save features to database
        feature_engineer.save_to_database(features_df)
        
        # Prepare data for modeling
        X, y, feature_names = feature_engineer.prepare_model_data(features_df)
        
        print(f"✓ Created {len(features_df)} subscriber records with {len(features_df.columns)} features")
        print(f"✓ Model data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return features_df, X, y, feature_names, feature_engineer
        
    except Exception as e:
        print(f"✗ Error in feature engineering: {e}")
        return None, None, None, None, None

def train_churn_model(X, y, feature_names):
    """Train and evaluate churn prediction models."""
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    try:
        # Initialize churn predictor
        predictor = ChurnPredictor()
        
        # Train models
        best_model = predictor.train_models(X, y, feature_names)
        
        # Generate model report
        predictor.generate_model_report('reports/model_report.txt')
        
        # Create performance plots
        predictor.plot_model_performance('reports/model_performance.png')
        
        # Save model
        model_path = 'models/churn_model.pkl'
        predictor.save_model(model_path)
        
        print(f"✓ Best model trained and saved to: {model_path}")
        print(f"✓ Model report saved to: reports/model_report.txt")
        print(f"✓ Performance plots saved to: reports/model_performance.png")
        
        return predictor
        
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return None

def generate_predictions(predictor, features_df, feature_engineer):
    """Generate predictions for all subscribers."""
    print("\n" + "="*60)
    print("STEP 4: GENERATING PREDICTIONS")
    print("="*60)
    
    try:
        # Prepare data for prediction
        X_pred, _, _ = feature_engineer.prepare_model_data(features_df)
        
        # Make predictions
        predictions, probabilities = predictor.predict_churn(X_pred)
        
        # Save predictions to database
        pred_df = predictor.save_predictions_to_db(
            features_df['subscriber_id'], 
            predictions, 
            probabilities
        )
        
        # Create high-risk subscribers report
        high_risk = pred_df[pred_df['churn_probability'] >= 0.7]
        high_risk_report = features_df[features_df['subscriber_id'].isin(high_risk['subscriber_id'])]
        
        # Save high-risk report
        high_risk_report.to_csv('exports/high_risk_subscribers.csv', index=False)
        
        print(f"✓ Generated predictions for {len(predictions)} subscribers")
        print(f"✓ High-risk subscribers (≥70% churn probability): {len(high_risk)}")
        print(f"✓ High-risk report saved to: exports/high_risk_subscribers.csv")
        
        return pred_df
        
    except Exception as e:
        print(f"✗ Error generating predictions: {e}")
        return None

def create_summary_report(features_df, pred_df):
    """Create a comprehensive summary report."""
    print("\n" + "="*60)
    print("STEP 5: GENERATING SUMMARY REPORT")
    print("="*60)
    
    try:
        # Merge features with predictions
        if pred_df is not None:
            full_df = features_df.merge(pred_df, on='subscriber_id', how='left')
        else:
            full_df = features_df
        
        # Create summary statistics
        summary_stats = {
            'total_subscribers': len(full_df),
            'churn_rate': full_df['churned'].mean() if 'churned' in full_df.columns else None,
            'avg_monthly_revenue': full_df['avg_monthly_revenue'].mean(),
            'avg_engagement_score': full_df['engagement_score'].mean(),
            'high_risk_count': len(full_df[full_df['churn_probability'] >= 0.7]) if 'churn_probability' in full_df.columns else None,
            'medium_risk_count': len(full_df[(full_df['churn_probability'] >= 0.4) & (full_df['churn_probability'] < 0.7)]) if 'churn_probability' in full_df.columns else None,
            'low_risk_count': len(full_df[full_df['churn_probability'] < 0.4]) if 'churn_probability' in full_df.columns else None
        }
        
        # Plan distribution
        plan_dist = full_df['plan'].value_counts()
        
        # Location distribution
        location_dist = full_df['location'].value_counts()
        
        # Create summary report
        with open('reports/summary_report.txt', 'w') as f:
            f.write("SUBSCRIBER CHURN PREDICTION - SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("KEY METRICS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Subscribers: {summary_stats['total_subscribers']:,}\n")
            if summary_stats['churn_rate']:
                f.write(f"Historical Churn Rate: {summary_stats['churn_rate']:.2%}\n")
            f.write(f"Average Monthly Revenue: ${summary_stats['avg_monthly_revenue']:.2f}\n")
            f.write(f"Average Engagement Score: {summary_stats['avg_engagement_score']:.2f}\n\n")
            
            if summary_stats['high_risk_count']:
                f.write("RISK ASSESSMENT:\n")
                f.write("-" * 18 + "\n")
                f.write(f"High Risk (≥70%): {summary_stats['high_risk_count']:,} subscribers\n")
                f.write(f"Medium Risk (40-70%): {summary_stats['medium_risk_count']:,} subscribers\n")
                f.write(f"Low Risk (<40%): {summary_stats['low_risk_count']:,} subscribers\n\n")
            
            f.write("PLAN DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            for plan, count in plan_dist.items():
                f.write(f"{plan}: {count:,} subscribers ({count/len(full_df):.1%})\n")
            f.write("\n")
            
            f.write("LOCATION DISTRIBUTION:\n")
            f.write("-" * 22 + "\n")
            for location, count in location_dist.items():
                f.write(f"{location}: {count:,} subscribers ({count/len(full_df):.1%})\n")
        
        # Export full dataset for BI tools
        full_df.to_csv('exports/full_subscriber_analytics.csv', index=False)
        
        print("✓ Summary report generated: reports/summary_report.txt")
        print("✓ Full dataset exported: exports/full_subscriber_analytics.csv")
        
    except Exception as e:
        print(f"✗ Error creating summary report: {e}")

def launch_dashboard():
    """Launch the interactive dashboard."""
    print("\n" + "="*60)
    print("STEP 6: LAUNCHING INTERACTIVE DASHBOARD")
    print("="*60)
    
    try:
        print("Starting Churn Analytics Dashboard...")
        print("Access the dashboard at: http://127.0.0.1:8050")
        print("Press Ctrl+C to stop the dashboard")
        
        # Import and run dashboard
        from dashboard import app
        app.run(debug=False, host='0.0.0.0', port=8050)
        
    except KeyboardInterrupt:
        print("\n✓ Dashboard stopped by user")
    except Exception as e:
        print(f"✗ Error launching dashboard: {e}")

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Subscriber Churn Prediction Pipeline')
    parser.add_argument('--generate-data', action='store_true', 
                       help='Generate sample subscriber data')
    parser.add_argument('--train-model', action='store_true',
                       help='Train churn prediction model')
    parser.add_argument('--launch-dashboard', action='store_true',
                       help='Launch interactive dashboard')
    parser.add_argument('--subscribers', type=int, default=1000,
                       help='Number of subscribers to generate (default: 1000)')
    parser.add_argument('--days', type=int, default=365,
                       help='Days of historical data (default: 365)')
    
    args = parser.parse_args()
    
    # If no specific steps are requested, run the full pipeline
    if not any([args.generate_data, args.train_model, args.launch_dashboard]):
        args.generate_data = True
        args.train_model = True
        args.launch_dashboard = True
    
    print("🚀 SUBSCRIBER CHURN PREDICTION PIPELINE")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  - Subscribers: {args.subscribers:,}")
    print(f"  - Historical days: {args.days}")
    print(f"  - Generate data: {args.generate_data}")
    print(f"  - Train model: {args.train_model}")
    print(f"  - Launch dashboard: {args.launch_dashboard}")
    print()
    
    # Setup
    setup_directories()
    
    # Step 1: Generate data
    if args.generate_data:
        df = generate_sample_data(args.subscribers, args.days)
        if df is None:
            print("✗ Failed to generate data. Exiting.")
            return
    
    # Step 2: Feature engineering
    if args.train_model:
        features_df, X, y, feature_names, feature_engineer = run_feature_engineering()
        if features_df is None:
            print("✗ Failed to run feature engineering. Exiting.")
            return
    
    # Step 3: Model training
    if args.train_model:
        predictor = train_churn_model(X, y, feature_names)
        if predictor is None:
            print("✗ Failed to train model. Exiting.")
            return
    
    # Step 4: Generate predictions
    if args.train_model:
        pred_df = generate_predictions(predictor, features_df, feature_engineer)
        if pred_df is None:
            print("✗ Failed to generate predictions. Exiting.")
            return
    
    # Step 5: Create summary report
    if args.train_model:
        create_summary_report(features_df, pred_df)
    
    # Step 6: Launch dashboard
    if args.launch_dashboard:
        launch_dashboard()
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 Data: data/subscriber_data.csv")
    print("  🗄️  Database: subscriber_analytics.db")
    print("  🤖 Model: models/churn_model.pkl")
    print("  📈 Reports: reports/")
    print("  📤 Exports: exports/")
    print("\nNext steps:")
    print("  1. Review the generated reports")
    print("  2. Access the dashboard at http://127.0.0.1:8050")
    print("  3. Use exports/full_subscriber_analytics.csv for Power BI")

if __name__ == "__main__":
    main() 