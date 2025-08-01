# Subscriber Churn Prediction Pipeline

A comprehensive analytics pipeline for predicting subscriber churn using machine learning and interactive dashboards.

## Overview

This project implements an end-to-end solution for subscriber churn prediction, featuring:

- Data generation and preprocessing
- Feature engineering with 25+ derived features
- Multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
- Interactive dashboard with Plotly/Dash
- SQLite database for data storage
- CSV exports for BI tools

## Features

### Data Processing
- Realistic subscriber data generation
- Comprehensive feature engineering
- Data validation and cleaning

### Machine Learning
- Multiple algorithm comparison
- Cross-validation and performance metrics
- Feature importance analysis
- Model persistence

### Visualization
- Interactive dashboard with filters
- Real-time metrics and charts
- Risk assessment visualizations
- Export capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ishabanya/subscriber-churn-predictor.git
cd subscriber-churn-predictor
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
Run the complete pipeline:
```bash
python main_pipeline.py
```

This will:
- Generate sample subscriber data
- Create engineered features
- Train ML models
- Generate predictions
- Launch the dashboard

### Custom Configuration
```bash
# Generate data only
python main_pipeline.py --generate-data

# Train model with custom parameters
python main_pipeline.py --subscribers 1000 --days 365

# Launch dashboard only
python main_pipeline.py --launch-dashboard
```

### Dashboard Access
After running the pipeline, access the dashboard at:
```
http://127.0.0.1:8050
```

## Project Structure

```
subscriber-churn-predictor/
├── data_generator.py          # Sample data generation
├── feature_engineering.py     # Feature engineering pipeline
├── churn_model.py            # ML model training
├── dashboard.py              # Interactive dashboard
├── main_pipeline.py          # Main orchestration
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── data/                     # Generated data
├── models/                   # Trained models
├── reports/                  # Analysis reports
└── exports/                  # BI-ready exports
```

## Model Performance

The pipeline evaluates multiple algorithms:

- **Random Forest**: AUC 0.737 (Best performing)
- **Gradient Boosting**: AUC 0.684
- **Logistic Regression**: AUC 0.645

## Generated Features

The pipeline creates 25+ engineered features including:

- Usage patterns (sessions, duration, frequency)
- Revenue metrics (volatility, trends, per-session)
- Engagement scores (composite metrics)
- Risk assessments (support, payments, trends)
- Value scoring (revenue + engagement)

## Dashboard Features

- Key metrics cards (subscribers, churn rate, revenue, risk)
- Interactive filters (plan, location, risk threshold)
- Churn distribution visualizations
- Risk score analysis
- Engagement vs risk scatter plots
- Revenue analysis by plan
- Support ticket correlations
- High-risk subscriber table

## Data Export

The pipeline generates several export files:

- `exports/full_subscriber_analytics.csv` - Complete dataset for BI tools
- `exports/high_risk_subscribers.csv` - Subscribers with high churn probability
- `reports/model_report.txt` - Detailed model performance
- `reports/summary_report.txt` - Business summary

## Customization

### Using Your Own Data
1. Replace `data/subscriber_data.csv` with your data
2. Ensure required columns are present
3. Run the pipeline with `--train-model --launch-dashboard`

### Model Tuning
- Modify hyperparameters in `churn_model.py`
- Add new algorithms to the pipeline
- Adjust feature selection as needed

### Dashboard Customization
- Add new visualizations in `dashboard.py`
- Modify filters and metrics
- Customize styling and layout

## Requirements

- Python 3.8+
- pandas >= 2.2.0
- scikit-learn >= 1.4.0
- plotly >= 5.18.0
- dash >= 2.16.0
- numpy >= 1.26.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions or issues:
1. Check the troubleshooting section in the code
2. Review the generated reports
3. Open an issue on GitHub 