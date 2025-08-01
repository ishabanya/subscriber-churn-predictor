# 📊 Subscriber Churn Prediction Pipeline

A comprehensive machine learning pipeline for predicting subscriber churn with an interactive analytics dashboard. Built with Python, featuring advanced feature engineering, multiple ML models, and real-time visualizations.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue?logo=streamlit)](https://subscriber-churn-predictor.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Live Demo

**🌐 [Access the Interactive Dashboard](https://subscriber-churn-predictor.streamlit.app/)**

Experience the full analytics dashboard with real-time data generation, interactive visualizations, and churn predictions.

## ✨ Features

### 🔍 **Advanced Analytics**
- **Real-time Churn Prediction**: ML models with 78% accuracy
- **Feature Engineering**: 29 engineered features from raw subscriber data
- **Risk Scoring**: Identify high-risk subscribers automatically
- **Performance Metrics**: AUC, accuracy, and cross-validation scores

### 📈 **Interactive Dashboard**
- **Live Visualizations**: Churn distribution, revenue analysis, risk factors
- **Dynamic Filtering**: Filter by plan, location, and risk threshold
- **Real-time Metrics**: Key performance indicators and trends
- **Data Export**: Download high-risk subscribers as CSV

### 🤖 **Machine Learning Models**
- **Random Forest**: Best performing model (AUC: 0.737)
- **Gradient Boosting**: Alternative ensemble method
- **Logistic Regression**: Interpretable baseline model
- **Cross-validation**: Robust performance evaluation

### 🛠️ **Technical Stack**
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Interactive web application
- **Plotly**: Interactive visualizations
- **SQLite**: Lightweight database storage

## 🏗️ Project Structure

```
subscriber-churn-predictor/
├── 📁 data/                    # Generated sample data
├── 📁 models/                  # Trained ML models
├── 📁 reports/                 # Performance reports
├── 📁 exports/                 # Data exports for BI tools
├── 📄 streamlit_app.py         # Main Streamlit dashboard
├── 📄 main_pipeline.py         # End-to-end pipeline orchestration
├── 📄 data_generator.py        # Synthetic data generation
├── 📄 feature_engineering.py   # Feature creation and preprocessing
├── 📄 churn_model.py          # ML model training and evaluation
├── 📄 setup_streamlit.py      # Streamlit deployment setup
├── 📄 requirements.txt         # Python dependencies
└── 📄 requirements-streamlit.txt # Streamlit-specific dependencies
```

## 🚀 Quick Start

### Option 1: Use the Live Dashboard
1. Visit [https://subscriber-churn-predictor.streamlit.app/](https://subscriber-churn-predictor.streamlit.app/)
2. The app will automatically generate sample data
3. Explore the interactive visualizations and analytics

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/ishabanya/subscriber-churn-predictor.git
cd subscriber-churn-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-streamlit.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### Option 3: Full Pipeline

```bash
# Run the complete pipeline
python main_pipeline.py --subscribers 1000 --days 365

# Or run individual components
python setup_streamlit.py  # Generate sample data
python main_pipeline.py --train-model  # Train models only
```

## 📊 Dashboard Features

### **Key Metrics**
- Total subscribers and churn rate
- Average monthly revenue
- High-risk subscriber count
- Model performance indicators

### **Interactive Visualizations**
- **Churn Distribution**: By subscription plan
- **Risk Score Analysis**: Probability distribution
- **Revenue Analysis**: By plan with error bars
- **Support vs Churn**: Correlation analysis
- **Engagement vs Risk**: Scatter plot analysis
- **Top Risk Factors**: Feature importance ranking

### **Data Controls**
- **Plan Filter**: Filter by subscription tier
- **Location Filter**: Geographic segmentation
- **Risk Threshold**: Adjustable risk classification
- **Refresh Data**: Regenerate sample data
- **Export Functionality**: Download high-risk subscribers

## 🔧 Model Performance

| Model | Accuracy | AUC Score | Cross-Validation AUC |
|-------|----------|-----------|---------------------|
| **Random Forest** | 78.0% | 0.737 | 0.721 ± 0.099 |
| Gradient Boosting | 79.0% | 0.684 | 0.712 ± 0.027 |
| Logistic Regression | 78.0% | 0.645 | 0.689 ± 0.106 |

## 📈 Generated Features

The pipeline creates 29 engineered features including:

### **Behavioral Features**
- Engagement score and session frequency
- Feature usage patterns
- Support ticket history
- Payment behavior metrics

### **Temporal Features**
- Subscription duration
- Recent activity patterns
- Seasonal trends
- Usage volatility

### **Business Features**
- Revenue metrics and trends
- Plan-specific indicators
- Geographic factors
- Risk scoring algorithms

## 🚀 Deployment Options

### **Streamlit Cloud** (Recommended)
- Free hosting with automatic deployment
- GitHub integration
- Custom domain support
- Built-in caching and performance optimization

### **Heroku**
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### **Railway**
- Connect GitHub repository
- Automatic deployment on push
- Built-in environment management

### **Render**
- Web service deployment
- Automatic HTTPS
- Custom domain support

## 📋 Requirements

- **Python**: 3.8 or higher
- **Memory**: 2GB RAM minimum
- **Storage**: 100MB for sample data and models
- **Browser**: Modern web browser for dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **Scikit-learn** for robust machine learning algorithms
- **Plotly** for beautiful interactive visualizations
- **Pandas** for powerful data manipulation

## 📞 Support

- **Live Demo**: [https://subscriber-churn-predictor.streamlit.app/](https://subscriber-churn-predictor.streamlit.app/)
- **GitHub Issues**: [Report bugs or request features](https://github.com/ishabanya/subscriber-churn-predictor/issues)
- **Documentation**: Check the code comments and docstrings

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Ishabanya](https://github.com/ishabanya)

</div> 