import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Subscriber Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
    def __init__(self, db_path='subscriber_analytics.db'):
        self.db_path = db_path
        
    def load_data(self):
        """Load data from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if tables exist
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='subscriber_features'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                st.info("No data found. Generating sample data...")
                conn.close()
                return self.generate_sample_data()
            
            # Load subscriber features
            features_df = pd.read_sql("SELECT * FROM subscriber_features", conn)
            
            # Load predictions if available
            try:
                predictions_df = pd.read_sql("SELECT * FROM churn_predictions", conn)
                # Merge predictions with features
                features_df = features_df.merge(predictions_df, on='subscriber_id', how='left')
            except:
                st.warning("No predictions found. Generating predictions...")
                features_df = self.generate_predictions(features_df)
            
            conn.close()
            return features_df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def generate_sample_data(self):
        """Generate sample data if none exists."""
        try:
            from data_generator import generate_subscriber_data
            from feature_engineering import FeatureEngineer
            from churn_model import ChurnPredictor
            
            # Generate sample data
            with st.spinner("Generating sample subscriber data..."):
                df = generate_subscriber_data(n_subscribers=500, days_history=180)
            
            # Feature engineering
            with st.spinner("Creating features..."):
                feature_engineer = FeatureEngineer()
                features_df = feature_engineer.create_features(df)
                feature_engineer.save_to_database(features_df)
            
            # Generate predictions
            features_df = self.generate_predictions(features_df)
            
            st.success("Sample data generated successfully!")
            return features_df
            
        except Exception as e:
            st.error(f"Error generating sample data: {e}")
            return pd.DataFrame()
    
    def generate_predictions(self, features_df):
        """Generate predictions for the features."""
        try:
            from feature_engineering import FeatureEngineer
            from churn_model import ChurnPredictor
            
            # Prepare model data
            feature_engineer = FeatureEngineer()
            X, y, feature_names = feature_engineer.prepare_model_data(features_df)
            
            # Train models and generate predictions
            predictor = ChurnPredictor()
            best_model = predictor.train_models(X, y, feature_names)
            predictions, probabilities = predictor.predict_churn(X)
            
            # Save predictions to database
            predictor.save_predictions_to_db(
                features_df['subscriber_id'], 
                predictions, 
                probabilities
            )
            
            # Merge predictions with features
            predictions_df = pd.DataFrame({
                'subscriber_id': features_df['subscriber_id'],
                'churn_probability': probabilities
            })
            
            return features_df.merge(predictions_df, on='subscriber_id', how='left')
            
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            return features_df

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cached_data():
    """Load data with caching to avoid regenerating on every refresh."""
    dashboard = StreamlitDashboard()
    return dashboard.load_data()

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Subscriber Churn Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data with caching
    df = load_cached_data()
    
    if df.empty:
        st.error("No data available. Please run the pipeline first to generate data.")
        st.info("Run: `python main_pipeline.py` to generate sample data and train models.")
        return
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", help="Regenerate sample data and predictions"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")
    
    # Plan filter
    plan_filter = st.sidebar.selectbox(
        "Plan Filter:",
        ["All"] + list(df['plan'].unique()),
        index=0
    )
    
    # Location filter
    location_filter = st.sidebar.selectbox(
        "Location Filter:",
        ["All"] + list(df['location'].unique()),
        index=0
    )
    
    # Risk threshold
    risk_threshold = st.sidebar.slider(
        "Risk Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Threshold for high-risk classification"
    )
    
    # Apply filters
    filtered_df = df.copy()
    if plan_filter != "All":
        filtered_df = filtered_df[filtered_df['plan'] == plan_filter]
    if location_filter != "All":
        filtered_df = filtered_df[filtered_df['location'] == location_filter]
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Subscribers",
            value=f"{len(filtered_df):,}",
            delta=None
        )
    
    with col2:
        churn_rate = filtered_df['churned'].mean() if 'churned' in filtered_df.columns else 0
        st.metric(
            label="Churn Rate",
            value=f"{churn_rate:.1%}",
            delta=None
        )
    
    with col3:
        avg_revenue = filtered_df['avg_monthly_revenue'].mean()
        st.metric(
            label="Avg Monthly Revenue",
            value=f"${avg_revenue:.0f}",
            delta=None
        )
    
    with col4:
        if 'churn_probability' in filtered_df.columns:
            high_risk = len(filtered_df[filtered_df['churn_probability'] >= risk_threshold])
        elif 'risk_score' in filtered_df.columns:
            high_risk = len(filtered_df[filtered_df['risk_score'] >= risk_threshold])
        else:
            high_risk = 0
        st.metric(
            label="High Risk Subscribers",
            value=f"{high_risk:,}",
            delta=None
        )
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution by Plan")
        if 'churned' in filtered_df.columns:
            churn_by_plan = filtered_df.groupby(['plan', 'churned']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            
            if 0 in churn_by_plan.columns:
                fig.add_trace(go.Bar(
                    x=churn_by_plan.index,
                    y=churn_by_plan[0],
                    name='Retained',
                    marker_color='lightgreen'
                ))
            
            if 1 in churn_by_plan.columns:
                fig.add_trace(go.Bar(
                    x=churn_by_plan.index,
                    y=churn_by_plan[1],
                    name='Churned',
                    marker_color='lightcoral'
                ))
            
            fig.update_layout(
                barmode='group',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No churn data available")
    
    with col2:
        st.subheader("Risk Score Distribution")
        if 'churn_probability' in filtered_df.columns:
            risk_col = 'churn_probability'
            title = "Churn Probability Distribution"
        elif 'risk_score' in filtered_df.columns:
            risk_col = 'risk_score'
            title = "Risk Score Distribution"
        else:
            st.info("No risk data available")
            return
        
        fig = px.histogram(
            filtered_df, 
            x=risk_col,
            nbins=30,
            color_discrete_sequence=['lightblue'],
            opacity=0.7
        )
        
        fig.update_layout(
            title=title,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue Analysis by Plan")
        revenue_stats = filtered_df.groupby('plan')['avg_monthly_revenue'].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=revenue_stats['plan'],
            y=revenue_stats['mean'],
            error_y=dict(type='data', array=revenue_stats['std']),
            name='Average Revenue',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Average Monthly Revenue by Plan",
            yaxis_title="Average Monthly Revenue ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Support Tickets vs Churn")
        if 'churned' in filtered_df.columns and 'total_support_tickets' in filtered_df.columns:
            support_by_churn = filtered_df.groupby('churned')['total_support_tickets'].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Retained', 'Churned'],
                y=[support_by_churn[0], support_by_churn[1]],
                marker_color=['lightgreen', 'lightcoral']
            ))
            
            fig.update_layout(
                title="Average Support Tickets by Churn Status",
                yaxis_title="Average Support Tickets",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Support data not available")
    
    # Engagement vs Risk Scatter
    st.subheader("Engagement vs Risk Analysis")
    if 'engagement_score' in filtered_df.columns and 'risk_score' in filtered_df.columns:
        fig = px.scatter(
            filtered_df,
            x='engagement_score',
            y='risk_score',
            color='plan',
            size='avg_monthly_revenue',
            hover_data=['subscriber_id', 'avg_sessions_per_month'],
            title="Engagement Score vs Risk Score"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Engagement data not available")
    
    # Risk Factors
    st.subheader("Top Risk Factors")
    if 'churned' in filtered_df.columns:
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'churned']
        
        correlations = []
        for col in numeric_cols:
            corr = filtered_df[col].corr(filtered_df['churned'])
            if not pd.isna(corr):
                correlations.append({'feature': col, 'correlation': abs(corr)})
        
        if correlations:
            corr_df = pd.DataFrame(correlations)
            corr_df = corr_df.sort_values('correlation', ascending=False).head(10)
            
            fig = px.bar(
                corr_df,
                x='correlation',
                y='feature',
                orientation='h',
                title="Top Risk Factors (Correlation with Churn)"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Churn data not available for correlation analysis")
    
    # High-risk subscribers table
    st.subheader("High-Risk Subscribers")
    if 'churn_probability' in filtered_df.columns:
        high_risk_df = filtered_df[filtered_df['churn_probability'] >= risk_threshold]
        risk_col = 'churn_probability'
    elif 'risk_score' in filtered_df.columns:
        high_risk_df = filtered_df[filtered_df['risk_score'] >= risk_threshold]
        risk_col = 'risk_score'
    else:
        st.info("No risk data available")
        return
    
    if high_risk_df.empty:
        st.info("No high-risk subscribers found")
    else:
        # Select columns for display
        display_cols = ['subscriber_id', 'plan', 'avg_monthly_revenue', 'engagement_score']
        if risk_col in high_risk_df.columns:
            display_cols.append(risk_col)
        
        display_df = high_risk_df[display_cols].head(20)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = high_risk_df.to_csv(index=False)
        st.download_button(
            label="Download High-Risk Subscribers CSV",
            data=csv,
            file_name=f"high_risk_subscribers_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main() 