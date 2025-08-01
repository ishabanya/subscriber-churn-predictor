import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sqlite3

class FeatureEngineer:
    def __init__(self, db_path='subscriber_analytics.db'):
        self.db_path = db_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        """
        Create comprehensive features for churn prediction.
        
        Args:
            df: Raw subscriber data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("Starting feature engineering...")
        
        # Convert date columns
        df['date'] = pd.to_datetime(df['date'])
        df['subscription_start'] = pd.to_datetime(df['subscription_start'])
        df['subscription_end'] = pd.to_datetime(df['subscription_end'])
        
        # Group by subscriber to create subscriber-level features
        subscriber_features = []
        
        for sub_id in df['subscriber_id'].unique():
            sub_data = df[df['subscriber_id'] == sub_id].copy()
            
            # Basic subscriber info
            latest_record = sub_data.iloc[-1]
            
            # Subscription duration features
            subscription_duration_days = (latest_record['date'] - latest_record['subscription_start']).days
            
            # Usage patterns
            avg_sessions = sub_data['sessions_count'].mean()
            total_sessions = sub_data['sessions_count'].sum()
            avg_session_duration = sub_data['avg_session_duration'].mean()
            
            # Revenue features
            avg_monthly_revenue = sub_data['monthly_revenue'].mean()
            total_revenue = sub_data['monthly_revenue'].sum()
            revenue_volatility = sub_data['monthly_revenue'].std()
            
            # Usage trend features
            if len(sub_data) > 1:
                sessions_trend = np.polyfit(range(len(sub_data)), sub_data['sessions_count'], 1)[0]
                revenue_trend = np.polyfit(range(len(sub_data)), sub_data['monthly_revenue'], 1)[0]
            else:
                sessions_trend = 0
                revenue_trend = 0
            
            # Engagement features
            months_active = len(sub_data)
            avg_features_used = sub_data['features_used'].mean()
            
            # Support and payment features
            total_support_tickets = sub_data['support_tickets'].sum()
            avg_support_tickets_per_month = total_support_tickets / months_active if months_active > 0 else 0
            late_payments = sub_data['late_payments'].iloc[0]  # Should be consistent per subscriber
            
            # Recency features (how recent was their last activity)
            days_since_last_activity = (datetime.now() - latest_record['date']).days
            
            # Plan features
            plan = latest_record['plan']
            plan_rank = {'Basic': 1, 'Premium': 2, 'Enterprise': 3}
            
            # Location features
            location = latest_record['location']
            
            # Age features
            age = latest_record['age']
            age_group = self._categorize_age(age)
            
            # Churn target
            churned = latest_record['churned']
            
            # Create feature dictionary
            features = {
                'subscriber_id': sub_id,
                'subscription_duration_days': subscription_duration_days,
                'avg_sessions_per_month': avg_sessions,
                'total_sessions': total_sessions,
                'avg_session_duration': avg_session_duration,
                'avg_monthly_revenue': avg_monthly_revenue,
                'total_revenue': total_revenue,
                'revenue_volatility': revenue_volatility,
                'sessions_trend': sessions_trend,
                'revenue_trend': revenue_trend,
                'months_active': months_active,
                'avg_features_used': avg_features_used,
                'total_support_tickets': total_support_tickets,
                'avg_support_tickets_per_month': avg_support_tickets_per_month,
                'late_payments': late_payments,
                'days_since_last_activity': days_since_last_activity,
                'plan': plan,
                'plan_rank': plan_rank[plan],
                'location': location,
                'age': age,
                'age_group': age_group,
                'payment_method': latest_record['payment_method'],
                'churned': churned
            }
            
            subscriber_features.append(features)
        
        features_df = pd.DataFrame(subscriber_features)
        
        # Create additional derived features
        features_df = self._create_derived_features(features_df)
        
        print(f"Feature engineering complete. Created {len(features_df)} subscriber records with {len(features_df.columns)} features.")
        return features_df
    
    def _categorize_age(self, age):
        """Categorize age into groups."""
        if age < 25:
            return '18-24'
        elif age < 35:
            return '25-34'
        elif age < 45:
            return '35-44'
        elif age < 55:
            return '45-54'
        else:
            return '55+'
    
    def _create_derived_features(self, df):
        """Create additional derived features."""
        
        # Engagement score (combination of usage metrics)
        df['engagement_score'] = (
            df['avg_sessions_per_month'] * 0.3 +
            df['avg_session_duration'] * 0.2 +
            df['avg_features_used'] * 0.3 +
            (1 / (1 + df['avg_support_tickets_per_month'])) * 0.2
        )
        
        # Revenue per session
        df['revenue_per_session'] = df['total_revenue'] / df['total_sessions'].replace(0, 1)
        
        # Usage frequency (sessions per day of subscription)
        df['usage_frequency'] = df['total_sessions'] / df['subscription_duration_days'].replace(0, 1)
        
        # Support intensity
        df['support_intensity'] = df['total_support_tickets'] / df['months_active'].replace(0, 1)
        
        # Risk score (factors that indicate churn risk)
        df['risk_score'] = (
            df['late_payments'] * 0.3 +
            df['support_intensity'] * 0.3 +
            (df['revenue_volatility'] / df['avg_monthly_revenue'].replace(0, 1)) * 0.2 +
            (df['sessions_trend'] < 0).astype(int) * 0.2
        )
        
        # Value score (combination of revenue and engagement)
        df['value_score'] = (
            df['avg_monthly_revenue'] * 0.4 +
            df['engagement_score'] * 0.3 +
            df['plan_rank'] * 0.3
        )
        
        return df
    
    def prepare_model_data(self, features_df):
        """
        Prepare data for machine learning model training.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
        """
        print("Preparing data for model training...")
        
        # Select features for modeling
        feature_columns = [
            'subscription_duration_days', 'avg_sessions_per_month', 'total_sessions',
            'avg_session_duration', 'avg_monthly_revenue', 'total_revenue',
            'revenue_volatility', 'sessions_trend', 'revenue_trend', 'months_active',
            'avg_features_used', 'total_support_tickets', 'avg_support_tickets_per_month',
            'late_payments', 'days_since_last_activity', 'plan_rank',
            'engagement_score', 'revenue_per_session', 'usage_frequency',
            'support_intensity', 'risk_score', 'value_score'
        ]
        
        # Categorical features to encode
        categorical_features = ['plan', 'location', 'age_group', 'payment_method']
        
        # Create feature matrix
        X = features_df[feature_columns].copy()
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in features_df.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(features_df[feature].fillna('Unknown'))
                self.label_encoders[feature] = le
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Target variable
        y = features_df['churned'].astype(int)
        
        feature_names = feature_columns + categorical_features
        
        print(f"Model data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")
        
        return X_scaled, y, feature_names
    
    def save_to_database(self, features_df, table_name='subscriber_features'):
        """Save engineered features to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        features_df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Features saved to database table: {table_name}")
    
    def load_from_database(self, table_name='subscriber_features'):
        """Load engineered features from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        features_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return features_df 