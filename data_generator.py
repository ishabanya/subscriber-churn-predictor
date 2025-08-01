import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_subscriber_data(n_subscribers=1000, days_history=365):
    """
    Generate realistic subscriber data for churn prediction analysis.
    
    Args:
        n_subscribers: Number of subscribers to generate
        days_history: Number of days of historical data
    
    Returns:
        DataFrame with subscriber data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate base subscriber data
    subscriber_ids = [f"SUB_{i:06d}" for i in range(1, n_subscribers + 1)]
    
    # Subscription plans
    plans = ['Basic', 'Premium', 'Enterprise']
    plan_weights = [0.5, 0.35, 0.15]
    
    # Generate subscription data
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_history)
    
    for sub_id in subscriber_ids:
        # Random subscription start date
        sub_start = start_date + timedelta(days=random.randint(0, days_history - 30))
        
        # Plan assignment
        plan = np.random.choice(plans, p=plan_weights)
        
        # Monthly revenue based on plan
        plan_revenue = {'Basic': 29.99, 'Premium': 79.99, 'Enterprise': 199.99}
        monthly_revenue = plan_revenue[plan]
        
        # Generate churn probability factors
        age = random.randint(18, 75)
        location = random.choice(['US', 'EU', 'Asia', 'Other'])
        
        # Usage patterns
        avg_sessions_per_month = random.randint(5, 50)
        avg_session_duration = random.randint(10, 120)
        
        # Support interactions
        support_tickets = random.randint(0, 10)
        
        # Payment history
        payment_method = random.choice(['Credit Card', 'PayPal', 'Bank Transfer'])
        late_payments = random.randint(0, 3)
        
        # Feature usage
        features_used = random.randint(1, 8)
        
        # Calculate churn probability based on features
        churn_prob = 0.1  # Base churn probability
        
        # Factors that increase churn
        if late_payments > 1:
            churn_prob += 0.2
        if support_tickets > 5:
            churn_prob += 0.15
        if avg_sessions_per_month < 10:
            churn_prob += 0.1
        if plan == 'Basic':
            churn_prob += 0.05
        
        # Factors that decrease churn
        if plan == 'Enterprise':
            churn_prob -= 0.1
        if features_used > 5:
            churn_prob -= 0.1
        if avg_session_duration > 60:
            churn_prob -= 0.05
        
        churn_prob = max(0.01, min(0.8, churn_prob))
        
        # Determine if subscriber churned
        churned = np.random.random() < churn_prob
        
        # Calculate subscription end date
        if churned:
            sub_end = sub_start + timedelta(days=random.randint(30, 365))
        else:
            sub_end = None
        
        # Generate monthly data points
        current_date = sub_start
        while current_date <= end_date and (sub_end is None or current_date <= sub_end):
            # Add some variability to monthly metrics
            sessions_this_month = max(0, int(avg_sessions_per_month * (0.5 + np.random.random())))
            session_duration_this_month = max(5, int(avg_session_duration * (0.7 + 0.6 * np.random.random())))
            
            # Revenue with some variation
            revenue_this_month = monthly_revenue * (0.9 + 0.2 * np.random.random())
            
            data.append({
                'subscriber_id': sub_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'plan': plan,
                'monthly_revenue': round(revenue_this_month, 2),
                'sessions_count': sessions_this_month,
                'avg_session_duration': session_duration_this_month,
                'support_tickets': support_tickets,
                'payment_method': payment_method,
                'late_payments': late_payments,
                'features_used': features_used,
                'age': age,
                'location': location,
                'churned': churned,
                'subscription_start': sub_start.strftime('%Y-%m-%d'),
                'subscription_end': sub_end.strftime('%Y-%m-%d') if sub_end else None
            })
            
            current_date += timedelta(days=30)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating subscriber data...")
    df = generate_subscriber_data(n_subscribers=1000, days_history=365)
    
    # Save to CSV
    df.to_csv('subscriber_data.csv', index=False)
    print(f"Generated {len(df)} records for {df['subscriber_id'].nunique()} subscribers")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print("Data saved to subscriber_data.csv") 