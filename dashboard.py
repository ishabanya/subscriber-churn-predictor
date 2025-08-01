import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Subscriber Churn Analytics Dashboard"

class DashboardData:
    def __init__(self, db_path='subscriber_analytics.db'):
        self.db_path = db_path
        
    def load_data(self):
        """Load data from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load subscriber features
            features_df = pd.read_sql("SELECT * FROM subscriber_features", conn)
            
            # Load predictions if available
            try:
                predictions_df = pd.read_sql("SELECT * FROM churn_predictions", conn)
                # Merge predictions with features
                features_df = features_df.merge(predictions_df, on='subscriber_id', how='left')
            except:
                print("No predictions table found")
            
            conn.close()
            return features_df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

# Initialize data loader
data_loader = DashboardData()

# Dashboard layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("📊 Subscriber Churn Analytics Dashboard", 
                   className="text-center mb-4 text-primary"),
            html.Hr()
        ])
    ]),
    
    # Key Metrics Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="total-subscribers", className="card-title"),
                    html.P("Total Subscribers", className="card-text")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="churn-rate", className="card-title"),
                    html.P("Churn Rate", className="card-text")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="avg-revenue", className="card-title"),
                    html.P("Avg Monthly Revenue", className="card-text")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="high-risk", className="card-title"),
                    html.P("High Risk Subscribers", className="card-text")
                ])
            ], className="text-center")
        ], width=3)
    ], className="mb-4"),
    
    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Plan Filter:"),
            dcc.Dropdown(
                id="plan-filter",
                options=[
                    {"label": "All Plans", "value": "All"},
                    {"label": "Basic", "value": "Basic"},
                    {"label": "Premium", "value": "Premium"},
                    {"label": "Enterprise", "value": "Enterprise"}
                ],
                value="All",
                clearable=False
            )
        ], width=3),
        dbc.Col([
            html.Label("Location Filter:"),
            dcc.Dropdown(
                id="location-filter",
                options=[
                    {"label": "All Locations", "value": "All"},
                    {"label": "US", "value": "US"},
                    {"label": "EU", "value": "EU"},
                    {"label": "Asia", "value": "Asia"},
                    {"label": "Other", "value": "Other"}
                ],
                value="All",
                clearable=False
            )
        ], width=3),
        dbc.Col([
            html.Label("Risk Threshold:"),
            dcc.Slider(
                id="risk-threshold",
                min=0,
                max=1,
                step=0.1,
                value=0.5,
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=6)
    ], className="mb-4"),
    
    # Main Charts
    dbc.Row([
        # Churn Distribution
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Churn Distribution by Plan"),
                dbc.CardBody([
                    dcc.Graph(id="churn-distribution")
                ])
            ])
        ], width=6),
        
        # Risk Score Distribution
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Score Distribution"),
                dbc.CardBody([
                    dcc.Graph(id="risk-distribution")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Engagement Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Engagement vs Churn Risk"),
                dbc.CardBody([
                    dcc.Graph(id="engagement-scatter")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Revenue Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Revenue Analysis by Plan"),
                dbc.CardBody([
                    dcc.Graph(id="revenue-analysis")
                ])
            ])
        ], width=6),
        
        # Support Tickets Analysis
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Support Tickets vs Churn"),
                dbc.CardBody([
                    dcc.Graph(id="support-analysis")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Feature Importance
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Risk Factors"),
                dbc.CardBody([
                    dcc.Graph(id="risk-factors")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Detailed Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("High-Risk Subscribers"),
                dbc.CardBody([
                    html.Div(id="risk-table")
                ])
            ])
        ], width=12)
    ])
    
], fluid=True)

# Callbacks
@app.callback(
    [Output("total-subscribers", "children"),
     Output("churn-rate", "children"),
     Output("avg-revenue", "children"),
     Output("high-risk", "children")],
    [Input("plan-filter", "value"),
     Input("location-filter", "value"),
     Input("risk-threshold", "value")]
)
def update_metrics(plan_filter, location_filter, risk_threshold):
    df = data_loader.load_data()
    if df.empty:
        return "N/A", "N/A", "N/A", "N/A"
    
    # Apply filters
    if plan_filter != "All":
        df = df[df['plan'] == plan_filter]
    if location_filter != "All":
        df = df[df['location'] == location_filter]
    
    total_subscribers = len(df)
    churn_rate = f"{df['churned'].mean():.1%}" if 'churned' in df.columns else "N/A"
    avg_revenue = f"${df['avg_monthly_revenue'].mean():.0f}" if 'avg_monthly_revenue' in df.columns else "N/A"
    
    # High risk subscribers (using churn probability if available, otherwise risk score)
    if 'churn_probability' in df.columns:
        high_risk = len(df[df['churn_probability'] >= risk_threshold])
    elif 'risk_score' in df.columns:
        high_risk = len(df[df['risk_score'] >= risk_threshold])
    else:
        high_risk = "N/A"
    
    return total_subscribers, churn_rate, avg_revenue, high_risk

@app.callback(
    Output("churn-distribution", "figure"),
    [Input("plan-filter", "value"),
     Input("location-filter", "value")]
)
def update_churn_distribution(plan_filter, location_filter):
    df = data_loader.load_data()
    if df.empty or 'churned' not in df.columns:
        return go.Figure()
    
    # Apply filters
    if plan_filter != "All":
        df = df[df['plan'] == plan_filter]
    if location_filter != "All":
        df = df[df['location'] == location_filter]
    
    # Create churn distribution by plan
    churn_by_plan = df.groupby(['plan', 'churned']).size().unstack(fill_value=0)
    
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
        title="Churn Distribution by Subscription Plan",
        xaxis_title="Plan",
        yaxis_title="Number of Subscribers",
        barmode='group',
        height=400
    )
    
    return fig

@app.callback(
    Output("risk-distribution", "figure"),
    [Input("plan-filter", "value"),
     Input("location-filter", "value")]
)
def update_risk_distribution(plan_filter, location_filter):
    df = data_loader.load_data()
    if df.empty:
        return go.Figure()
    
    # Apply filters
    if plan_filter != "All":
        df = df[df['plan'] == plan_filter]
    if location_filter != "All":
        df = df[df['location'] == location_filter]
    
    # Use churn probability if available, otherwise risk score
    if 'churn_probability' in df.columns:
        risk_col = 'churn_probability'
        title = "Churn Probability Distribution"
    elif 'risk_score' in df.columns:
        risk_col = 'risk_score'
        title = "Risk Score Distribution"
    else:
        return go.Figure()
    
    fig = px.histogram(
        df, 
        x=risk_col,
        nbins=30,
        color_discrete_sequence=['lightblue'],
        opacity=0.7
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Risk Score",
        yaxis_title="Number of Subscribers",
        height=400
    )
    
    return fig

@app.callback(
    Output("engagement-scatter", "figure"),
    [Input("plan-filter", "value"),
     Input("location-filter", "value")]
)
def update_engagement_scatter(plan_filter, location_filter):
    df = data_loader.load_data()
    if df.empty:
        return go.Figure()
    
    # Apply filters
    if plan_filter != "All":
        df = df[df['plan'] == plan_filter]
    if location_filter != "All":
        df = df[df['location'] == location_filter]
    
    # Check if required columns exist
    required_cols = ['engagement_score', 'risk_score']
    if not all(col in df.columns for col in required_cols):
        return go.Figure()
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='engagement_score',
        y='risk_score',
        color='plan',
        size='avg_monthly_revenue',
        hover_data=['subscriber_id', 'avg_sessions_per_month'],
        title="Engagement Score vs Risk Score"
    )
    
    fig.update_layout(height=500)
    
    return fig

@app.callback(
    Output("revenue-analysis", "figure"),
    [Input("plan-filter", "value"),
     Input("location-filter", "value")]
)
def update_revenue_analysis(plan_filter, location_filter):
    df = data_loader.load_data()
    if df.empty or 'avg_monthly_revenue' not in df.columns:
        return go.Figure()
    
    # Apply filters
    if plan_filter != "All":
        df = df[df['plan'] == plan_filter]
    if location_filter != "All":
        df = df[df['location'] == location_filter]
    
    # Revenue analysis by plan
    revenue_stats = df.groupby('plan')['avg_monthly_revenue'].agg(['mean', 'std', 'count']).reset_index()
    
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
        xaxis_title="Plan",
        yaxis_title="Average Monthly Revenue ($)",
        height=400
    )
    
    return fig

@app.callback(
    Output("support-analysis", "figure"),
    [Input("plan-filter", "value"),
     Input("location-filter", "value")]
)
def update_support_analysis(plan_filter, location_filter):
    df = data_loader.load_data()
    if df.empty or 'total_support_tickets' not in df.columns:
        return go.Figure()
    
    # Apply filters
    if plan_filter != "All":
        df = df[df['plan'] == plan_filter]
    if location_filter != "All":
        df = df[df['location'] == location_filter]
    
    # Support tickets vs churn
    if 'churned' in df.columns:
        support_by_churn = df.groupby('churned')['total_support_tickets'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Retained', 'Churned'],
            y=[support_by_churn[0], support_by_churn[1]],
            marker_color=['lightgreen', 'lightcoral']
        ))
        
        fig.update_layout(
            title="Average Support Tickets by Churn Status",
            xaxis_title="Churn Status",
            yaxis_title="Average Support Tickets",
            height=400
        )
    else:
        fig = px.histogram(
            df,
            x='total_support_tickets',
            nbins=20,
            title="Support Tickets Distribution"
        )
        fig.update_layout(height=400)
    
    return fig

@app.callback(
    Output("risk-factors", "figure"),
    [Input("plan-filter", "value"),
     Input("location-filter", "value")]
)
def update_risk_factors(plan_filter, location_filter):
    df = data_loader.load_data()
    if df.empty:
        return go.Figure()
    
    # Apply filters
    if plan_filter != "All":
        df = df[df['plan'] == plan_filter]
    if location_filter != "All":
        df = df[df['location'] == location_filter]
    
    # Calculate correlation with churn if available
    if 'churned' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'churned']
        
        correlations = []
        for col in numeric_cols:
            corr = df[col].corr(df['churned'])
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
            return fig
    
    # Fallback: show risk score distribution by plan
    if 'risk_score' in df.columns:
        fig = px.box(
            df,
            x='plan',
            y='risk_score',
            title="Risk Score Distribution by Plan"
        )
        fig.update_layout(height=500)
        return fig
    
    return go.Figure()

@app.callback(
    Output("risk-table", "children"),
    [Input("plan-filter", "value"),
     Input("location-filter", "value"),
     Input("risk-threshold", "value")]
)
def update_risk_table(plan_filter, location_filter, risk_threshold):
    df = data_loader.load_data()
    if df.empty:
        return "No data available"
    
    # Apply filters
    if plan_filter != "All":
        df = df[df['plan'] == plan_filter]
    if location_filter != "All":
        df = df[df['location'] == location_filter]
    
    # Identify high-risk subscribers
    if 'churn_probability' in df.columns:
        high_risk_df = df[df['churn_probability'] >= risk_threshold]
        risk_col = 'churn_probability'
    elif 'risk_score' in df.columns:
        high_risk_df = df[df['risk_score'] >= risk_threshold]
        risk_col = 'risk_score'
    else:
        return "No risk data available"
    
    if high_risk_df.empty:
        return "No high-risk subscribers found"
    
    # Select columns for display
    display_cols = ['subscriber_id', 'plan', 'avg_monthly_revenue', 'engagement_score']
    if risk_col in high_risk_df.columns:
        display_cols.append(risk_col)
    
    display_df = high_risk_df[display_cols].head(20)
    
    # Create table
    table = dbc.Table.from_dataframe(
        display_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True
    )
    
    return table

if __name__ == '__main__':
    print("Starting Churn Analytics Dashboard...")
    print("Access the dashboard at: http://127.0.0.1:8050")
    app.run(debug=True, host='0.0.0.0', port=8050) 