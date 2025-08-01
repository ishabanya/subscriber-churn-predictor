import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime

class ChurnPredictor:
    def __init__(self, db_path='subscriber_analytics.db'):
        self.db_path = db_path
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.model_performance = {}
        
    def train_models(self, X, y, feature_names):
        """
        Train multiple churn prediction models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
        """
        print("Training churn prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to try
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            
            # Store results
            self.model_performance[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_test': y_test
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
            print(f"Cross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Select best model based on AUC score
        best_model_name = max(self.model_performance.keys(), 
                            key=lambda x: self.model_performance[x]['auc_score'])
        self.best_model = self.model_performance[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best AUC score: {self.model_performance[best_model_name]['auc_score']:.3f}")
        
        # Get feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.best_model
    
    def hyperparameter_tuning(self, X, y, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for the best model.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name of the model to tune
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingClassifier(random_state=42)
        
        else:
            print("Hyperparameter tuning not implemented for this model.")
            return
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        
        return self.best_model
    
    def predict_churn(self, X, threshold=0.5):
        """
        Make churn predictions using the best model.
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for churn prediction
            
        Returns:
            predictions: Binary predictions
            probabilities: Churn probabilities
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        probabilities = self.best_model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def save_model(self, filepath='churn_model.pkl'):
        """Save the trained model to disk."""
        if self.best_model is not None:
            joblib.dump(self.best_model, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No trained model to save.")
    
    def load_model(self, filepath='churn_model.pkl'):
        """Load a trained model from disk."""
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def save_predictions_to_db(self, subscriber_ids, predictions, probabilities, table_name='churn_predictions'):
        """Save predictions to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'subscriber_id': subscriber_ids,
            'churn_probability': probabilities,
            'churn_prediction': predictions,
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Save to database
        pred_df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"Predictions saved to database table: {table_name}")
        return pred_df
    
    def generate_model_report(self, output_file='model_report.txt'):
        """Generate a comprehensive model performance report."""
        if not self.model_performance:
            print("No model performance data available.")
            return
        
        with open(output_file, 'w') as f:
            f.write("CHURN PREDICTION MODEL REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model comparison
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-" * 30 + "\n")
            for name, metrics in self.model_performance.items():
                f.write(f"{name}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n")
                f.write(f"  AUC Score: {metrics['auc_score']:.3f}\n")
                f.write(f"  CV AUC: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std'] * 2:.3f})\n\n")
            
            # Best model details
            best_name = max(self.model_performance.keys(), 
                          key=lambda x: self.model_performance[x]['auc_score'])
            best_metrics = self.model_performance[best_name]
            
            f.write(f"BEST MODEL: {best_name}\n")
            f.write("-" * 20 + "\n")
            f.write(f"AUC Score: {best_metrics['auc_score']:.3f}\n")
            f.write(f"Accuracy: {best_metrics['accuracy']:.3f}\n\n")
            
            # Classification report
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 20 + "\n")
            f.write(classification_report(best_metrics['y_test'], best_metrics['y_pred']))
            f.write("\n")
            
            # Feature importance
            if self.feature_importance is not None:
                f.write("TOP 10 FEATURE IMPORTANCE\n")
                f.write("-" * 25 + "\n")
                for _, row in self.feature_importance.head(10).iterrows():
                    f.write(f"{row['feature']}: {row['importance']:.4f}\n")
        
        print(f"Model report saved to {output_file}")
    
    def plot_model_performance(self, save_path='model_performance.png'):
        """Create visualization of model performance."""
        if not self.model_performance:
            print("No model performance data available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        model_names = list(self.model_performance.keys())
        auc_scores = [self.model_performance[name]['auc_score'] for name in model_names]
        accuracies = [self.model_performance[name]['accuracy'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, auc_scores, width, label='AUC Score', alpha=0.8)
        axes[0, 0].bar(x + width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC curves
        for name, metrics in self.model_performance.items():
            fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
            axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {metrics["auc_score"]:.3f})')
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confusion matrix for best model
        best_name = max(self.model_performance.keys(), 
                       key=lambda x: self.model_performance[x]['auc_score'])
        best_metrics = self.model_performance[best_name]
        
        cm = confusion_matrix(best_metrics['y_test'], best_metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_name}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Model performance plots saved to {save_path}") 