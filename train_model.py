"""
MLflow Model Training Script
Trains models on California Housing dataset with manual MLflow tracking
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Set experiment name
experiment_name = "california_housing_experiment"
mlflow.set_experiment(experiment_name)

# Load California Housing dataset
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

def train_and_log_model(model, model_name, hyperparameters, tags=None):
    """
    Train a model and log it to MLflow
    
    Args:
        model: sklearn model instance
        model_name: Name for the model
        hyperparameters: Dict of hyperparameters to log
        tags: Optional dict of tags to log
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Set tags if provided
        if tags:
            mlflow.set_tags(tags)
        
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        
        # Train model
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Log custom metrics (mean absolute percentage error)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        mlflow.log_metric("train_mape", train_mape)
        mlflow.log_metric("test_mape", test_mape)
        
        # Log model with signature inference
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=mlflow.models.infer_signature(X_test[:5], y_test_pred[:5])
        )
        
        # Create and log feature importance plot
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.title(f"Feature Importances - {model_name}")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            
            os.makedirs("artifacts", exist_ok=True)
            plt.savefig(f"artifacts/{model_name}_feature_importance.png")
            mlflow.log_artifact(f"artifacts/{model_name}_feature_importance.png")
            plt.close()
        
        print(f"\n{model_name} Results:")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test MAPE: {test_mape:.2f}%")
        print(f"  Run ID: {run.info.run_id}")
        
        return run.info.run_id

if __name__ == "__main__":
    # Train Random Forest Regressor
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    rf_tags = {
        "model_type": "RandomForest",
        "version": "baseline",
        "dataset": "california_housing"
    }
    rf_model = RandomForestRegressor(**rf_params)
    rf_run_id = train_and_log_model(rf_model, "RandomForest_Baseline", rf_params, rf_tags)
    
    # Train Gradient Boosting Regressor
    gb_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42
    }
    gb_tags = {
        "model_type": "GradientBoosting",
        "version": "improved",
        "dataset": "california_housing"
    }
    gb_model = GradientBoostingRegressor(**gb_params)
    gb_run_id = train_and_log_model(gb_model, "GradientBoosting_Improved", gb_params, gb_tags)
    
    # Train another Gradient Boosting with different hyperparameters for comparison
    gb_params_v2 = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": 42
    }
    gb_tags_v2 = {
        "model_type": "GradientBoosting",
        "version": "final",
        "dataset": "california_housing"
    }
    gb_model_v2 = GradientBoostingRegressor(**gb_params_v2)
    gb_v2_run_id = train_and_log_model(gb_model_v2, "GradientBoosting_Final", gb_params_v2, gb_tags_v2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Random Forest Run ID: {rf_run_id}")
    print(f"Gradient Boosting Run ID: {gb_run_id}")
    print(f"Gradient Boosting V2 Run ID: {gb_v2_run_id}")
    print("\nView results in MLflow UI: mlflow ui")
