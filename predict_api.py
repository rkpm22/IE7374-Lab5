"""
MLflow Model Serving API Client
Makes predictions via HTTP POST requests to MLflow model server
"""

import requests
import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import sys

# Default server URL
DEFAULT_SERVER_URL = "http://localhost:5001/invocations"

def format_predictions_for_api(data, format_type="pandas"):
    """
    Format data for MLflow API
    
    Args:
        data: numpy array or pandas DataFrame
        format_type: "pandas" or "split"
    
    Returns:
        Formatted data dict
    """
    if isinstance(data, pd.DataFrame):
        if format_type == "pandas":
            return {"dataframe_records": data.to_dict(orient="records")}
        elif format_type == "split":
            return {
                "dataframe_split": {
                    "columns": list(data.columns),
                    "data": data.values.tolist()
                }
            }
    else:
        # numpy array
        if format_type == "pandas":
            df = pd.DataFrame(data)
            return {"dataframe_records": df.to_dict(orient="records")}
        elif format_type == "split":
            return {
                "dataframe_split": {
                    "data": data.tolist()
                }
            }
    return None

def make_prediction(server_url, data, format_type="pandas"):
    """
    Make prediction request to MLflow model server
    
    Args:
        server_url: URL of the MLflow model server
        data: Input data (numpy array or pandas DataFrame)
        format_type: "pandas" or "split"
    
    Returns:
        Response object
    """
    try:
        # Format data
        payload = format_predictions_for_api(data, format_type)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Make request
        response = requests.post(
            server_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        # Check for errors
        response.raise_for_status()
        
        return response
    
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to server at {server_url}")
        print("Make sure the MLflow model server is running:")
        print("  mlflow models serve -m <model_uri> -p 5001 --no-conda")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)

def print_formatted_predictions(predictions, actual=None):
    """
    Print predictions in a formatted way
    
    Args:
        predictions: List or array of predictions
        actual: Optional actual values for comparison
    """
    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)
    
    if isinstance(predictions, list):
        pred_array = np.array(predictions)
    else:
        pred_array = predictions
    
    if actual is not None:
        if isinstance(actual, list):
            actual_array = np.array(actual)
        else:
            actual_array = actual
        
        print(f"{'Index':<8} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
        print("-" * 60)
        for i in range(len(pred_array)):
            error = abs(pred_array[i] - actual_array[i])
            print(f"{i:<8} {pred_array[i]:<15.4f} {actual_array[i]:<15.4f} {error:<15.4f}")
        
        # Calculate statistics
        mae = np.mean(np.abs(pred_array - actual_array))
        rmse = np.sqrt(np.mean((pred_array - actual_array) ** 2))
        print("\n" + "-" * 60)
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
    else:
        print(f"{'Index':<8} {'Predicted Value':<20}")
        print("-" * 30)
        for i, pred in enumerate(pred_array):
            print(f"{i:<8} {pred:<20.4f}")
    
    print("="*60 + "\n")

def main():
    """Main function to test model serving API"""
    
    # Get server URL from command line or use default
    server_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER_URL
    
    print(f"Connecting to MLflow model server at: {server_url}")
    
    # Load test data
    print("\nLoading California Housing test data...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Use a subset for testing
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test 1: Single prediction
    print("\n" + "="*60)
    print("TEST 1: Single Sample Prediction")
    print("="*60)
    single_sample = X_test[0:1]
    response = make_prediction(server_url, single_sample)
    predictions = response.json()
    
    # Handle different response formats
    if isinstance(predictions, list):
        pred_values = predictions
    elif isinstance(predictions, dict) and "predictions" in predictions:
        pred_values = predictions["predictions"]
    else:
        pred_values = predictions
    
    print(f"Input shape: {single_sample.shape}")
    print(f"Response status: {response.status_code}")
    print(f"Prediction: {pred_values}")
    
    # Test 2: Batch predictions (multiple samples)
    print("\n" + "="*60)
    print("TEST 2: Batch Predictions (10 samples)")
    print("="*60)
    batch_samples = X_test[:10]
    batch_actual = y_test[:10]
    
    response = make_prediction(server_url, batch_samples)
    predictions = response.json()
    
    # Handle different response formats
    if isinstance(predictions, list):
        batch_predictions = predictions if isinstance(predictions[0], (int, float)) else [p[0] for p in predictions]
    elif isinstance(predictions, dict) and "predictions" in predictions:
        batch_predictions = predictions["predictions"]
    else:
        batch_predictions = predictions
    
    print(f"Input shape: {batch_samples.shape}")
    print(f"Response status: {response.status_code}")
    print_formatted_predictions(batch_predictions, batch_actual)
    
    # Test 3: Using pandas DataFrame format
    print("\n" + "="*60)
    print("TEST 3: Batch Predictions with DataFrame Format")
    print("="*60)
    df_samples = pd.DataFrame(batch_samples, columns=data.feature_names)
    response = make_prediction(server_url, df_samples, format_type="pandas")
    predictions = response.json()
    
    # Handle different response formats
    if isinstance(predictions, list):
        df_predictions = predictions if isinstance(predictions[0], (int, float)) else [p[0] for p in predictions]
    elif isinstance(predictions, dict) and "predictions" in predictions:
        df_predictions = predictions["predictions"]
    else:
        df_predictions = predictions
    
    print(f"Input DataFrame shape: {df_samples.shape}")
    print(f"Response status: {response.status_code}")
    print_formatted_predictions(df_predictions[:5], batch_actual[:5])  # Show first 5
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()
