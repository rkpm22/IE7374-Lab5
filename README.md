# MLflow Experiment Tracking Lab

This lab demonstrates MLflow experiment tracking using the **California Housing Dataset** from scikit-learn. The lab covers autologging, manual tracking, model loading, and model serving.

## Dataset

**California Housing Dataset** (Regression Task)
- **Features**: 8 features including median income, house age, rooms, bedrooms, population, occupancy, latitude, and longitude
- **Target**: Median house value (in hundreds of thousands of dollars)
- **Samples**: ~20,000
- **Task Type**: Regression

## Lab Structure

### Part 1: MLflow Autologging
- Enable MLflow autologging for scikit-learn
- Train Random Forest and Gradient Boosting models
- Automatically log parameters, metrics, and models
- View results in MLflow UI

### Part 2: Manual Tracking
- Use `mlflow.start_run()` context manager
- Manually log hyperparameters (3+)
- Manually log metrics (3+): RMSE, MAE, R², and custom MAPE
- Log models with signature inference
- Log artifacts (feature importance plots)
- Use tags to categorize runs

### Part 3: Model Loading
- Load previously saved models using run ID
- Make predictions on test data
- Verify model correctness

### Part 4: Model Serving
- Serve models as REST APIs using `mlflow models serve`
- Make predictions via HTTP POST requests
- Test with batch predictions
- Robust error handling in API client

## Files

- **`lab_notebook.ipynb`**: Main Jupyter notebook with all 4 parts implemented
- **`train_model.py`**: Python script for training models with manual MLflow tracking
- **`predict_api.py`**: Python script for testing MLflow model serving API
- **`requirements.txt`**: Project dependencies

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the notebook:**
   ```bash
   jupyter notebook lab_notebook.ipynb
   ```

3. **Or run the training script:**
   ```bash
   python train_model.py
   ```

4. **View MLflow UI:**
   ```bash
   mlflow ui
   ```
   Open http://localhost:5000 in your browser

## Model Serving

1. **Start the MLflow model server:**
   ```bash
   # First, get a run ID from MLflow UI or from training output
   mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda
   ```

2. **Test the API:**
   ```bash
   python predict_api.py
   ```

   Or use the API directly:
   ```bash
   curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_split": {
         "data": [[8.3252, 41.0, 6.98412698, 1.02380952, 322.0, 2.55555556, 37.88, -122.23]]
       }
     }'
   ```

## Results Summary

The lab demonstrates:

- ✅ **Autologging**: Automatic tracking of sklearn models
- ✅ **Manual Tracking**: Full control over logging with `mlflow.start_run()`
- ✅ **Model Loading**: Retrieval and prediction with saved models
- ✅ **Model Serving**: REST API for production deployments

### Variations Implemented:

1. **Additional Metrics**: Custom MAPE (Mean Absolute Percentage Error) metric
2. **Tags and Notes**: Used `mlflow.set_tags()` to categorize runs (baseline, improved, final)
3. **Artifact Logging**: Logged feature importance visualizations using `mlflow.log_artifact()`
4. **Multiple Runs Comparison**: Compared different models and hyperparameters
5. **Better API Testing**: Created robust prediction script with error handling and formatted output

## Key MLflow Features Demonstrated

- `mlflow.sklearn.autolog()`: Automatic logging for scikit-learn
- `mlflow.set_experiment()`: Named experiment management
- `mlflow.start_run()`: Manual run tracking
- `mlflow.log_params()`: Parameter logging
- `mlflow.log_metrics()`: Metric logging
- `mlflow.set_tags()`: Tagging runs for organization
- `mlflow.sklearn.log_model()`: Model logging with signatures
- `mlflow.log_artifact()`: Artifact logging (visualizations)
- `mlflow.load_model()`: Model loading
- `mlflow.models.serve()`: Model serving

## Notes

- All experiments are stored in the `mlruns/` directory by default
- Run IDs are printed after each training run for reference
- Model serving requires the model server to be running before making API requests
- Use `mlflow ui` to visually compare runs and select the best model