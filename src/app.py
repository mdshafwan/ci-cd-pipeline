from fastapi import FastAPI
import joblib
import pandas as pd
import time
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

app = FastAPI(title="Churn Prediction API with Prometheus Monitoring")

# 1. DEFINE PROMETHEUS METRICS
# Counter: Total number of prediction requests served
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total number of prediction requests')

# Histogram: Distribution of request latency in seconds
PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds', 'Time spent processing prediction')

# Gauge: Tracks the live prediction outputs (e.g. ratio of Churn vs No Churn)
CHURN_PREDICTIONS = Counter('ml_churn_predictions_total', 'Number of positive churn predictions')

# Load Trained Pipeline Model
model = joblib.load("production_models/current_model.pkl")

# Add Prometheus /metrics endpoint to FastAPI
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.post("/predict")
def predict(data: dict):
    start_time = time.time()
    
    # Track request count
    PREDICTION_COUNT.inc()

    # Convert incoming JSON payload to DataFrame
    df = pd.DataFrame([data])
    
    # Make Prediction
    prediction = model.predict(df)[0]
    
    # Track Churn count metric if prediction is positive (1)
    if prediction == 1:
        CHURN_PREDICTIONS.inc()

    # Record Latency Metric
    latency = time.time() - start_time
    PREDICTION_LATENCY.observe(latency)

    return {
        "churn_prediction": int(prediction),
        "latency_seconds": round(latency, 4)
    }