import pandas as pd
from src.data_loader import load_data
from src.model import train_model, predict
from src.utils import evaluate

# Load data
df = load_data('data/raw/sample_fraud.csv')

# Prepare features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Train model
model = train_model(X)

# Predict
y_pred = predict(model, X)

# Convert Isolation Forest outputs to 0 (normal), 1 (fraud)
y_pred = [1 if p == -1 else 0 for p in y_pred]

# Evaluate
evaluate(y, y_pred)
