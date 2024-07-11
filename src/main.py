from config import NEWSAPI_KEY, TICKER, START_DATE, END_DATE, LOOK_BACK
from data_collection import collect_data
from data_preprocessing import preprocess_data
from model import train_model
from evaluation import evaluate_model, plot_training_history
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    # Collect data
    data = collect_data(TICKER, START_DATE, END_DATE, NEWSAPI_KEY)
    
    # Consider using only the last N years of data
    N_YEARS = 5
    data = data.last(f'{N_YEARS}Y')
    
    # Preprocess data
    X, y, scaler = preprocess_data(data, LOOK_BACK)
    
    # Split data, ensuring the test set is the most recent data
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train model
    model, history = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, scaler)
    
    # Plot training history
    plot_training_history(history)