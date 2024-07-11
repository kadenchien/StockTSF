import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    
    y_test_inv = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), scaler.n_features_in_-1))]))[:, 0]
    predictions_inv = scaler.inverse_transform(np.hstack([predictions, np.zeros((len(predictions), scaler.n_features_in_-1))]))[:, 0]
    
    mse = np.mean((y_test_inv - predictions_inv)**2)
    rmse = np.sqrt(mse)
    
    print(f"Root Mean Squared Error: {rmse}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(predictions_inv, label='Predicted Price')
    plt.title('PANW Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()