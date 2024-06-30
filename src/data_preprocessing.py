import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data_path):
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # Use only 'Close' price for this example
    df = df[['Close']]
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    return scaled_data, scaler

if __name__ == "__main__":
    scaled_data, scaler = preprocess_data('data/raw/stock_data.csv')
    print(scaled_data[:10])