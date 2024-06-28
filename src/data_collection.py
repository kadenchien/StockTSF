import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.to_csv('data/raw/stock_data.csv')
    return stock_data

if __name__ == "__main__":
    data = fetch_stock_data('PANW', '2010-01-01', '2023-06-27')
    print(data.head())