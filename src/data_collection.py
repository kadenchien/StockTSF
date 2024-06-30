import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.to_csv('data/raw/stock_data.csv')
    return stock_data

def fetch_ratios(ticker):

    ratios = {}

    # Liquidity Ratios
    ratios['Current Ratio'] = ticker.balance_sheet.loc['Total Current Assets'] / ticker.balance_sheet.loc['Total Current Liabilities']
    
    # Profitability Ratios
    ratios['Gross Margin'] = ticker.financials.loc['Gross Profit'] / ticker.financials.loc['Total Revenue']
    ratios['Net Profit Margin'] = ticker.financials.loc['Net Income'] / ticker.financials.loc['Total Revenue']
    ratios['ROA'] = ticker.financials.loc['Net Income'] / ticker.balance_sheet.loc['Total Assets']
    
    # Efficiency Ratios
    ratios['Asset Turnover'] = ticker.financials.loc['Total Revenue'] / ticker.balance_sheet.loc['Total Assets']
    
    # Leverage Ratios
    ratios['Debt to Equity'] = ticker.balance_sheet.loc['Total Liabilities'] / ticker.balance_sheet.loc['Total Stockholder Equity']

    return pd.DataFrame(ratios)

if __name__ == "__main__":
    data = fetch_stock_data('AAPL', '2010-01-01', '2023-06-27')
    #print(data.head())
    # ratios = fetch_ratios(yf.Ticker("AAPL"))
    print(yf.Ticker("AAPL").balance_sheet)