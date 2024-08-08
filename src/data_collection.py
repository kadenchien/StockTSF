import yfinance as yf
import pandas as pd
import config as config
import requests
from textblob import TextBlob
from datetime import datetime, timedelta

def get_news_sentiment(ticker, api_key, date):
    end_date = date + timedelta(days=1)
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&sortBy=popularity&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    if not articles:
        return 0  # Neutral sentiment if no articles
    
    sentiment_sum = 0
    for article in articles[:10]:  # Analyze top 10 articles
        text = article['title'] + " " + article['description']
        analysis = TextBlob(text)
        sentiment_sum += analysis.sentiment.polarity
    
    return sentiment_sum / min(len(articles), 10)

def get_balance_sheet(api_key, ticker, years):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if "annualReports" in data:
        balance_sheets = pd.DataFrame(data["annualReports"])
        balance_sheets['fiscalDateEnding'] = pd.to_datetime(balance_sheets['fiscalDateEnding'])
        balance_sheets.set_index('fiscalDateEnding', inplace=True)
        
        # Get the most recent `years` years of data
        balance_sheets = balance_sheets.head(years)
        return balance_sheets
    else:
        print("Error fetching balance sheet data")
        return None

def get_financials(api_key, ticker, years):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if "annualReports" in data:
        financials = pd.DataFrame(data["annualReports"])
        financials['fiscalDateEnding'] = pd.to_datetime(financials['fiscalDateEnding'])
        financials.set_index('fiscalDateEnding', inplace=True)
        
        # Get the most recent `years` years of data
        financials = financials.head(years)
        return financials
    else:
        print("Error fetching financials data")
        return None

def collect_data(ticker, start_date, end_date, newsapi_key, av_api_key, years):
    # Fetch stock data
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)
    stock_data.to_csv("stock_data.csv")
    
    # Ensure the stock_data index is timezone-naive
    stock_data.index = stock_data.index.tz_localize(None)

    # Fetch financial ratios (annual balance sheet data from Alpha Vantage)
    balance_sheet = get_balance_sheet(av_api_key, ticker, years)
    financials = get_financials(av_api_key, ticker, years)
    
    if balance_sheet is not None and financials is not None:
        current_ratio = balance_sheet['totalCurrentAssets'] / balance_sheet['totalCurrentLiabilities']
        debt_to_equity = balance_sheet['totalLiabilities'] / balance_sheet['totalShareholderEquity']
        roi = financials['netIncome'] / balance_sheet['totalAssets']
        
        # Ensure all indices are timezone-naive
        current_ratio.index = current_ratio.index.tz_localize(None)
        debt_to_equity.index = debt_to_equity.index.tz_localize(None)
        roi.index = roi.index.tz_localize(None)

        # Resample ratios to match daily stock data
        current_ratio = current_ratio.resample('D').ffill()
        debt_to_equity = debt_to_equity.resample('D').ffill()
        roi = roi.resample('D').ffill()
        
        # Combine stock data with ratios
        combined_data = pd.concat([stock_data, current_ratio, debt_to_equity, roi], axis=1)
        combined_data.columns = list(stock_data.columns) + ['CurrentRatio', 'DebtToEquity', 'ROI']
    else:
        combined_data = stock_data
    
    # Add news sentiment data
    sentiment_data = []
    for date in combined_data.index:
        if date.date() >= (datetime.now() - timedelta(days=30)).date():
            sentiment = get_news_sentiment(ticker, newsapi_key, date.date())
        else:
            sentiment = 0  # or use a more sophisticated fallback method
        sentiment_data.append(sentiment)

    combined_data['Sentiment'] = sentiment_data
    
    return combined_data.dropna()

if __name__ == "__main__":
    # Collect data
    data = collect_data(config.TICKER, config.START_DATE, config.END_DATE, config.NEWSAPI_KEY, config.AV_API_KEY, config.YEARS)
    data.to_csv("data.csv")
