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

def collect_data(ticker, start_date, end_date, newsapi_key):
    # Fetch stock data
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)

    stock_data.to_csv("stock_data.csv")
    
    # Ensure the stock_data index is timezone-naive
    stock_data.index = stock_data.index.tz_localize(None)

    # Fetch financial ratios (quarterly)
    financials = stock.quarterly_financials
    balance_sheet = stock.quarterly_balance_sheet
    # Calculate financial ratios
    current_ratio = balance_sheet.loc['Current Assets'] / balance_sheet.loc['Current Liabilities']
    debt_to_equity = balance_sheet.loc['Total Liabilities Net Minority Interest'] / balance_sheet.loc['Stockholders Equity']
    roi = financials.loc['Net Income'] / balance_sheet.loc['Total Assets']
    
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
    data = collect_data(config.TICKER, config.START_DATE, config.END_DATE, config.NEWSAPI_KEY)
    data.to_csv("data.csv")