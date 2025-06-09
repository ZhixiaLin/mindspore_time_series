import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta
import os

def collect_stock_data(symbol='^GSPC', start_date='1949-01-01', end_date='1960-12-31'):
    """Collect S&P 500 historical data"""
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data = stock_data['Close'].reset_index()
    stock_data.columns = ['Date', 'Close']
    return stock_data

def collect_economic_data(series_id='GDP', start_date='1949-01-01', end_date='1960-12-31'):
    """Collect economic data from FRED"""
    try:
        data = pdr.data.get_data_fred(series_id, start_date, end_date)
        data = data.reset_index()
        data.columns = ['Date', series_id]
        return data
    except Exception as e:
        print(f"Error collecting {series_id}: {e}")
        return None

def main():
    # Create data directory if it doesn't exist
    if not os.path.exists('../data'):
        os.makedirs('../data')
    
    # Collect stock data
    stock_data = collect_stock_data()
    stock_data.to_csv('../data/sp500.csv', index=False)
    
    # Collect economic data
    economic_indicators = ['GDP', 'UNRATE', 'CPIAUCSL']  # GDP, Unemployment Rate, CPI
    for indicator in economic_indicators:
        data = collect_economic_data(indicator)
        if data is not None:
            data.to_csv(f'../data/{indicator.lower()}.csv', index=False)

if __name__ == "__main__":
    main() 