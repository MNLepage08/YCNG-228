from yahoo_fin import stock_info as si
from functools import reduce
from datetime import datetime, timedelta

def get_stock_price_model():
    # get list of S&P 500 tickers
    sp = si.tickers_sp500()
    # pull data for each S&P stock
    price_data = {ticker: si.get_data(ticker) for ticker in sp}
    combined = reduce(lambda x, y: x.append(y), price_data.values())
    return combined

def get_last_stock_price(ticker, date, last=False):
    if last:
        #now = datetime.now()
        #start_date = now - timedelta(days=10)
        start_date = date - timedelta(days=10)
        return si.get_data(ticker, start_date)
    return si.get_data(ticker)