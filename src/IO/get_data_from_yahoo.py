from datetime import datetime, timedelta
from functools import reduce
from yahoo_fin import stock_info as si


def get_last_stock_price(ticker, date, last=False):
    if last:
        date_object = datetime.strptime(date, '%m-%d-%Y').date()
        start_date = date_object - timedelta(days=10)
        return si.get_data(ticker, start_date)

    sp = si.tickers_sp500()
    # pull data for each S&P stock
    price_data = {ticker: si.get_data(ticker) for ticker in sp}
    combined = reduce(lambda x, y: x.append(y), price_data.values())

    return combined


def get_train_stock_price():
    # get list of S&P 500 tickers
    sp = si.tickers_sp500()
    # pull data for each S&P stock
    price_data = {ticker: si.get_data(ticker) for ticker in sp}
    combined = reduce(lambda x, y: x.append(y), price_data.values())
    return combined


def ticker_stock():
    sp = si.tickers_sp500()
    return sp

