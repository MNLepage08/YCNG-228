from yahoo_fin import stock_info as si
from datetime import datetime, timedelta
from functools import reduce

def test_func(ticker,date):
    return str(ticker)+str(date)

def data_train():

    # get list of S&P 500 tickers
    sp = si.tickers_sp500()
    # pull data for each S&P stock
    price_data = {ticker: si.get_data(ticker) for ticker in sp}
    combined = reduce(lambda x, y: x.append(y), price_data.values())
    return combined


def data_pred(my_ticker):

    now = datetime.now()
    start_date\
        = now - timedelta(days=10)
    pred = si.get_data(my_ticker, start_date)
    return pred