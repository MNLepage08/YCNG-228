from yahoo_fin import stock_info as si
from datetime import datetime, timedelta
from functools import reduce
import pandas as pd


def ticker_stock():
    sp = si.tickers_sp500()
    return sp


def data_train():

    # get list of S&P 500 tickers
    sp = ticker_stock()
    # pull data for each S&P stock
    price_data = {ticker: si.get_data(ticker) for ticker in sp}
    combined = reduce(lambda x, y: x.append(y), price_data.values())

    # # take the sector
    # cie_info = {}
    # sp3 = combined['ticker'].unique().tolist()
    #
    # for ticker in sp3:
    #     # print(ticker)
    #     cie_info[ticker] = si.get_company_info(ticker)
    #     print(ticker)
    #
    # combined_info = pd.concat(cie_info)
    # print(combined_info)
    # combined_info = combined_info.reset_index()
    # combined_info.rename(columns={'level_0': 'ticker'}, inplace=True)
    # sector = combined_info.loc[combined_info['Breakdown'].isin(['sector'])]
    # print(sector)
    #
    # # merge
    # combined = combined.merge(sector, on='ticker')
    # combined.rename(columns={'Value': 'sector'}, inplace=True)
    # combined = combined.drop('Breakdown', axis=1)
    # print(combined)
    return combined


def data_pred(my_ticker):

    now = datetime.now()
    start_date = now - timedelta(days=90)
    pred = si.get_data(my_ticker, start_date)
    pred = pred[pred['close'].notnull()]
    return pred