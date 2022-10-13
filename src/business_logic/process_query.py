import configparser
import logging

import joblib

from src.IO.get_data_from_yahoo import get_last_stock_price
from src.IO.storage_tools import create_bucket, get_model_from_bucket, upload_file_to_bucket
from src.algo.dummy_model import Stock_model

def create_business_logic():
    data_fetcher = get_last_stock_price
    return BusinessLogic(Stock_model(data_fetcher))

class BusinessLogic:
