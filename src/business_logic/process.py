import pickle
import configparser
import logging
import joblib

from src.IO.storage_tools import upload_file_to_bucket, create_bucket, get_model_from_bucket


# def create_business_logic():
    # data_fetcher = get_last_stock_price
    # return BusinessLogic(Stock_model(data_fetcher))


root_bucket = 'mnl009_model_bucket_ycng_228'
config = configparser.ConfigParser()
config.read('application.conf')
create_bucket(root_bucket)


# def get_version():
#    return config['DEFAULT']['version']

# def get_bucket_name():
#    return f'{root_bucket}_{get_version().replace(".", "")}'


def load_model_in_bucket():
    model_filename = 'my_model.pkl'
    log = logging.getLogger()
    log.warning(f'training model for GCP')
    # model = pickled_model.fit(model_filename)
    with open(model_filename, 'wb') as f:
        joblib.dump(model_filename, f)
    upload_file_to_bucket(model_filename, root_bucket)
    model = get_model_from_bucket(model_filename, root_bucket)
    return model


def predict_model(my_data_predict):
    pickled_model = pickle.load(open('my_model.pkl', 'rb'))
    pred_test = pickled_model.predict(my_data_predict)
    return pred_test.flatten()[-1]


def predict_model_from_GCP(my_data_predict):
    #model = load_model_in_bucket()
    model_filename = 'my_model.pkl'
    model = get_model_from_bucket(model_filename, root_bucket)
    pred_test = model.predict(my_data_predict)
    return pred_test.flatten()[-1]
