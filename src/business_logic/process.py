import pickle
import configparser
import logging
import joblib

from src.IO.storage_tools import upload_file_to_bucket, create_bucket, get_model_from_bucket


root_bucket = 'mnl009_model_bucket_ycng_228'
config = configparser.ConfigParser()
config.read('application.conf')
create_bucket(root_bucket)


# def get_version():
#    return config['DEFAULT']['version']

# def get_bucket_name():
#    return f'{root_bucket}_{get_version().replace(".", "")}'


def load_model_in_bucket():
    model_filename = 'my_model_v6.pkl'
    log = logging.getLogger()
    log.warning(f'training model for GCP')
    upload_file_to_bucket(model_filename, root_bucket)


def predict_model(my_data_predict):
    pickled_model = pickle.load(open('my_model_v6.pkl', 'rb'))
    pred_test = pickled_model.predict(my_data_predict)
    return pred_test.flatten()[-1]


def predict_model_from_GCP(my_data_predict):
    model_filename = 'my_model_v6.pkl'
    model = get_model_from_bucket(model_filename, root_bucket)
    pred_test = model.predict(my_data_predict)
    return pred_test.flatten()[-1]
