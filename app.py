from flask import Flask, request
from src.IO.get_data import data_pred, data_train, ticker_stock
from src.algo.model import train_model
from src.algo.transfomData import transform_data
from src.business_logic.process import predict_model, load_model_in_bucket, predict_model_from_GCP


app = Flask(__name__)


@app.route('/', methods=['GET'])
def welcome_msg():
    return f'Hi, please use the route: get_predict_data/ticker/YYYY-MM-DD \n'


@app.route('/get_data_train/<my_date>', methods=['GET'])
def get_stock_value(my_date):
    # my_ticker = request.args.get('ticker')
    # my_date = request.args.get('date')
    # print(my_date,my_ticker)
    my_sp = ticker_stock()
    my_train_dataframe = data_train()
    train_model(my_train_dataframe, my_sp, my_date)
    return "Train done!"

@app.route('/get_predict_data/<my_ticker>/<my_date>', methods=['GET'])
def get_predict_value(my_ticker, my_date):
    my_data_predict = data_pred(my_ticker)
    my_date_predict = my_date
    my_test = transform_data(my_data_predict, my_date_predict)
    my_predict_value = predict_model(my_test)
    return my_predict_value

@app.route('/test1/<my_ticker>/<my_date>', methods=['GET'])
def test1(my_ticker, my_date):
    load_model_in_bucket()
    my_data_predict = data_pred(my_ticker)
    my_date_predict = my_date
    my_test = transform_data(my_data_predict, my_date_predict)
    my_predict_value2 = predict_model_from_GCP(my_test)
    return my_predict_value2


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8081, debug=True)

