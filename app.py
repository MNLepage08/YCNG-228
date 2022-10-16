from flask import Flask, request

#from src.business_logic.process_query import create_business_logic
from src.IO.get_data import data_pred,data_train,test_func

app = Flask(__name__)


@app.route('/', methods=['GET'])
def welcome_msg():
    return f'Test home page!\nEX: /<date>\n'
# return f'Hi, you should use a better route:!\nEX: get_stock_val/<ticker>\n'

@app.route('/get_data', methods=['GET'])
def get_stock_value():
    my_ticker = request.args.get('ticker')
    my_date = request.args.get('date')
    print(my_date,my_ticker)
    return data_train()

    #return data_pred(my_ticker)

    # # return '''
    #               <h1>The ticker value is: {}</h1>
    #               <h1>The date value is: {}</h1>
    #               '''.format(my_ticker, my_date)


#    bl = create_business_logic()
#    prediction = bl.do_predictions_for(ticker)

#    return f'{prediction}\n'


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8081, debug=True)

