#import pandas as pd
#import numpy as np
#from sklearn.linear_model import LogisticRegression
import pickle
import os
from pathlib import Path


def predict_model(my_data_predict):

    # cwd = os.getcwd()
    # print("print current dir - fpr process ")
    # print(cwd)
    # path1 = Path(cwd)
    # my_model_path=path1.parent.parent

    # my_model_path=str(my_model_path)
    # print("--------Go to parent OF parent ----------")
    # print(my_model_path)

    #my_model_path='/Users/marie-noellepage/Documents/Formation/McGill/2.AppliedArtificialIntelligence/1.PredictiveandClassificationModelling/YCNG-228/'
    pickled_model = pickle.load(open('my_model.pkl', 'rb'))
    pred_test = pickled_model.predict(my_data_predict)
    return pred_test.flatten()[-1]
