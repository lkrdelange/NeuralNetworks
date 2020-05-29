import os
import pandas as pd


class Reader(object):

    @staticmethod
    def readfile():
        path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(path, 'M3Forecast.xls')
        original_data = pd.read_excel(file, 'SINGLE')
        return original_data
