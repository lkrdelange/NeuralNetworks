import pandas as pd


# Class for reading the data
class Reader(object):

    @staticmethod
    def readfile():
        original_data = pd.read_excel('D:\\Neural\\data\\M3Forecast.xls', 'SINGLE')
        return original_data
