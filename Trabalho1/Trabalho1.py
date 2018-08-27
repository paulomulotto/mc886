import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''MAIN'''
if __name__ == '__main__':

    '''Le o arquivo csv com os dados dos diamantes'''
    pd_data = pd.read_csv("diamonds-test.csv")

    '''Transforma o dataframe do pandas para numpay array'''
    np_data = pd_data.values
