import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    '''Le o arquivo csv com os dados dos diamantes'''
    pd_data = pd.read_csv("diamonds-test.csv")

    '''Realiza a separacao das features cujo valor, nao sao numericos'''
    pd_data_value = pd.get_dummies(pd_data)

    '''Transforma o dataframe do pandas para numpay array'''
    np_data = pd_data_value.values

'''MAIN'''
if __name__ == '__main__':

    main()



