import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import *


'''Funcao que realiza a obtencao de theta por meio da Equacao Normal'''
def normal_eaquation(pd_data_value):

    '''Adiciona a coluna x0 com valores 1 para a realizacao da multiplicacao de matrizes'''
    pd_data_value.insert(0, 'x0', 1)

    '''Transforma o dataframe do pandas para numpay array'''
    np_data = pd_data_value.values

    '''Obtem a matriz y'''
    y = pd_data_value['price'].values

    '''Obtem a matriz x'''
    x = pd_data_value.drop(columns=['price']).values

    '''Obtem a transposta de x'''
    x_t = x.transpose()

    '''Multiplica x transposta por x'''
    result1 = np.matmul(x_t, x)

    '''Inverte o resultado da primeira multiplicacao'''
    result2 = np.linalg.inv(result1)

    '''Multiplica o resultado da inversa por x transposto'''
    result3 = np.matmul(result2, x_t)

    '''Obtem a matriz com thetas'''
    thetas = np.matmul(result3, y)


def main():

    '''Le o arquivo csv com os dados dos diamantes'''
    pd_data = pd.read_csv("diamonds-test.csv")

    '''Realiza a separacao das features cujo valor, nao sao numericos'''
    pd_data_value = pd.get_dummies(pd_data)

    normal_eaquation(pd_data_value)


'''MAIN'''
if __name__ == '__main__':

    main()



