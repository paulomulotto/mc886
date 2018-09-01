import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''Funcao que realiza a obtencao de theta por meio da Equacao Normal'''
def normal_eaquation(x_treino, y_treino):

    '''Obtem a transposta de x'''
    x_t = x_treino.transpose()

    '''Multiplica x transposta por x'''
    result1 = np.matmul(x_t, x_treino)

    '''Inverte o resultado da primeira multiplicacao'''
    result2 = np.linalg.inv(result1)

    '''Multiplica o resultado da inversa por x transposto'''
    result3 = np.matmul(result2, x_t)

    '''Obtem a matriz com thetas'''
    thetas = np.matmul(result3, y_treino)

    return thetas



