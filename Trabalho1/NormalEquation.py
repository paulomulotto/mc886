import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''Funcao que realiza a obtencao de theta por meio da Equacao Normal'''
def normal_eaquation(pd_data_value):

    '''Adiciona a coluna x0 com valores 1 para a realizacao da multiplicacao de matrizes'''
    pd_data_value.insert(0, 'x0', 1)

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

    '''Cria um array vazio para armazenar o valor das predicts'''
    predicts = np.empty(len(y))

    erro = 0
    '''Calcula o erro e os predicts'''
    for i in range(0, len(x)):
        #print("{}          {}".format(np.sum(thetas[:] * x[i, :]), y[i]))
        erro += np.sum(((thetas[:] * x[i, :]) - y[i])**2)
        predicts[i] = np.sum(thetas[:] * x[i, :])

    print(erro / (2 * len(x)))

    plt.plot(range(0,len(x)), y , 'r--', range(0,len(x)), predicts, 'g--')
    #plt.show()


    return predicts



