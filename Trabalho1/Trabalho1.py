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
    x = pd_data_value.drop(columns=['price', 'depth', 'table', 'cut', 'clarity', 'color']).values

    #print(pd_data_value.drop(columns=['price', 'depth', 'table', 'cut', 'clarity', 'color']).columns.values)

    '''Eleva a feature 'carat' ao quadrado'''
    x[:, 1] = x[:, 1]**2

    '''Eleva a frature 'x' ao quadrado'''
    x[:, 2] = x[:, 2] ** 2

    '''Eleva a frature 'y' ao quadrado'''
    x[:, 3] = x[:, 3] ** 2

    '''Eleva a frature 'z' ao quadrado'''
    x[:, 4] = x[:, 4] ** 2

    '''Eleva a frature 'depth' ao quadrado'''
    #x[:, 5] = x[:, 5] ** 2

    '''Eleva a frature 'table' ao quadrado'''
    #x[:, 6] = x[:, 6] ** 2

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

    '''Calcula o erro e os predicts'''
    for i in range(0, len(x)):
        #print("{}          {}".format(np.sum(thetas[:] * x[i, :]), y[i]))
        erro = np.sum(((thetas[:] * x[i, :]) - y[i])**2)
        predicts[i] = np.sum(thetas[:] * x[i, :])

    print(erro/(2*len(x)))

    return predicts


def main():

    '''Le o arquivo csv com os dados dos diamantes'''
    pd_data = pd.read_csv("diamonds-train.csv")

    '''Aplica para cada feature com valores nao numericos, um respectivo peso, dado pelo fornecedor dos dados'''
    pd_data_value = pd_data.replace({'cut': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5},
                                     'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                                     'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6,
                                                 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1},
                                     })

    '''Chama a funcao normal_equation, que retorna o valor dos predicts de preco dos diamantes'''
    teste = normal_eaquation(pd_data_value)













    '''PARTE UTILIZADA SO PARA PLOTAR GRAFICOS. PODE IGNORAR'''
    '''Obtem a matriz y'''
    y = pd_data['price'].values

    '''Obtem a matriz x'''
    x = pd_data_value.drop(columns=['price'])['cut'].values

    #plt.scatter(x=x, y=teste)
    #plt.show()


'''MAIN'''
if __name__ == '__main__':

    main()



