import pandas as pd
from LogisticRegression import *

def imprime_dados(x, y, x_treino, y_treino, x_validacao, y_validacao):
    print('x: [{}][{}]'.format(len(x), len(x[0])))
    print('y: [{}]'.format(len(y)))
    print('x_treino: [{}][{}]'.format(len(x_treino), len(x_treino[0])))
    print('y_treino: [{}]'.format(len(y_treino)))
    print('x_validacao: [{}][{}]'.format(len(x_validacao), len(x_validacao[0])))
    print('y_validacao: [{}]'.format(len(y_validacao)))

'''Dado um arquivo csv (name), le os dados e adiciona a coluna x0 a matriz de features'''
def le_dados(name):

    #Le os dados do arquivo csv
    data = pd.read_csv(name)

    #Obtem a matriz x das features e y dos targets
    y = data['label'].values
    x = data.drop(columns=['label']).values

    #Adiciona a coluna x0 a matriz x
    x = np.insert(x, obj=0, values=0, axis=1)

    return x, y

'''Dado duas matrizes, x (features) e y (target), separa os dados em treino (80%) e validacao (20%)'''
def split_treino_validacao(x, y):

    # Separa os dados em treino(80%) e validacao(20%)
    index = int(len(x) * 0.8)

    x_treino = x[:index].copy()
    y_treino = y[:index].copy()

    x_validacao = x[index:].copy()
    y_validacao = y[index:].copy()

    return x_treino, y_treino, x_validacao, y_validacao

def main():

    #Caminho para o arquivo csv com os dados do problema
    name = 'fashion-mnist-dataset/fashion-mnist_train.csv'

    #Le os dados do arquivo
    x, y = le_dados(name=name)

    #Obtem os dados de treino e validacao
    x_treino, y_treino, x_validacao, y_validacao = split_treino_validacao(x, y)

    theta = LogisticRegression(x_treino, y_treino, 0.1, 100)

    print(theta)

if __name__ == '__main__':
    main()