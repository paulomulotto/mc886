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

'''Funcao que aplica o metodo one-vs-all para mais que duas classes'''
def one_vs_all(x, y, learning_rate, iterations, num_classes):

    #Array h com as predicts de cada classe para cada imagem
    # h = np.array([])

    for i in range(0, num_classes):
        #Transforma y em uma classificacao binaria
        y_binary = np.where(y == i, 1, 0)

        h = LogisticRegression(x, y_binary, learning_rate, iterations)

    print(len(h))


def main():

    #Caminho para o arquivo csv com os dados do problema
    name = 'fashion-mnist-dataset/fashion-mnist_train.csv'

    #Le os dados do arquivo
    x, y = le_dados(name=name)

    #Obtem os dados de treino e validacao
    x_treino, y_treino, x_validacao, y_validacao = split_treino_validacao(x, y)

    #Aplica o metodo de one-vs-all
    one_vs_all(x_treino, y_treino, 0.000001, 500, 10)

if __name__ == '__main__':
    main()