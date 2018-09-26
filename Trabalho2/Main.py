import pandas as pd
from LogisticRegression import *
from SoftmaxRegression import *

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
    h = []

    #Array com os thetas (modelos)
    thetas = []

    #Calcula os predicts para cada classe
    for i in range(0, num_classes):
        #Transforma y em uma classificacao binaria
        y_binary = np.where(y == i, 1, 0)

        #Armazena um array com todos os hs e thetas retornados
        h_aux, thetas_aux = LogisticRegression(x, y_binary, learning_rate, iterations)

        h.append(h_aux)
        thetas.append(thetas_aux)

    h = np.array(h)
    thetas = np.array(thetas)

    #Obtem os predicts do metodo
    predicts = np.argmax(h.T, axis=1)

    # Contador de acertos
    contador = np.sum(predicts == y)

    return h.T, thetas, contador

'''Funcao que aplica o metodo softmax regression'''
def softmax_regression():
    return


def main():

    #Caminho para o arquivo csv com os dados do problema
    name = 'fashion-mnist-dataset/fashion-mnist_train.csv'

    #Le os dados do arquivo
    x, y = le_dados(name=name)

    #Obtem os dados de treino e validacao
    x_treino, y_treino, x_validacao, y_validacao = split_treino_validacao(x, y)

    #Aplica o metodo de one-vs-all
    # h, thetas, acertos = one_vs_all(x=x_treino, y=y_treino, learning_rate=0.001, iterations=5, num_classes=10)
    # print('Acuracia: {0:.2f}%'.format((acertos / len(y_treino)) * 100))

    #Aplica o metodo softmax regression
    SoftmaxRegression(x=x_treino, y=y_treino, learning_rate=0.01, iterations=5, num_classes=10)



if __name__ == '__main__':
    main()