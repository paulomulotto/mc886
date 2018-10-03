import pandas as pd
from LogisticRegression import *
from SoftmaxRegression import *
from NeuralNetwork import *

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
        h_aux, thetas_aux = LogisticRegression(x=x, y=y_binary, learning_rate=learning_rate, iterations=iterations)

        h.append(h_aux)
        thetas.append(thetas_aux)

    h = np.array(h)
    thetas = np.array(thetas)

    return h.T, thetas

'''Funcao que aplica o metodo softmax regression'''
def softmax_regression(x, y, learning_rate, iterations, num_classes):

    #Array com todos os targets y para cada classe
    y_final = []

    # Transforma y em uma classificacao binaria
    for i in range(0, num_classes):

        y_binary = np.where(y == i, 1, 0)
        y_final.append(y_binary)

    y_final = np.array(y_final).T

    h, thetas = SoftmaxRegression(x=x, y=y_final, learning_rate=learning_rate, iterations=iterations, num_classes=num_classes)

    return h, thetas

'''Funcao que aplica a rede neural de uma camada escondida'''
def one_hidden_layer(x, y, num_neurons, num_classes, iterations, learning_rate):

    # Array com todos os targets y para cada classe
    y_final = []

    # Transforma y em uma classificacao binaria
    for i in range(0, num_classes):
        y_binary = np.where(y == i, 1, 0)
        y_final.append(y_binary)

    y_final = np.array(y_final).T


    #Aplica a rede neural com 1 camda escondida
    h, theta_hidden, theta_output = OneHiddenLayer(x=x, y=y_final, num_neurons=num_neurons,
                                                   num_classes=num_classes, iterations=iterations, learning_rate=learning_rate)

    # Obtem os predicts do metodo
    predicts = np.argmax(h, axis=1)

    # Contador de acertos
    contador = np.sum(predicts == y)

    #print(h.shape)
    # print(theta_hidden.shape)
    # print(theta_output.shape)

    return contador, theta_hidden, theta_output

'''Funcao que aplica a rede neural de uma camada escondida'''
def two_hidden_layer(x, y, num_neurons, num_classes, iterations, learning_rate):

    # Array com todos os targets y para cada classe
    y_final = []

    # Transforma y em uma classificacao binaria
    for i in range(0, num_classes):
        y_binary = np.where(y == i, 1, 0)
        y_final.append(y_binary)

    y_final = np.array(y_final).T

    #Aplica a rede neural com 1 camda escondida
    h, fst_theta_hidden, snd_theta_hidden, theta_output = TwoHiddenLayers(x=x, y=y_final, num_neurons=num_neurons,
                                                                          num_classes=num_classes, iterations=iterations,
                                                                          learning_rate=learning_rate)
    # Obtem os predicts do metodo
    predicts = np.argmax(h, axis=1)

    # Contador de acertos
    contador = np.sum(predicts == y)

    return contador, fst_theta_hidden, snd_theta_hidden, theta_output
    # print(h.shape)
    # print(theta_hidden.shape)
    # print(theta_output.shape)

'''Calcula a acuracia das funcoes logisticas'''
def calcula_acuracia_logistica(x, y, thetas, metodo):


    #A partir do modelo (thetas) encontra os predicts
    predicts = np.dot(x, thetas)
    predicts = np.argmax(predicts, axis=1)

    #Calcula o numero de acertos
    acertos = np.sum(predicts == y)

    print('Acuracia(' + metodo + '): '+ '{0:.2f}%'. format((acertos / len(y)) * 100))


'''Calcula a acuracia das redes neurais'''
def calcula_acuracia_rede_neural(x, y, fst_theta_hidden, snd_theta_hidden, theta_output, metodo, camadas):

    if(camadas == 1):

        # Calcula a primeira camada de neuronios

        '''Sigmoid'''
        fst_hidden = SigmoidFunction(x, fst_theta_hidden)

        '''Tanh'''
        # fst_hidden = Tanh(x, fst_theta_hidden)

        '''ReLu'''
        # fst_hidden = ReLu(x, fst_theta_hidden)

        fst_hidden = np.insert(fst_hidden, obj=0, values=1, axis=1)

        # Calcula a camada de output

        '''Sigmoid'''
        output = SigmoidFunction(fst_hidden, theta_output)

        '''Tanh'''
        # output = Tanh(fst_hidden, theta_output)

        '''ReLu'''
        # output = ReLu(fst_hidden, theta_output)

        # Obtem os predicts para as classes
        predicts = np.argmax(output, axis=1)

        # Calcula o numero de acertos da rede
        acertos = np.sum(predicts == y)

    else:

        #Calcula a primeira camada de neuronios

        '''Sigmoid'''
        fst_hidden = SigmoidFunction(x, fst_theta_hidden)

        '''Tanh'''
        # fst_hidden = Tanh(x, fst_theta_hidden)

        '''ReLu'''
        # fst_hidden = ReLu(x, fst_theta_hidden)

        fst_hidden = np.insert(fst_hidden, obj=0, values=1, axis=1)

        #Calcula a segunda camada de neuronios

        '''Sigmoid'''
        snd_hidden = SigmoidFunction(fst_hidden, snd_theta_hidden)

        '''Tanh'''
        # snd_hidden = Tanh(fst_hidden, snd_theta_hidden)

        '''ReLu'''
        # snd_hidden = ReLu(fst_hidden, snd_theta_hidden)

        snd_hidden = np.insert(snd_hidden, obj=0, values=1, axis=1)

        #Calcula a camda de output

        '''Sigmoid'''
        output = SigmoidFunction(snd_hidden, theta_output)

        '''Tanh'''
        # output = Tanh(snd_hidden, theta_output)

        '''ReLu'''
        # output = ReLu(snd_hidden, theta_output)

        #Obtem os predicts para as classes
        predicts = np.argmax(output, axis=1)

        #Calcula o numero de acertos da rede
        acertos = np.sum(predicts == y)

    print('Acuracia(Rede neural - ' + metodo + ': ' '{0:.2f}%'.format((acertos / len(y)) * 100))

def main():

    #Caminho para o arquivo csv com os dados do problema
    name = 'fashion-mnist-dataset/fashion-mnist_train.csv'

    #Le os dados do arquivo
    x, y = le_dados(name=name)

    #Obtem os dados de treino e validacao
    x_treino, y_treino, x_validacao, y_validacao = split_treino_validacao(x, y)

    '''CAMINHO PARA O CONJUNTO DE TESTE'''
    name_test = 'fashion-mnist-dataset/fashion-mnist_test.csv'

    x_test, y_test = le_dados(name=name_test)

    '''UTILIZA REGRESSÃO LOGÍSTICA'''

    #Adiciona a coluna x0 (bias == 0) a matriz x
    # x_treino = np.insert(x_treino, obj=0, values=0, axis=1)
    # x_validacao = np.insert(x_validacao, obj=0, values=0, axis=1)
    # x_test = np.insert(x_test, obj=0, values=0, axis=1)
    #
    # # Aplica o metodo de one-vs-all
    # h, thetas = one_vs_all(x=x_treino, y=y_treino, learning_rate=0.0001, iterations=100, num_classes=10)
    # calcula_acuracia_logistica(x=x_treino, y=y_treino, thetas=thetas.T, metodo='One-vs_All - Treino')
    # calcula_acuracia_logistica(x=x_validacao, y=y_validacao, thetas=thetas.T, metodo='One-vs-All - Validacao')
    #
    # #Aplica o metodo softmax regression
    # h, thetas = softmax_regression(x=x_treino, y=y_treino, learning_rate=0.0001, iterations=100, num_classes=10)
    # calcula_acuracia_logistica(x=x_treino, y=y_treino, thetas=thetas.T, metodo='Softmax - Treino')
    # calcula_acuracia_logistica(x=x_validacao, y=y_validacao, thetas=thetas.T, metodo='Softmax - Validacao')


    '''UTILIZA REDES NEURAIS'''

    # Adiciona a coluna x0 (bias == 1) a matriz x
    # x_treino = np.insert(x_treino, obj=0, values=1, axis=1)
    # x_validacao = np.insert(x_validacao, obj=0, values=1, axis=1)
    # x_test = np.insert(x_test, obj=0, values=1, axis=1)

    # Aplica a rede neural com 1 camada escondida


    '''80 por cento Sigmoid'''
    # acertos, theta_hidden, theta_output = one_hidden_layer(x=x_treino, y=y_treino, num_neurons=600, num_classes=10,
    #                                                        iterations=20, learning_rate=0.0001)
    #
    # '''80 por cento com Tanh'''
    # # acertos, theta_hidden, theta_output = one_hidden_layer(x=x_treino, y=y_treino, num_neurons=600, num_classes=10,
    # #                                                        iterations=20, learning_rate=0.000001)
    #
    # '''70 por cento com ReLu'''
    # # acertos, theta_hidden, theta_output = one_hidden_layer(x=x_treino, y=y_treino, num_neurons=600, num_classes=10,
    # #                                                        iterations=30, learning_rate=0.0000001)
    #
    # calcula_acuracia_rede_neural(x=x_treino, y=y_treino, fst_theta_hidden=theta_hidden, snd_theta_hidden=0,
    #                              theta_output=theta_output, metodo='1 Hidden Layer (Treino)', camadas=1)
    # calcula_acuracia_rede_neural(x=x_validacao, y=y_validacao, fst_theta_hidden=theta_hidden, snd_theta_hidden=0,
    #                              theta_output=theta_output, metodo='1 Hidden Layer (Validacao)', camadas=1)
    # calcula_acuracia_rede_neural(x=x_test, y=y_test, fst_theta_hidden=theta_hidden, snd_theta_hidden=0,
    #                              theta_output=theta_output, metodo='1 Hidden Layer (Teste)', camadas=1)

    #Aplica a rede neural com 2 camadas escondidas

    '''80 por cento Sigmoid e Tanh'''
    # acertos, fst_theta_hidden, snd_theta_hidden, theta_output = two_hidden_layer(x=x_treino, y=y_treino,
    #                                                                              num_neurons=600, num_classes=10,
    #                                                                              iterations=20, learning_rate=0.0001)
    # '''37 por cento com ReLu'''
    # # acertos, fst_theta_hidden, snd_theta_hidden, theta_output = two_hidden_layer(x=x_treino, y=y_treino,
    # #                                                                              num_neurons=600, num_classes=10,
    # #                                                                              iterations=30, learning_rate=0.0000001)
    #
    # calcula_acuracia_rede_neural(x=x_treino, y=y_treino, fst_theta_hidden=fst_theta_hidden,
    #                              snd_theta_hidden=snd_theta_hidden,theta_output=theta_output,
    #                              metodo='2 Hidden Layer (Treino)', camadas=2)
    # calcula_acuracia_rede_neural(x=x_validacao, y=y_validacao, fst_theta_hidden=fst_theta_hidden,
    #                              snd_theta_hidden=snd_theta_hidden,theta_output=theta_output,
    #                              metodo='2 Hidden Layer (Validacao)', camadas=2)

if __name__ == '__main__':
    main()