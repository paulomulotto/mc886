import numpy as np

'''Funcao que aplica a funcao sigmoid'''
def SigmoidFunction(theta, x):

    #Calcula z
    z = np.dot(x, theta)

    #Calcula h (predict)
    h = 1 / (1 + np.exp(-z))

    return h

'''Funcao que realiza a regressao logistica'''
def LogisticRegression(x, y, learning_rate, iterations):

    #Inicaliza o vetor de parametros theta
    theta = np.zeros(x.shape[1])

    #Realiza o numero de iteracoes
    for i in range(0, iterations):

        # Calcula h (predict)
        h = SigmoidFunction(theta, x)

        # Calcula o novo theta
        theta = theta - (learning_rate * (np.dot((h - y), x) / len(y)))

    return theta