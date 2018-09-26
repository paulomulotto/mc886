import numpy as np

#Aplica a funcao de calculo do predict da funcao Softmax
def SoftMax_Prediction(x, thetas):

    #Calcula os predicts para cada x
    scores = np.exp(np.dot(x, thetas.T))

    #Calcula sum_1->k(exp(x.thetas))
    denominador = np.sum(scores, axis=1)

    #Calcula os predicts
    h = []

    '''Loop onde serao calculados os predicts. As linhas separam os predicts por classe e as colunas pelo x 
    correspondente'''
    for i in range(0, 10):
        h.append(scores[:,i]/denominador[:])

    '''h agora armazena cada x em uma linha, e cada coluna da linha, corresponde ao predict para a determinada
    classe'''
    h = np.array(h).T
    print(h.shape)

    return


def SoftmaxRegression(x, y, learning_rate, iterations, num_classes):

    #Thetas
    thetas = np.zeros((num_classes, x.shape[1]))

    #Calcula o predict h
    SoftMax_Prediction(x, thetas)