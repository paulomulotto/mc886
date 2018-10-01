import numpy as np

'''Funcao que aplica a funcao sigmoid'''
def SigmoidFunction(theta, x):

    #Calcula z
    z = np.dot(x, theta)

    #Calcula h (predict)
    h = 1 / (1 + np.exp(-z))

    return h

#Aplica a funcao de calculo do predict da funcao Softmax
def SoftMax_Prediction(x, thetas):

    #Calcula os predicts para cada x
    scores = np.exp(np.dot(x, thetas))

    #Calcula sum_1->k(exp(x.thetas))
    denominador = np.sum(scores, axis=1).reshape(scores.shape[0],1)

    #Calcula os predicts
    h = scores/denominador

    return h

'''Funcao que utiliza uma rede neural com 1 camada escondida'''
def OneHiddenLayer(x, y, num_neurons, num_classes):

    #Cria a matriz com os thetas para a camada escondida
    theta_hidden = np.random.rand(x.shape[1], num_neurons)

    #Calcula os valores dos neuronios da camada escondida
    first_layer = SigmoidFunction(theta_hidden, x)

    #Adiciona a coluna a0 (bias) a matriz dos neuronios da primeira camada escondida
    first_layer = np.insert(first_layer, obj=0, values=1, axis=1)

    #Cria a matriz com os thetas para a camada de output
    theta_output = np.random.rand(first_layer.shape[1], num_classes)

    #Aplica Softmax function para normalizar os predicts
    output_layer = SoftMax_Prediction(first_layer, theta_output)

    return output_layer, theta_hidden, theta_output

'''Funcao que utiliza uma rede neural com 2 camadas escondidas'''
def TwoHiddenLayers(x, y, num_neurons, num_classes):

    #Cria a matriz com os thetas para a camada escondida
    fst_theta_hidden = np.random.rand(x.shape[1], (num_neurons - 100))

    #Calcula os valores dos neuronios para a primeira camada escondida
    first_layer = SigmoidFunction(fst_theta_hidden, x)

    #Adiciona a coluna a0 (bias) a matriz dos neuronios da primeira camada escondida
    first_layer = np.insert(first_layer, obj=0, values=1, axis=1)

    #Cria a matriz com os thetas para a segunda camada escondida
    snd_theta_hidden = np.random.rand(first_layer.shape[1], 100)

    #Calcula os valores dos neuronios para a segunda camada escondida
    second_layer = SigmoidFunction(snd_theta_hidden, first_layer)

    # Adiciona a coluna a0 (bias) a matriz dos neuronios da segunda camada escondida
    second_layer = np.insert(second_layer, obj=0, values=1, axis=1)

    #Cria a matriz com os thetas para a camada de output
    theta_output = np.random.rand(second_layer.shape[1], num_classes)

    #Aplica Softmax function para normalizar os predicts
    output_layer = SoftMax_Prediction(second_layer, theta_output)

    return output_layer, fst_theta_hidden, snd_theta_hidden, theta_output