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
def OneHiddenLayer(x, y, num_neurons, num_classes, iterations, learning_rate):

    '''Cria as matrizes com os thetas'''
    #Cria a matriz com os thetas para a camada escondida
    theta_hidden = np.random.rand(x.shape[1], num_neurons)

    # Cria a matriz com os thetas para a camada de output
    theta_output = np.random.rand(num_neurons + 1, num_classes)

    # Mini-batch Gradient Descent
    dataSet_size = x.shape[0]
    minibatch_size = 1

    # Pedict final
    h_final = np.array([])

    # Itera por numero de epocas
    for i in range(0, iterations):

        print("Épocas: ", i)

        for j in range(0, dataSet_size, minibatch_size):
        # for j in range(0, dataSet_size, minibatch_size):

            # Calcula as novas matrizes utilizadas para atualizar theta
            x_mini = x[j:j + minibatch_size]
            y_mini = y[j:j + minibatch_size]

            # Calcula os valores dos neuronios da camada escondida
            first_layer = SigmoidFunction(theta_hidden, x_mini)

            # Adiciona a coluna a0 (bias) a matriz dos neuronios da primeira camada escondida
            first_layer = np.insert(first_layer, obj=0, values=1, axis=1)

            # #Aplica Softmax function para normalizar os predicts
            output_layer = SigmoidFunction(theta_output, first_layer)
            #output_layer = SoftMax_Prediction(first_layer, theta_output)

            # Delta output
            # delta_output = np.multiply((-(y_mini - output_layer)), output_layer * (1 - output_layer))
            delta_output = (output_layer - y_mini)

            # Calcula delta da primeira camada escondida
            delta_hidden = np.multiply((first_layer * (1 - first_layer)), np.dot(delta_output, theta_output.T))

            # Retira o delta do bias para que o calculo do delta da primeira camada escondida seja correto
            delta_hidden = np.delete(delta_hidden, axis=1, obj=0)

            # Calcula o erro do gradient descent da camada output para primeira camada escondida
            error1 = np.dot(delta_output.T, first_layer)

            # Calcula o erro do gradient descent da primeira camada escondida, para a camada de input
            error = np.dot(delta_hidden.T, x_mini)

            # Atualiza o theta da primeira camada escondida para a camada de output
            theta_output = theta_output - (learning_rate * error1.T)

            # Atualiza o theta da primeira camada escondida para a camada de output
            theta_hidden = theta_hidden - (learning_rate * error.T)

            # Armazena somento o ultimo h calculado, que eh o mais atualizado
            if (j == 0):
                h_final = output_layer
            else:
                # Concatena o resultado final dos predicts
                h_final = np.append(h_final, output_layer, axis=0)

            # Retira o bias, que sera adionado novamente apos o temrino dessa iteracao, para evitar acumulo de bias
            first_layer = np.delete(first_layer, axis=1, obj=0)

        #print('delta_output', delta_output)
    # # print('output_layer: ', output_layer.shape)
    # # print('theta_output: ', theta_output.shape)
    # # print('y: ', y.shape)
    # # print('delta_output', delta_output.shape)
    # # print('delta_hidden: ', delta_hidden.shape)
    # # print('theta_hidden: ', theta_hidden.shape)
    # # print('first_layer: ', first_layer.shape)
    # # print('error1: ', error1.shape)
    # # print('error: ', error.shape)

    return h_final, theta_hidden, theta_output

'''Funcao que utiliza uma rede neural com 2 camadas escondidas'''
def TwoHiddenLayers(x, y, num_neurons, num_classes, iterations, learning_rate):

    '''Cria as matrizes com os thetas'''
    #Cria a matriz com os thetas para a camada escondida
    fst_theta_hidden = np.random.rand(x.shape[1], num_neurons)

    #Cria a matriz com os thetas para a segunda camada escondida
    snd_theta_hidden = np.random.rand(num_neurons + 1, num_neurons)

    #Cria a matriz com os thetas para a camada de output
    theta_output = np.random.rand(num_neurons + 1, num_classes)

    # Mini-batch Gradient Descent
    dataSet_size = x.shape[0]
    minibatch_size = 400

    # Pedict final
    h_final = np.array([])

    # Itera por numero de epocas
    for i in range(0, iterations):

        print("Épocas: ", i)
        for j in range(0, dataSet_size, minibatch_size):

            # Calcula as novas matrizes utilizadas para atualizar theta
            x_mini = x[j:j + minibatch_size]
            y_mini = y[j:j + minibatch_size]

            # Calcula os valores dos neuronios para a primeira camada escondida
            first_layer = SigmoidFunction(fst_theta_hidden, x_mini)

            # Adiciona a coluna a0 (bias) a matriz dos neuronios da primeira camada escondida
            first_layer = np.insert(first_layer, obj=0, values=1, axis=1)

            # Calcula os valores dos neuronios para a segunda camada escondida
            second_layer = SigmoidFunction(snd_theta_hidden, first_layer)

            # Adiciona a coluna a0 (bias) a matriz dos neuronios da segunda camada escondida
            second_layer = np.insert(second_layer, obj=0, values=1, axis=1)

            # Aplica Softmax function para normalizar os predicts
            # output_layer = SoftMax_Prediction(second_layer, theta_output)

            output_layer = SigmoidFunction(theta_output, second_layer)

            # Delta output
            delta_output = np.multiply((-(y_mini - output_layer)), output_layer * (1 - output_layer))

            # Calcula delta da segunda camada escondida
            delta_snd_hidden = np.multiply((second_layer * (1 - second_layer)), np.dot(delta_output, theta_output.T))

            # Retira o delta do bias para que o calculo do delta da primeira camada escondida seja correto
            delta_snd_hidden = np.delete(delta_snd_hidden, axis=1, obj=0)

            # Calcula delta da primeira camada escondida
            delta_fst_hidden = np.multiply((first_layer * (1 - first_layer)),
                                           np.dot(delta_snd_hidden, snd_theta_hidden.T))

            # Retira o bias para que o calculo do erro dos thetas da camada de input seja correto
            delta_fst_hidden = np.delete(delta_fst_hidden, axis=1, obj=0)

            # Calcula o erro do gradient descent da camada output para a segunda camada escondida
            error2 = np.dot(delta_output.T, second_layer)

            # Calcula o erro do gradient descent da segunda camada escondida para a primeira camada escondida
            error1 = np.dot(delta_snd_hidden.T, first_layer)

            # Calcula o erro do gradient descent da segunda camada escondida para a primeira camada escondida
            error = np.dot(delta_fst_hidden.T, x_mini)

            # Atualiza o theta da segunda camada escondida para a camada de output
            theta_output = theta_output - (learning_rate * error2.T)

            # Atualiza o theta da primeira camada escondida para a segunda camada escondida
            snd_theta_hidden = snd_theta_hidden - (learning_rate * error1.T)

            # Atualiza o theta da camada de input para a primeira camada escondida
            fst_theta_hidden = fst_theta_hidden - (learning_rate * error.T)

            # Armazena somento o ultimo h calculado, que eh o mais atualizado
            if (j == 0):
                h_final = output_layer
            else:
                # Concatena o resultado final dos predicts
                h_final = np.append(h_final, output_layer, axis=0)

            # Retira o bias, que sera adionado novamente apos o temrino dessa iteracao, para evitar acumulo de bias
            first_layer = np.delete(first_layer, axis=1, obj=0)
            second_layer = np.delete(second_layer, axis=1, obj=0)

    # print('output_layer: ', output_layer.shape)
    # print('second_layer: ', second_layer.shape)
    # print('theta_output: ', theta_output.shape)
    # print('y: ', y.shape)
    # print('delta_output', delta_output.shape)
    # print('second_layer: ', second_layer.shape)
    # print('delta_output: ', delta_output.shape)
    # print('delta_snd_hidden: ', delta_snd_hidden.shape)
    # print('theta_output: ', theta_output.shape)
    # print('snd_theta_hidden: ', snd_theta_hidden.shape)
    # print('first_layer: ', first_layer.shape)
    # print('delta_fst_hidden: ', delta_fst_hidden.shape)
    # print('error2: ', error2.shape)
    # print('error1: ', error1.shape)
    # print('error: ', error.shape)
    # print('fst_theta_hidden: ', fst_theta_hidden.shape)

    return h_final, fst_theta_hidden, snd_theta_hidden, theta_output