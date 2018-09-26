import numpy as np

'''Funcao que aplica a funcao sigmoid'''
def SigmoidFunction(theta, x):

    #Calcula z
    z = np.dot(x, theta)

    #Calcula h (predict)
    h = 1 / (1 + np.exp(-z))

    return h

#Atualiza theta com Batch Gradient Descent
def Batch_Gradient_Descent(x, y, theta, learning_rate, iterations):


    #Batch Gradient Descent
    for i in range(0, iterations):

        # Calcula h (predict)
        h = SigmoidFunction(theta, x)

        # Calcula o novo theta
        theta = theta - (learning_rate * (np.dot((h - y), x) / len(y)))

    return h, theta


# Atualiza theta com Stochastic Gradient Descent
def Stochastic_Gradient_Descent(x, y, theta, learning_rate, iterations):

    # Stochastic Gradient Descent
    for i in range(0, x.shape[0]):

        #Calcula h (predict)
        h = SigmoidFunction(theta, x[i])

        # Calcula o novo theta
        theta = theta - (learning_rate * (np.dot((h - y[i]), x[i]) / len(y)))

    return h, theta


# Atualiza theta com Mini-Batch Gradient Descent
def Mini_Batch_Gradient_Descent(x, y, theta, learning_rate, iterations):

    # Mini-batch Gradient Descent
    dataSet_size = x.shape[0]
    minibatch_size = 400

    # Pedict final
    h_final = np.array([])

    # Randomiza os dados para obter os mini batches
    # x = np.insert(x, obj=0, values=y, axis=1)
    # np.random.shuffle(x)
    # y = x[:,0]
    # x = x[:,1:]

    #Itera por numero de epocas
    for i in range(0, iterations):

        for j in range(0, dataSet_size, minibatch_size):

            # Calcula as novas matrizes utilizadas para atualizar theta
            x_mini = x[j:j + minibatch_size]
            y_mini = y[j:j + minibatch_size]

            # Calcula h (predict)
            h = SigmoidFunction(theta, x_mini)

            # Calcula o novo theta
            theta = theta - (learning_rate * (np.dot((h - y_mini), x_mini) / len(y)))

            #Armazena somento o ultimo h calculado, que eh o mais atualizado
            if(i == iterations - 1):
                #Concatena o resultado final dos predicts
                h_final = np.append(h_final, h)


    return h_final, theta

'''Funcao que realiza a regressao logistica'''
def LogisticRegression(x, y, learning_rate, iterations):

    #Inicaliza o vetor de parametros theta
    theta = np.zeros(x.shape[1])

    #Obtem theta e h por meio de Mini-Batch Gradient Descent
    h, theta = Mini_Batch_Gradient_Descent(x, y, theta, learning_rate, iterations)

    return h, theta