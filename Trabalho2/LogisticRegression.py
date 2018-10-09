import numpy as np

'''Funcao que aplica a funcao sigmoid'''
def SigmoidFunction(x, theta):

    # np.seterr(over='ignore')

    #Calcula z
    z = np.dot(x, theta)

    #Calcula h (predict)
    h = np.float64(1 / (1 + np.float64(np.exp(-z))))

    return h


# Atualiza thetas com Mini-Batch Gradient Descent
def Mini_Batch_Gradient_Descent(x, y, theta, learning_rate, iterations):

    # Mini-batch Gradient Descent
    dataSet_size = x.shape[0]
    minibatch_size = 400

    # Pedict final
    h_final = np.array([])

    #Itera por numero de epocas
    for i in range(0, iterations):

        for j in range(0, dataSet_size, minibatch_size):

            # Calcula as novas matrizes utilizadas para atualizar theta
            x_mini = x[j:j + minibatch_size]
            y_mini = y[j:j + minibatch_size]

            # Calcula h (predict)
            h = SigmoidFunction(x_mini, theta)

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