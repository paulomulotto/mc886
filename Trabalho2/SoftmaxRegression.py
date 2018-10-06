import numpy as np

#Aplica a funcao de calculo do predict da funcao Softmax
def SoftMax_Prediction(x, thetas):

    np.seterr(over='ignore')

    #Calcula os predicts para cada x
    scores = np.exp(np.dot(x, thetas.T))

    #Calcula sum_1->k(exp(x.thetas))
    denominador = np.sum(scores, axis=1).reshape(scores.shape[0], 1)

    #Calcula os predicts
    h = scores/denominador

    return h


# Atualiza theta com Mini-Batch Gradient Descent
def Mini_Batch_Gradient_Descent(x, y, thetas, learning_rate, iterations, y_real):

    # Mini-batch Gradient Descent
    dataSet_size = x.shape[0]
    minibatch_size = 400

    # Pedict final
    h_final = np.array([])

    # Numero de acertos por iteracao
    cont_acertos = np.array([])

    #Itera por numero de epocas
    for i in range(0, iterations):

        #Armazena o numero de acertos da iteracao
        acertos = 0

        print('Época: {}'.format(i))

        for j in range(0, dataSet_size, minibatch_size):

            # Calcula as novas matrizes utilizadas para atualizar theta
            x_mini = x[j:j + minibatch_size]
            y_mini = y[j:j + minibatch_size]
            y_real_mini = y_real[j:j + minibatch_size]

            # Calcula h (predict)
            h = SoftMax_Prediction(x=x_mini, thetas=thetas)

            # Calcula o novo theta
            thetas = thetas - (learning_rate * np.dot((h - y_mini).T, x_mini) / len(y))

            #Armazena somento o ultimo h calculado, que eh o mais atualizado
            if(i == iterations - 1):
                if(j == 0):
                    h_final = h
                else:
                    #Concatena o resultado final dos predicts
                    h_final = np.append(h_final, h, axis=0)


            # Calcula o numero de acertos
            predicts = np.argmax(h, axis=1)
            acertos += np.sum(predicts == y_real_mini)

        cont_acertos = np.append(cont_acertos, acertos)

    return h_final, thetas, cont_acertos


def SoftmaxRegression(x, y, learning_rate, iterations, num_classes, y_real):

    #Thetas
    thetas = np.zeros((num_classes, x.shape[1]))

    #Calcula o predict h
    h, thetas, cont_acertos = Mini_Batch_Gradient_Descent(x=x, y=y, thetas=thetas, learning_rate=learning_rate,
                                                          iterations=iterations, y_real=y_real)

    return h, thetas, cont_acertos