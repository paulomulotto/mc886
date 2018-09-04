# Gradient Descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd


# Calcula J(thetas)
def calc_custo(y, x, thetas):

    custo = sum((x.dot(thetas)-y)**2)/(2*len(y))

    return custo



def gradient_descent(x, y, criterio_convergencia, alfa, maximo_iteracoes):

    thetas = np.random.randn(x.shape[1])

    m = y.shape[0]
    
    n = 0  #numero máximo de iterações
    custos = np.zeros(maximo_iteracoes)


    convergiu = False
    while not convergiu:
        predicao = x.dot(thetas)
        erro = predicao - y
        derivada = x.T.dot(erro) / m
        thetas = thetas - alfa * derivada
        custos[n] = calc_custo(y, x, thetas)

        '''Verifica se atingiu o numero maximo de iteracoes'''
        if n == maximo_iteracoes - 1:
            print("Você atingiu ", maximo_iteracoes, " iterações")

            plt.plot(range(len(custos)), custos)
            plt.xlabel("Numero da Iteração")
            plt.ylabel("Custo ( J )")
            # plt.ylim(2000000, 9999999)
            # plt.xlim(0, len(custos))

            plt.savefig("GraficoCustoporiteracao.png")

            return thetas, custos

        '''Verifica se a funcao convergiu'''
        if n > 0:
            if abs(custos[n - 1] - custos[n]) < criterio_convergencia:  # convergiu (entrou minimo local)
                convergiu = True
                plt.plot(range(len(custos) - 2), custos[2:])
                plt.savefig("Grafico Custo por iteracao.png")
                print("Convergiu")
                return thetas, custos
            if custos[n - 1] < custos[n]:  # Se não estiver convergindo, então pare (ajuste learning rate)
                print("Ajuste seu learning rate")
                return thetas, custos
        n = n + 1

