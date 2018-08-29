
# coding: utf-8

# In[1]:


# Gradient Descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Calcula J(thetas)
def calc_custo(y, x, thetas):
    somatorio = 0
    for i in range(0,y.shape[0]):
        somatorio += np.power(sum(x[i]*thetas[:]) - y[i], 2)
        
    custo = somatorio / (2*y.shape[0])
    return custo



def gradient_descent(pd_data_value, criterio_convergencia, alfa, maximo_iteracoes):
    '''Adiciona a coluna x0 com valores 1 para a realizacao da multiplicacao de matrizes'''
    pd_data_value.insert(0, 'x0', 1)

    '''Obtem a matriz y'''
    y = pd_data_value['price'].values

    '''Obtem a matriz x'''
    x = pd_data_value.drop(columns=['price']).values
    
    
    # Definição das constantes básicas
    thetas = np.zeros(x.shape[1])
    novos_thetas = np.zeros(x.shape[1])
    custo_velho = calc_custo(y, x, thetas)
    
    convergiu = False
    n = 0 #numero máximo de iterações
    while not convergiu:
        #Calcula os thetas para uma rodada
        for theta in range(0, thetas.shape[0]):
            #calcula a derivada (somatorio / m)  para thetaX
            derivada = 0
            for i in range(0,y.shape[0]):
                derivada += (sum(x[i]*thetas[:]) - y[i])*x[i,theta]

            # 1/m * SUM => derivada
            derivada = derivada / y.shape[0]


            # Aplicacao do learning rate
            termo = derivada * alfa
            #atualiza os novos thetas de acordo com os antigos (simultaneously update for every j = 0,1,2,...)
            novos_thetas[theta] = thetas[theta] - termo

        #calcula o custo J
        custo_novo = calc_custo(y, x, novos_thetas)
        print(custo_novo)
        
        #Verifica se convergiu
        if abs(custo_velho-custo_novo) <= criterio_convergencia: #convergiu
            convergiu = True
            print("Convergiu")
            return thetas

        custo_velho = custo_novo
        n = n+1
        thetas = novos_thetas

        if n == maximo_iteracoes:
            return thetas

    
def main():

    '''Le o arquivo csv com os dados dos diamantes'''
    pd_data = pd.read_csv("diamonds-test.csv")

    '''Realiza a separacao das features cujo valor, nao sao numericos'''
    pd_data_value = pd.get_dummies(pd_data)

    thetas = gradient_descent(pd_data_value, 100, 0.0001, 1000)
    print(thetas)
    



'''MAIN'''
if __name__ == '__main__':

    main()

