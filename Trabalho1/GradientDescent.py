# coding: utf-8

# In[1]:


# Gradient Descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Calcula J(thetas)
def calc_custo(y, x, thetas):
    
    # custo = sum((x.dot(theta))-y)**2/(2*len(y))
    
    # somatorio = 0
    # for i in range(0,y.shape[0]):
    #     somatorio += np.power(sum(x[i]*thetas[:]) - y[i], 2)
        
    # custo = somatorio / (2*y.shape[0])
    custo = sum((x.dot(thetas)-y)**2)/(2*len(y))
    #print(custo)
    return custo



def gradient_descent(x, y, criterio_convergencia, alfa, maximo_iteracoes):

    # '''Obtem a matriz y'''
    # y = pd_data_value['price'].values

    # '''Obtem a matriz x'''
    # x = pd_data_value.drop(columns=['price']).values
    
    
    # Definição das constantes básicas
    #thetas = np.random.rand(x.shape[1])
    thetas = np.random.rand(x.shape[1])
    
    

    m = y.shape[0]
    
    n = 0 #numero máximo de iterações
    custos = np.zeros(maximo_iteracoes)

    convergiu = False
    while not convergiu:
        predicao = x.dot(thetas)
        erro = predicao - y
        derivada = x.T.dot(erro) / m
        thetas = thetas - alfa * derivada
        custos[n] = calc_custo(y, x, thetas)

        if n == maximo_iteracoes - 1:
            print("Você atingiu ", maximo_iteracoes, " iterações")

            plt.plot(range(len(custos)), custos)
            # plt.ylim(2000000, 9999999)
            # plt.xlim(0, len(custos))

            plt.show()

            return thetas, custos
        if n > 0:
            if abs(custos[n - 1] - custos[n]) < criterio_convergencia:  # convergiu (entrou minimo local)
                convergiu = True
                plt.plot(range(len(custos) - 2), custos[2:])
                plt.show()
                print("Convergiu")
                return thetas, custos
            if custos[n - 1] < custos[n]:  # Se não estiver convergindo, então pare (ajuste learning rate)
                print("Ajuste seu learning rate")
                return thetas, custos
        n = n + 1








    '''convergiu = False
    while not convergiu:
        predicao = x.dot(thetas)
        erro = predicao - y
        derivada = x.T.dot(erro) / m
        thetas = thetas - alfa*derivada
        custos[n] = calc_custo(y,x,thetas)
        
        if n == maximo_iteracoes-1:
            print("Você atingiu ", maximo_iteracoes, " iterações")
            
            
            plt.plot(range(len(custos)), custos)
            # plt.ylim(2000000, 9999999)
            # plt.xlim(0, len(custos))
            
            plt.show()
            
            return thetas, custos
        if n > 0:
            if abs(custos[n-1]-custos[n]) < criterio_convergencia: #convergiu (entrou minimo local)
                convergiu = True
                plt.plot(range(len(custos)-2), custos[2:])
                plt.show()
                print("Convergiu")
                return thetas, custos
            if custos[n-1] < custos[n]: #Se não estiver convergindo, então pare (ajuste learning rate)
                print("Ajuste seu learning rate")
                return thetas, custos
        n = n+1'''

    

        #Calcula os thetas para uma rodada
        # for theta in range(0, thetas.shape[0]):
        #     #calcula a derivada (somatorio / m)  para thetaX
        #     derivada = 0
        #     for i in range(0,y.shape[0]):
        #         derivada += (sum(x[i]*thetas[:]) - y[i])*x[i,theta]

        #     # 1/m * SUM => derivada
        #     derivada = derivada / y.shape[0]


        #     # Aplicacao do learning rate
        #     termo = derivada * alfa
        #     #atualiza os novos thetas de acordo com os antigos (simultaneously update for every j = 0,1,2,...)
        #     novos_thetas[theta] = thetas[theta] - termo

        # #calcula o custo J
        # custo_novo = calc_custo(y, x, novos_thetas)
        # print(custo_novo)
        
        #Verifica se convergiu
        # if abs(custo_velho-custo_novo) <= criterio_convergencia: #convergiu
        #     convergiu = True
        #     print("Convergiu")
        #     return thetas

        # custo_velho = custo_novo
        # n = n+1
        # thetas = novos_thetas

        # if n == maximo_iteracoes:
        #     return thetas

    
# def main():

#     '''Le o arquivo csv com os dados dos diamantes'''
#     pd_data = pd.read_csv("treino.csv")

#     '''Realiza a separacao das features cujo valor, nao sao numericos'''
#     pd_data_value = pd.get_dummies(pd_data)

#     thetas, _ = gradient_descent(pd_data_value, 0.5, 0.000000002852, 4000)
    
#     print(thetas)


# '''MAIN'''
# if __name__ == '__main__':

#     main()