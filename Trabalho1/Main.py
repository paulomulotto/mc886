from NormalEquation import *


'''Funcao que realiza as operacoes de alteracao dos dados do dataset'''
def dados(pd_data):

    '''Aplica para cada feature com valores nao numericos, um respectivo peso, dado pelo fornecedor dos dados'''
    pd_data_value = pd_data.replace({'cut': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5},
                                     'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                                     'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6,
                                                 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1},
                                     })

    '''Adiciona a coluna x0 com valores 1 para a realizacao da multiplicacao de matrizes'''
    pd_data_value.insert(0, 'x0', 1)

    '''Separa o conjunto de treino em treino e validacao'''
    msk = np.random.rand(len(pd_data_value)) < 0.7

    treino = pd_data_value[msk]
    treino.to_csv('treino.csv')

    validacao = pd_data_value[~msk]
    validacao.to_csv('validacao.csv')

def erro(thetas, x, y):

    erro = 0

    '''Calcula o erro J'''
    for i in range(0, len(x)):
        erro += (np.sum((thetas[:] * x[i, :])) - y[i])**2

    print(erro / (2 * len(x)))

'''Calcula os predicts e plota o grafico de comoparacao entre predict e target'''
def predicts(thetas, x, y_target):
    m = x.shape[0]
    y_predicts = np.zeros(m)
    for i in range(m):
        y_predicts[i] = np.sum(thetas[:] * x[i])

    plt.xlabel("Diamante")
    plt.ylabel("Valor")
    plt.plot(range(0,m), y_predicts, 'r--', range(0,m), y_target, 'g--')
    plt.show()

def main():

    '''Le o arquivo csv com os dados dos diamantes(treino)'''
    treino = pd.read_csv("treino.csv")

    '''Le o arquivo csv com os dados dos diamantes(validacao)'''
    validacao = pd.read_csv("validacao.csv")


    '''Obtem a matriz y (target) (treino)'''
    y_treino = treino['price'].values

    '''Obtem a matriz x (treino)'''
    x_treino = treino.drop(columns=['price']).values

    '''Obtem a matriz y (target) (validacao)'''
    y_validacao = validacao['price'].values

    '''Obtem a matriz x (validacao)'''
    x_validacao = validacao.drop(columns=['price']).values


    '''NORMAL EQUATION'''

    '''Obtem os thetas com a funcao de normal equation, a partir do dataset de treino'''
    thetas_normal = normal_eaquation(x_treino, y_treino)

    '''Calcula o erro (J) a partir dos thetas obtidos, e do dataset de validacao'''
    erro(thetas_normal, x_validacao, y_validacao)
    erro(thetas_normal, x_treino, y_treino)

    predicts(thetas_normal, x_treino, y_treino)


'''MAIN'''
if __name__ == '__main__':

    main()
