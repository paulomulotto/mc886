from NormalEquation import *
import itertools as it
from GradientDescent import *
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from datetime import datetime

'''Calcula o erro J'''
def erro(thetas, x, y, tipo):

    erro = np.sum(((x.dot(thetas)) - y)**2)/(2*len(y))

    print('----------------------------------------------------------------------------------------------')
    print('Erro {}: {}'.format(tipo, erro))
    print('----------------------------------------------------------------------------------------------')

    return erro

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

    '''Le o arquivo csv com os dados dos diamantes(teste)'''
    teste = pd.read_csv("teste.csv")

    '''Obtem a matriz y (target) (treino)'''
    y_treino = treino['price'].values

    '''Obtem a matriz x (treino)'''
    x_treino = treino.drop(columns=['price']).values

    '''Obtem a matriz y (target) (validacao)'''
    y_validacao = validacao['price'].values

    '''Obtem a matriz x (validacao)'''
    x_validacao = validacao.drop(columns=['price']).values

    '''Obtem a matriz y (target) (teste)'''
    y_teste = teste['price'].values

    '''Obtem a matriz x (teste)'''
    x_teste = teste.drop(columns=['price']).values

    x_treino[:, 1] = x_treino[:, 1] ** 1
    x_treino[:, 2] = x_treino[:, 2] ** 1
    x_treino[:, 3] = x_treino[:, 3] ** 1
    x_treino[:, 4] = x_treino[:, 4] ** 1
    x_treino[:, 5] = x_treino[:, 5] ** 1
    x_treino[:, 6] = x_treino[:, 6] ** 3
    x_treino[:, 7] = x_treino[:, 7] ** 2
    x_treino[:, 8] = x_treino[:, 8] ** 1
    x_treino[:, 9] = x_treino[:, 9] ** 1
    x_treino[:, 10] = x_treino[:, 10] ** 3

    x_validacao[:, 1] = x_validacao[:, 1] ** 1
    x_validacao[:, 2] = x_validacao[:, 2] ** 1
    x_validacao[:, 3] = x_validacao[:, 3] ** 1
    x_validacao[:, 4] = x_validacao[:, 4] ** 1
    x_validacao[:, 5] = x_validacao[:, 5] ** 1
    x_validacao[:, 6] = x_validacao[:, 6] ** 3
    x_validacao[:, 7] = x_validacao[:, 7] ** 2
    x_validacao[:, 8] = x_validacao[:, 8] ** 1
    x_validacao[:, 9] = x_validacao[:, 9] ** 1
    x_validacao[:, 10] = x_validacao[:, 10] ** 3

    x_teste[:, 1] = x_teste[:, 1] ** 1
    x_teste[:, 2] = x_teste[:, 2] ** 1
    x_teste[:, 3] = x_teste[:, 3] ** 1
    x_teste[:, 4] = x_teste[:, 4] ** 1
    x_teste[:, 5] = x_teste[:, 5] ** 1
    x_teste[:, 6] = x_teste[:, 6] ** 3
    x_teste[:, 7] = x_teste[:, 7] ** 2
    x_teste[:, 8] = x_teste[:, 8] ** 1
    x_teste[:, 9] = x_teste[:, 9] ** 1
    x_teste[:, 10] = x_teste[:, 10] ** 3

    '''Obtem os thetas com a funcao de normal equation, a partir do dataset de treino'''
    thetas_normal = normal_eaquation(x_treino, y_treino)

    '''Obtem os thetas com a funcao de gradient descent, a partir do dataset de treino'''
    thetas_gradient, custos = gradient_descent(x_treino, y_treino, 0.5, 0.0000000000536, 300000)

    '''Obtem os thetas com a funcao de SGDRegressor, a partir do dataset de treino'''
    clf = linear_model.SGDRegressor(learning_rate="constant", eta0=0.0000000000536, max_iter=300000)
    clf = clf.fit(x_treino, y_treino)


    '''Calcula o erro (J) a partir dos thetas obtidos, e do dataset de validacao e teste'''
    print('----------------------------------------------------------------------------------------------')
    print('Normal Equation')
    print('----------------------------------------------------------------------------------------------')
    erro_treino = erro(thetas_normal, x_treino, y_treino, 'Treino')
    erro_validacao = erro(thetas_normal, x_validacao, y_validacao, 'Validacao')
    erro_teste = erro(thetas_normal, x_teste, y_teste, 'Teste')

    print('\n---------------------------------------------------------------------------------------------')
    print('Gradient Descent')
    print('----------------------------------------------------------------------------------------------')
    erro_gradient_treino = erro(thetas_gradient, x_treino, y_treino, 'Treino')
    erro_gradient_validacao = erro(thetas_gradient, x_validacao, y_validacao, 'Validacao')
    erro_gradient_teste = erro(thetas_gradient, x_teste, y_teste, 'Teste')

    print('\n----------------------------------------------------------------------------------------------')
    print('SGDRegressor')
    print('----------------------------------------------------------------------------------------------')
    erro_SGD_library = erro(clf.coef_, x_treino, y_treino, 'Treino')
    erro_SGD_library = erro(clf.coef_, x_validacao, y_validacao, 'Validacao')
    erro_SGD_library = erro(clf.coef_, x_teste, y_teste, 'Teste')




'''MAIN'''
if __name__ == '__main__':

    main()
