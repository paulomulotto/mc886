'''Plot dos pontos num gráfico'''
'''x = np_data[:, 0]
y = np_data[:, 9]

plt.scatter(x, y, s=10)
plt.show()'''

a = 4
b = 5
c = a + b

print(c)
print(a)

 '''Cria todas as permutacoes para poder modificar as features '''
    expo = np.array(list(it.product([1, 2, 3], repeat=10)))


    '''Variavel que armazena o erro da validacao'''
    erro_final = 100000000000000000000000000000000000000000000000000000000000000000000000000000000

    for i in range(0, len(expo)):

        x_treino = treino.drop(columns=['price']).values
        x_validacao = validacao.drop(columns=['price']).values

        x_treino[:, 1] = x_treino[:, 1] ** int(expo[i, 0])
        x_treino[:, 2] = x_treino[:, 2] ** int(expo[i, 1])
        x_treino[:, 3] = x_treino[:, 3] ** int(expo[i, 2])
        x_treino[:, 4] = x_treino[:, 4] ** int(expo[i, 3])
        x_treino[:, 5] = x_treino[:, 5] ** int(expo[i, 4])
        x_treino[:, 6] = x_treino[:, 6] ** int(expo[i, 5])
        x_treino[:, 7] = x_treino[:, 7] ** int(expo[i, 6])
        x_treino[:, 8] = x_treino[:, 8] ** int(expo[i, 7])
        x_treino[:, 9] = x_treino[:, 9] ** int(expo[i, 8])
        x_treino[:, 10] = x_treino[:, 10] ** int(expo[i, 9])

        x_validacao[:, 1] = x_validacao[:, 1] ** int(expo[i, 0])
        x_validacao[:, 2] = x_validacao[:, 2] ** int(expo[i, 1])
        x_validacao[:, 3] = x_validacao[:, 3] ** int(expo[i, 2])
        x_validacao[:, 4] = x_validacao[:, 4] ** int(expo[i, 3])
        x_validacao[:, 5] = x_validacao[:, 5] ** int(expo[i, 4])
        x_validacao[:, 6] = x_validacao[:, 6] ** int(expo[i, 5])
        x_validacao[:, 7] = x_validacao[:, 7] ** int(expo[i, 6])
        x_validacao[:, 8] = x_validacao[:, 8] ** int(expo[i, 7])
        x_validacao[:, 9] = x_validacao[:, 9] ** int(expo[i, 8])
        x_validacao[:, 10] = x_validacao[:, 10] ** int(expo[i, 9])

        '''Obtem os thetas com a funcao de normal equation, a partir do dataset de treino'''
        thetas_normal = normal_eaquation(x_treino, y_treino)

        '''Calcula o erro (J) a partir dos thetas obtidos, e do dataset de validacao'''

        erro_validacao = erro(thetas_normal, x_validacao, y_validacao)
        erro_treino = erro(thetas_normal, x_treino, y_treino)

        print('----------------------------------------------------------------------------------------------')
        print('{} / {}'.format(i, len(expo)))
        print('Erro treino: {}'.format(erro_treino))
        print('Erro validacao: {}'.format(erro_validacao))
        print(expo[i])

        if(erro_validacao < erro_final):
            erro_final = erro_validacao
            modelo = expo[i]

            print('Erro final: {}'.format(erro_final))
            print('Modelo: {}'.format(modelo))

        print('----------------------------------------------------------------------------------------------\n')

    print(erro_final)
    print(modelo)
