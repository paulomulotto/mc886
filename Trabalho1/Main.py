from NormalEquation import *

def main():

    '''Le o arquivo csv com os dados dos diamantes'''
    pd_data = pd.read_csv("diamonds-train.csv")

    '''Aplica para cada feature com valores nao numericos, um respectivo peso, dado pelo fornecedor dos dados'''
    pd_data_value = pd_data.replace({'cut': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5},
                                     'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                                     'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6,
                                                 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1},
                                     })

    '''Chama a funcao normal_equation, que retorna o valor dos predicts de preco dos diamantes'''
    teste = normal_eaquation(pd_data_value)













    '''PARTE UTILIZADA SO PARA PLOTAR GRAFICOS. PODE IGNORAR'''
    '''Obtem a matriz y'''
    y = pd_data['price'].values

    '''Obtem a matriz x'''
    x = pd_data_value.drop(columns=['price'])['carat'].values

    # plt.scatter(x=x, y=teste, x=x, y=)
    # plt.show()


'''MAIN'''
if __name__ == '__main__':

    main()
