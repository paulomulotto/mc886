import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as km
from sklearn.cluster import MiniBatchKMeans as minikm
from sklearn.cluster import AgglomerativeClustering as ac
from sklearn.cluster import Birch
import pandas as pd
from sklearn import metrics
import time

'''Dado um arquivo csv (name), le os dados (normalizados) presentes nela'''
def le_dados(name):

    # Le os dados do arquivo csv
    data = pd.read_csv(name)

    # Obtem os dados do csv
    data = data.values

    return data

'''Função que roda o Kmeans em sua versao classica'''
def original_kmeans(number_clusters, data):

    #Arrays com informacoes dos clusters
    clusters = [] #Array que armazena o numero de clusters por iteracao
    cost_clusters = [] #Array que armazena o custo para o calculo dos clusters para cada iteracao

    for i in range(1, number_clusters + 1):

        '''Utiliza K-means para clusterizar os dados'''
        kmeans = km(n_clusters=i, n_jobs=-1, n_init=5)
        kmeans.fit(X=data)

        '''Recupera a quantidade de clusters e o custo para cada clusterizacao'''
        clusters.append(i)
        cost_clusters.append(kmeans.inertia_)

        print("k: {} / Cost: {}".format(clusters[i - 1], cost_clusters[i - 1]))

    '''Retorna os arrays com a quantidade de clsuters por iteracao e o custo para o calculo de cada clusterizacao e 
    tambem o objeto kmeans que possui as informacoes da clusterizacao'''
    return clusters, cost_clusters, kmeans


'''Funcao que roda o Kmeans com o metodo mini-batch'''
def mini_batch_kmeans(number_clusters, data):

    # Arrays com informacoes dos clusters
    clusters = []  # Array que armazena o numero de clusters por iteracao
    cost_clusters = []  # Array que armazena o custo para o calculo dos clusters para cada iteracao

    #Variavel com o numero de clusters que melhor separa o dataset
    custo = 10000000000
    best_cluster = 0
    best_model = None

    for i in range(1, number_clusters + 1):

        '''Utiliza Mini-Batch K-means para clusterizar os dados'''
        mini_kmeans = minikm(n_clusters=i, init='random')
        mini_kmeans.fit(X=data)

        '''Recupera a quantidade de clusters e o custo para cada clusterizacao'''
        clusters.append(i)
        cost_clusters.append(mini_kmeans.inertia_)

        '''Pega o numero de clusters com o melhor resultado'''
        if(mini_kmeans.inertia_ < custo):
            custo = mini_kmeans.inertia_
            best_cluster = i
            best_model = mini_kmeans

        print("k: {} / Cost: {}".format(clusters[i - 1], cost_clusters[i - 1]))

    '''Retorna os arrays com a quantidade de clsuters por iteracao e o custo para o calculo de cada clusterizacao e 
    tambem o objeto kmeans que possui as informacoes da clusterizacao'''
    return clusters, cost_clusters, best_model, best_cluster

'''Plota o grafico do erro por numero de clusters'''
def grafico_erro_x_cluster(clusters, cost_clusters):

    plt.plot(clusters, cost_clusters)
    plt.xlabel("Número de Clusters")
    plt.ylabel("Erro para cada Cluster")
    plt.show()

'''Funcao Main'''
def main():

    '''Nome do arquivo csv com os dados (Bag of Words)'''
    name = '/home/vaoki/ec/mc886/mc886/Trabalho3/health-dataset/bags.csv'

    '''Nome do arquivo csv com os dados (Word2Vec)'''
    # name = '/home/vaoki/ec/mc886/mc886/Trabalho3/health-dataset/word2vec.csv'

    '''Recebe os dados presentes no csv'''
    data = le_dados(name=name)

    '''----------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------DEFINE BATCHES KMEANS---------------------------------------------'''
    '''----------------------------------------------------------------------------------------------------------'''
    #Numero de clusters
    number_clusters = 1000

    #Mini_Batch Kmeans = 0 / Batch Kmeans = 1 / Birch = 2
    algorithm = 1

    if(algorithm == 0):

        '''Utiliza a funcao com a implementacao Mini-Batch Kmeans'''
        # clusters, cost_clusters, mini_kmeans, best_cluster = mini_batch_kmeans(number_clusters=number_clusters, data=data)
        # labels = mini_kmeans.labels_
        # grafico_erro_x_cluster(clusters=clusters, cost_clusters=cost_clusters)

        '''Utiliza Mini-Batch K-means para clusterizar os dados'''
        mini_kmeans = minikm(n_clusters=number_clusters, init='random')
        mini_kmeans.fit(X=data)
        labels = mini_kmeans.labels_

    elif(algorithm == 1):

        '''Utiliza a funcao com a implementação classica do Kmeans'''
        # clusters, cost_clusters, kmeans = original_kmeans(number_clusters=number_clusters, data=data)
        # labels = kmeans.labels_
        # grafico_erro_x_cluster(clusters=clusters, cost_clusters=cost_clusters)

        start = time.time()
        '''Utiliza K-means para clusterizar os dados'''
        kmeans = km(n_clusters=number_clusters, n_jobs=-1, n_init=10, verbose=1)
        kmeans.fit(X=data)
        labels = kmeans.labels_
        end = time.time()

    elif(algorithm == 2):

        cluster = Birch(n_clusters=number_clusters, threshold=0.0000000000000000000001).fit(data)
        labels = cluster.labels_


    '''Calcula uma metrica de verificacao de confiabilidade com o metodo Silhouette Coeficient '''

    '''Calcula o score'''
    score = metrics.silhouette_score(data, labels=labels, metric='euclidean')
    print(score)
    print('Tempo: {}'.format((end - start)))


    # print('Melhor clusterizacao: {}'.format(best_cluster))

if __name__ == '__main__':
    main()



