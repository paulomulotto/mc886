import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as km
from sklearn.cluster import MiniBatchKMeans as minikm
from sklearn.cluster import Birch
import pandas as pd
from sklearn import metrics
import time
from sklearn.decomposition import PCA
from pyclustering.cluster import kmeans
from sklearn.metrics import davies_bouldin_score

'''Dado um arquivo csv (name), le os dados (normalizados) presentes nela'''
def le_dados(name):

    # Le os dados do arquivo csv
    data = pd.read_csv(name, header=None)

    # Obtem os dados do csv
    data = data.values

    return data

'''Função que roda o Kmeans em um range (1 - number_clusters) e escolhe a melhor clusterizacao, escolhida de acordo
com a menor inertia'''
def best_original_kmeans(number_clusters, data):

    #Arrays com informacoes dos clusters
    clusters = [] #Array que armazena o numero de clusters por iteracao
    cost_clusters = [] #Array que armazena o custo para o calculo dos clusters para cada iteracao

    # Variavel com o numero de clusters que melhor separa o dataset
    custo = 10000000000
    best_cluster = 0
    best_model = None

    for i in range(1, number_clusters + 1):

        '''Utiliza K-means para clusterizar os dados'''
        kmeans = km(n_clusters=i, n_jobs=-1, n_init=10)
        kmeans.fit(X=data)

        '''Recupera a quantidade de clusters e o custo para cada clusterizacao'''
        clusters.append(i)
        cost_clusters.append(kmeans.inertia_)

        '''Pega o numero de clusters com o melhor resultado'''
        if(kmeans.inertia_ < custo):
            custo = kmeans.inertia_
            best_cluster = i
            best_model = kmeans

        print("k: {} / Cost: {}".format(clusters[i - 1], cost_clusters[i - 1]))

    '''Retorna os arrays com a quantidade de clsuters por iteracao e o custo para o calculo de cada clusterizacao e 
    tambem o objeto kmeans que possui as informacoes da clusterizacao'''
    return clusters, cost_clusters, best_model, best_cluster

'''Função que roda o Mini Batch-Kmeans em um range (1 - number_clusters) e escolhe a melhor clusterizacao, escolhida de acordo
com a menor inertia'''
def best_mini_batch_kmeans(number_clusters, data):

    # Arrays com informacoes dos clusters
    clusters = []  # Array que armazena o numero de clusters por iteracao
    cost_clusters = []  # Array que armazena o custo para o calculo dos clusters para cada iteracao

    #Variavel com o numero de clusters que melhor separa o dataset
    custo = 10000000000
    best_cluster = 0
    best_model = None

    for i in range(1, number_clusters + 1):

        '''Utiliza Mini-Batch K-means para clusterizar os dados'''
        mini_kmeans = minikm(n_clusters=i, init_size=(i + 1))
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


'''Funcao que armazena, para cada cluster, os twites pertencentes a ele'''
def samples_per_clusters(number_clusters, labels):

    samples_per_clusters = []

    for i in range(0, number_clusters):
        aux = [(j + 1) for j in np.where(labels == (i + 1))[0].tolist()]
        samples_per_clusters.append(aux)

    return samples_per_clusters