# -*- coding: utf-8 -*-
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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
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
        kmeans = km(n_clusters=i, n_jobs=-1, n_init=20, verbose=False, tol=0.000001)
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
        mini_kmeans = minikm(n_clusters=i, init_size=(i + 1), tol=0.000001, n_init=100, reassignment_ratio=1)
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


''' Imprime informações sobre os resultados '''
def informacoes(labels, data):
    
    print("Número de Clusters:")
    print(max(labels))

    '''' Imprime o número de elementos por cluster '''
    
    ''' Separa o arquivo health.txt em linhas '''
    lines = [line.rstrip('\n') for line in open('health-dataset/health.txt')]

    print("Imprimindo Tweets em arquivos separados")
    
    '''Cria um arquivo com informações do teste '''
    nome_arquivo_info = "tweets/info.txt"
    arquivo_info = open(nome_arquivo_info,"w")
    arquivo_info.write("Número de elementos por clusters ( " + str(max(labels)) + " ):\n")

    for i in range(-1,max(labels)):
        tam_cluster = len(np.where(labels==i)[0])
        print("cluster" + str(i) + " com " + str(tam_cluster) + " tweets.")
        
        arquivo_info.write("cluster" + str(i) + " com " + str(tam_cluster) + " tweets.\n")

        ''' Imprime os tweets de cada cluster em um arquivo separado para cada cluster'''
        
        nome_arquivo = "tweets/cluster" + str(i) + ".txt"
        file = open(nome_arquivo,"w")
        file.write(nome_arquivo + "\n")
        file.write("Numero de tweets: " + str(tam_cluster) + "\n\n")
        file.write("Tweets:\n")

        ''' Verifica qual tweet pertence ao cluster e insere no arquivo do cluster a qual o tweet pertence. '''
        for j in range(0, len(lines)):
            if labels[j] == i:
                file.write(lines[j] + "\n")    
        file.close()


    '''Calcula uma metrica de verificacao de confiabilidade com o metodo Silhouette Coeficient '''

    '''Calcula o score'''
    score = metrics.silhouette_score(data, labels=labels, metric='euclidean')
    print(score)
    arquivo_info.write(str(score))
    arquivo_info.close()





'''Funcao que armazena, para cada cluster, os twites pertencentes a ele'''
def samples_per_clusters(number_clusters, labels):

    samples_per_clusters = []

    for i in range(0, number_clusters):
        aux = [(j + 1) for j in np.where(labels == (i + 1))[0].tolist()]
        samples_per_clusters.append(aux)

    return samples_per_clusters
