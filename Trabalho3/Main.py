# -*- coding: utf-8 -*-
from Clustering import *


'''Funcao Main'''
def main():

    '''Nome do arquivo csv com os dados (Bag of Words)'''
    name = 'bags.csv'

    '''Nome do arquivo csv com os dados (Word2Vec)'''
    # name = '/home/vaoki/ec/mc886/mc886/Trabalho3/health-dataset/word2vec.csv'

    '''Recebe os dados presentes no csv'''
    data = le_dados(name=name)

    '''----------------------------------------------------------------------------------------------------------'''
    '''------------------------------------------DEFINE ALGORITMOS-----------------------------------------------'''
    '''----------------------------------------------------------------------------------------------------------'''
    #Numero de clusters
    number_clusters = 100

    #Mini_Batch Kmeans = 0 / Batch Kmeans = 1 / Birch = 2
    algorithm = 3

    #Utiliza PCA
    pca = False

    if(pca == True):

        print('PCA')    

        '''Utilzia o PCA para redimensionar os dados'''
        data = PCA(n_components=0.95, svd_solver='full').fit_transform(data)

    print(data.shape)

    if(algorithm == 0):

        print('Mini-Batch Kmeans')

        '''Utiliza Mini-Batch K-means para clusterizar os dados'''
        # mini_kmeans = minikm(n_clusters=number_clusters, init_size=number_clusters, verbose=True).fit(X=data)
        clusters, cost_clusters, best_model, best_cluster = best_mini_batch_kmeans(number_clusters=number_clusters,
                                                                                   data=data)
        labels = best_model.labels_

        print("Número de clusters com o melhor resultado: {}".format(best_cluster))

        '''Calcula uma metrica de verificacao de confiabilidade com o metodo Silhouette Coeficient '''

        '''Calcula o score'''
        score = metrics.silhouette_score(data, labels=labels, metric='euclidean')
        print(score)

    elif(algorithm == 1):

        print('Kmeans')

        '''Utiliza K-means para clusterizar os dados'''
        kmeans = km(n_clusters=number_clusters, n_jobs=-1, n_init=20, verbose=1, tol=0.001).fit(X=data)
        labels = kmeans.labels_

        # clusters, cost_clusters, best_model, best_cluster = best_original_kmeans(number_clusters, data)
        # labels = best_model.labels_
        # print("Número de clusters com o melhor resultado: {}".format(best_cluster))

        '''Calcula uma metrica de verificacao de confiabilidade com o metodo Silhouette Coeficient '''

        '''Calcula o score'''
        score = metrics.silhouette_score(data, labels=labels, metric='euclidean')
        print(score)


    elif(algorithm == 2):

        print('Birch')

        cluster = Birch(n_clusters=number_clusters, threshold=0.5).fit(data)
        labels = cluster.labels_

        '''Calcula uma metrica de verificacao de confiabilidade com o metodo Silhouette Coeficient '''

        '''Calcula o score'''
        score = metrics.silhouette_score(data, labels=labels, metric='euclidean')
        print(score)


    elif (algorithm == 3):

        print('DBSCAN')

        clusters = DBSCAN(eps=0.99, min_samples=4, metric='euclidean', n_jobs=-1, p='float').fit(data)

        # labels = clusters.labels_

        informacoes(clusters.labels_, data)

        # '''Calcula uma metrica de verificacao de confiabilidade com o metodo Silhouette Coeficient '''

        # '''Calcula o score'''
        # score = metrics.silhouette_score(data, labels=labels, metric='euclidean')
        # print(score)

        #Teste feito:
        # Score: -0,11
        # t = cluster.DBSCAN(eps=20, min_samples=5, metric='euclidean', n_jobs=-1, p='float').fit(data)
        # np.where(t.labels_==[0,1,3,4,5,6,7,8,9,10][0])
        # para ver tamanho dos clusters: len(np.where(t.labels_==[-1,0,1,3,4,5,6,7,8,9,10][0])[0])
        # max(t.labels_)

        



if __name__ == '__main__':
    main()



