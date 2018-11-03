from Clustering import *

'''Funcao Main'''
def main():

    '''Nome do arquivo csv com os dados (Bag of Words)'''
    name = '/home/vaoki/ec/mc886/mc886/Trabalho3/health-dataset/bags.csv'

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
    algorithm = 1

    #Utiliza PCA
    pca = True

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


if __name__ == '__main__':
    main()



