import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as km
import pandas as pd

'''Dado um arquivo csv (name), le os dados (normalizados) presentes nela'''
def le_dados(name):

    # Le os dados do arquivo csv
    data = pd.read_csv(name)

    # Obtem os dados do csv
    data = data.values

    #Normaliza os dados
    data = data / len(data[0])

    return data

'''Nome do arquivo csv com os dados'''
name = '/home/vaoki/ec/mc886/mc886/Trabalho3/health-dataset/bags.csv'

'''Recebe os dados presentes no csv'''
data = le_dados(name=name)

'''Utiliza K-means para clusterizar os dados'''
kmeans = km(n_clusters=10)
kmeans.fit(X=data)

# print(kmeans.cluster_centers_)
elements_per_clusters = np.zeros(10, dtype=np.int)

elements_per_clusters[0] = np.sum(kmeans.labels_ == 0)
elements_per_clusters[1] = np.sum(kmeans.labels_ == 1)
elements_per_clusters[2] = np.sum(kmeans.labels_ == 2)
elements_per_clusters[3] = np.sum(kmeans.labels_ == 3)
elements_per_clusters[4] = np.sum(kmeans.labels_ == 4)
elements_per_clusters[5] = np.sum(kmeans.labels_ == 5)
elements_per_clusters[6] = np.sum(kmeans.labels_ == 6)
elements_per_clusters[7] = np.sum(kmeans.labels_ == 7)
elements_per_clusters[8] = np.sum(kmeans.labels_ == 8)
elements_per_clusters[9] = np.sum(kmeans.labels_ == 9)

print(elements_per_clusters)