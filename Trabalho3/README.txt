Regressão Logística e Redes Neurais

Versão Python: 3.5.2
Bibliotecas utilizadas:
- pandas: versão 0.23.4
- numpy: versão 1.15.1
- matplotlib: versão 3.0.0
- sklearn: versão 0.20.0

Execução do programa: para executar o programa, basta utilizar o comando 'python3 Main.py' no terminal. O arquivo Clustering.py que realizam os métodos desses modelos devem estar na mesma pasta que a Main.py e são importados direto em Main.py.

Configuração do programa: Em Main.py algumas variáveis podem ser configuradas para que os modelos sejam executados:

Em Main.py
 - Na linha 21 'number_clusters' define o número de clusters utilizados para os métodos Kmeans e Birch
 - Na linha 24 'algorithm' define qual algoritmo vai ser usado
    0: Mini_Batch Kmeans
    1: Batch Kmeans
    2: Birch
    3: DBSCAN

 - Na linha 27 'pca' define se irá utiliza PCA ou não (True or False)

 - Na linha 34  'n_components' define o número de componentes para manter.

 - Na linha 116 'max_dist' define a máxima distância que um ponto deve estar para ser considerado pertencente a um possível cluster.
 - Na linha 118 'min_samples' define o número minimo de tweets que devem ser considerados dentro da distância máxima 'max_dist' 

 Após executar o programa será gerado na pasta tweets um arquivo por cluster com os tweets referentes ao cluster além de um arquivo info.txt com informações básicas dos clusters e o score calculado pelo 'silhouette_score'.

 Data set deve seguir a seguinte estrutura de diretórios:
 | Trabalho 3
 |- Main.py
 |- Clustering.py
 |- health.txt
 |- tweets (diretório vazio)
 |- health-dataset
 |  |-bags.csv
 |  |-health.txt (* SEM A PRIMEIRA LINHA - APENAS COM OS TWEETS)
