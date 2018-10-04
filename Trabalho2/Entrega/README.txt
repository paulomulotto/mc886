Regressão Logística e Redes Neurais

Versão Python: 3.5.2
Bibliotecas utilizadas:
- pandas: versão 0.23.4
- numpy: versão 1.15.1

Execução do programa: para excutar o programa, basta utilizar o comando 'python Main.py' no terminal. Os arquivos NerualNetwork.py, LogisticRegression.py e SoftmaxRegression.py que realizam os métodos desses modelos devem estar na mesma pasta que Main.py e são importados direto em Main.py. 

Configuração do programa: Em Main.py algumas variáveis podem ser configuradas para que os modelos sejam executados. Por padrão, o programa está setado para rodar as Regressões Logísticas: 

- Na linha 266 de Main.py, a variável tipo_treinamento decide se será utilizado Regressão Logística caso seja setada como 1 ou Redes Neurais, caso seja setado como 2. 
- Caso o modelo de Redes Neurais seja escolhido, na linha 271 a variável qtd_camadas decide quantas camdas internas serão utilizadas, podendo ser 1 ou 2. Por padrão o programa irá executar para 1 camada com a função sigmoid. 
- Na linha 313 de Main.py, caso o modelo de Regressão Logística tenha sido escolhido, as variáveis learning_rate e iterations podem ser setadas com diferentes valores para gerar diferentes resultados no modelo.
- Na linha 350 de Main.py, caso o modelo de Redes Neurais seja escolhido, as variáveis activation_function, learning_rate, iterations e num_neurons setam respectivamente, a função de ativação que será utilizada (1: Sigmoid, 2: Tanh e 3: ReLu), o learning rate que será utilizado, o número de iterações e o número de neurônios.

Data set: na mesma pasta onde se encontra os arquivos .py, deve estar uma pasta com o nome 'fashion-mnist-dataset' e dentro desta pasta devem estar os arquivos 'fashion-mnist_test.csv' e 'fashion-mnist_train.csv', que são os datasets. O programa se encarrega de separar o conjunto de treino em treino (80%) e validação (20%).
