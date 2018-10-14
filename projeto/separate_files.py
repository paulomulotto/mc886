import pandas as pd
import os, shutil

df = pd.read_csv('validacao.csv')
# Name of Breeds
breeds = df['breed'].unique()
# Caminho da Pasta Principal
os.chdir("/home/paulo/Documents/Faculdade/2s-208/MC886/mc886/Projeto/Teste Udemy/all/")
# Separa as fotos em pastas por raça de cachorro
for breed in breeds:
    list_id = df[df.breed == breed]
    folderName = breed
    pasta_destino = os.path.join('validacao', folderName)
    print(folderName)
    if not os.path.exists(pasta_destino):
        os.mkdir(pasta_destino)
    for id in list_id.id:
        imagem = id + ".jpg"
        # Trocar quando necessário
        pasta_origem = os.path.join('train', imagem)
        shutil.copy(pasta_origem, os.path.join(pasta_destino, imagem))