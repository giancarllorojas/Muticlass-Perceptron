import matplotlib.pyplot as plt
from random import shuffle
from random import seed
import numpy as np
import sys

if len(sys.argv) < 4:
    print("Execute python3 iris.py loops_de_treino semente_randomiza plotar_sim_ou_nao")
    sys.exit()

num_treino = int(sys.argv[1])
semente    = int(sys.argv[2])
plotar     = int(sys.argv[3])

# retorna o sinal de um numero, usado para agrupar os tipos de iris
# Usei essa função só no inicio, depois adaptei para usar o argmax para classificao nao-binaria
def sinal(valor):
    if valor < 0:
        return -1
    elif valor > 0:
        return 1
    else:
        return 0

# Faz uma previsao usando os pesos
def preve(entrada, pesos):
    #multiplica a entrada pelos pesos do modelo
    #print("######")
    #print(entrada)
    #print(pesos)
    prev_val = np.dot(pesos, entrada)
    #print("####")

    #print(prev_val)

    # retorna o maior valor encontrado como resultado da classificação
    return np.argmax(prev_val)

def plot_pesos(pesos, figure):
    plt.figure(figure)
    ww = pesos
    ww1 = [ww[1],-ww[0]]
    ww2 = [-ww[1],ww[0]]
    plt.plot([ww1[0], ww2[0]],[ww1[1], ww2[1]])

def classes(numero):
    if numero == 0:
        return 'Iris-setosa'
    if numero == 1:
        return 'Iris-versicolor'
    if numero == 2:
        return 'Iris-virginica'

# Treina o dataset de entrada
def treina(data, quantidade):
    # pesos iniciais como 0.0
    pesos = np.zeros((3,4))

    for i in range(quantidade):
        erro_total = 0
        for entrada in data:
            val = entrada[0]
            res = entrada[1]
            
            previsao = preve(val, pesos)
            if(previsao != res):
                pesos = atualiza_pesos(pesos, val, previsao, res)
                erro_total += 1
        print("Loop de treino: " + str(i) + " - Erro: " + str(erro_total))
    return pesos

def atualiza_pesos(pesos, entrada, previsto, resultado):
    #print(pesos, entrada)
    peso_class = pesos[previsto]

    pesos[resultado] += entrada
    pesos[previsto] -= entrada

    return pesos

def formata_dados(linha):
    d = linha.strip().split(',')

    valores = [float(i) for i in d[:-1]]
    arg1 = valores[0] + valores[1]**2
    arg2 = valores[2] + valores[3]**3
    valores[1] = valores[1]**2
    valores[3] = valores[3]**3

    #print(arg1, arg2)

    if d[4] == 'Iris-setosa':
        d[4] = 0
        plt.figure(1)
        plt.plot([1,2,3,4], d[:-1], 'r')
        plt.figure(2)
        plt.plot(arg1, arg2, 'ro')
    elif d[4] == 'Iris-versicolor':
        d[4] = 1
        plt.figure(1)
        plt.plot([1,2,3,4], d[:-1], 'g')
        plt.figure(2)
        plt.plot(arg1, arg2, 'go')
    elif d[4] == 'Iris-virginica':
        d[4] = 2
        plt.figure(1)
        plt.plot([1,2,3,4], d[:-1], 'b')
        plt.figure(2)
        plt.plot(arg1, arg2, 'bo')

    dados = np.array([np.array(valores), d[4]])
    #print(dados)
    return dados


iris_dados = []
with open("IrisTreino.txt", "r") as i:
    linha_atual = 0
    
    # use as primeiras 130 linhas para treinar os pesos
    for linha in i:
        dados_linha = formata_dados(linha)
        iris_dados.append(dados_linha)
    
    # Randomizando o dataset para pegar uma quantidade equivalente de dados de cada
    seed(semente)
    shuffle(iris_dados)
    pesos = treina(iris_dados, num_treino)
    plt.figure(2)
    

if plotar == 1:
    plt.show()

with open("IrisValidacao.txt", "r") as i:
    # faz a predição
    corretos = 0
    errados  = 0
    for linha in i:
        dados_linha = formata_dados(linha)
        valores   = dados_linha[0]
        resultado = dados_linha[1]

        previsao = preve(valores, pesos)
        if previsao == resultado:
            corretos += 1
        else:
            errados += 1
            print("Erro: ", linha, "Previsto: ", classes(previsao), "Correto: ", classes(resultado))
    print("Analisados " + str(corretos + errados) + " entradas com taxa de acerto de " + str((corretos/(corretos+errados))*100) + "%")