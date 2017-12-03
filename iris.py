import matplotlib.pyplot as plt
from random import shuffle
from random import seed
import numpy as np

# retorna o sinal de um numero, usado para agrupar os tipos de iris
def sinal(valor):
    if valor < 0:
        return -1
    elif valor > 0:
        return 1
    else:
        return 0

# Faz uma previsao usando os pesos
def preve(entrada, pesos):
    valor = 0

    for i in range(len(entrada)):
        valor += pesos[i] * entrada[i]
    return sinal(valor)

def plot_pesos(pesos, figure, color):
    plt.figure(figure)
    ww = pesos
    ww1 = [ww[1],-ww[0]]
    ww2 = [-ww[1],ww[0]]
    plt.plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],color)


# Treina o dataset de entrada
def treina(data):
    # pesos iniciais como 0.0
    pesos = np.zeros(len(data[0][0]))

    for entrada in data:
        val = entrada[0]
        res = entrada[1]
        previsao = preve(val, pesos)
        if(previsao != res):

            pesos = atualiza_pesos(pesos, val, res)
    return pesos

def atualiza_pesos(pesos, entrada, resultado):
    #print(pesos, entrada)
    for i in range(len(entrada)):
        #print(pesos)
        pesos[i] = pesos[i] + resultado*entrada[i]
        #print(pesos)

    return pesos

def formata_dados(linha):
    d = linha.strip().split(',')

    valores = [float(i) for i in d[:-1]]
    arg1 = valores[0] + valores[1]
    arg2 = valores[2] + valores[3]

    #print(arg1, arg2)

    if d[4] == 'Iris-setosa':
        d[4] = -1
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
        d[4] = 1
        plt.figure(1)
        plt.plot([1,2,3,4], d[:-1], 'b')
        plt.figure(2)
        plt.plot(arg1, arg2, 'bo')

    dados = np.array([np.array([arg1, arg2]), d[4]])
    #print(dados)
    return dados


iris_dados = []
with open("IrisDataset.txt", "r") as i:
    linha_atual = 0
    
    # use as primeiras 130 linhas para treinar os pesos
    for linha in i:
        dados_linha = formata_dados(linha)
        iris_dados.append(dados_linha)

        '''
        # pare caso chegue na linha 130
        linha_atual += 1
        if(linha_atual >= 130):
            break
        '''
    
    # Tive que randomizar o dataset senão ele não treinava direito
    seed(1000)
    shuffle(iris_dados)
    pesos = treina(iris_dados)
    plt.figure(2)
    print(pesos[0], pesos[1])
    #plt.plot(np.linspace(pesos[0],pesos[1]), "y")
    print(pesos)
    plot_pesos(pesos, 2, 'y')
    

plt.show()


with open("IrisDataset.txt", "r") as i:
    # faz a predição
    for linha in i:
        dados_linha = formata_dados(linha)
        valores   = dados_linha[0]
        resultado = dados_linha[1]

        previsao = preve(valores, pesos)
        print("Correto: " + str(resultado) + " - Previsto: " + str(previsao))