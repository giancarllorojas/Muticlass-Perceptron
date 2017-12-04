import matplotlib.pyplot as plt
import numpy as np
import sys
import re
import matplotlib.cm as cm
from matplotlib.colors import Normalize

if len(sys.argv) < 4:
    print("Execute python3 iris.py loops_de_treino numero_de_classes plotar_sim_ou_nao")
    sys.exit()

num_treino  = int(sys.argv[1])
num_classes = int(sys.argv[2])
plotar      = int(sys.argv[3])

# Faz uma previsão e retorna o salario estimado
def preve(entrada, pesos, classes):
    c = preve_classe(entrada, pesos)

    valor = classes[c]

    if c+1 < len(classes):
        valor = (classes[c] + classes[c+1])/2
        #print(classes[c], classes[c+1])
    else:
        valor = classes[c]
    return valor


# Faz uma previsao usando os pesos
def preve_classe(entrada, pesos):
    #multiplica a entrada pelos pesos do modelo
    prev_val = np.dot(pesos, entrada)

    # retorna o maior valor encontrado como resultado da classificação
    return np.argmax(prev_val)

def plot_pesos(pesos, figure):
    plt.figure(figure)
    ww = pesos
    ww1 = [ww[1],-ww[0]]
    ww2 = [-ww[1],ww[0]]
    plt.plot([ww1[0], ww2[0]],[ww1[1], ww2[1]])


# Treina o dataset de entrada
def treina(data, classes, quantidade):
    pesos = np.zeros((len(classes),len(data[0][0])))
    #print(pesos)
    for i in range(quantidade):
        erro_total = 0
        for entrada in data:
            val = entrada[0]
            res = entrada[1]
            
            previsao = preve_classe(val, pesos)
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

'''
Função simples de padronização dos dados para valores numéricos
'''
def padroniza_dado(entrada):
    if entrada == 'male':
        return 1
    if entrada == 'female':
        return 2
    if entrada == 'full':
        return 1
    if entrada == 'associate':
        return 2
    if entrada == 'assistant':
        return 3
    if entrada == 'doctorate':
        return 1
    if entrada == 'masters':
        return 2

    return entrada

'''
Formata uma linha do arquivo de entrada, removendo os espaços e separando os dados
'''
def formata_dados(linha):
    linha = re.sub('[ ]+', '\t', linha)
    d = linha.strip().split('\t')
    d = [int(padroniza_dado(x)) for x in d]

    data = d[:-1] 
    #data = [d[0]**3 + d[1], d[2] + d[3], d[4]]


    return np.array([data, d[-1]])

'''
Dado um valor retorna em qual classe de valores esse valor está
'''
def descobre_classe(classes_intervalo, valor):
    for i in range(len(classes_intervalo)):
        if valor < classes_intervalo[i]:
            return i

'''
Gera os intervalos para uma quantidade num_classes de classes
'''
def gera_classes(dados_salarios, num_classes):
    salarios = dados_salarios[:,1]
    minimo = salarios[np.argmin(salarios)]
    maximo = salarios[np.argmax(salarios)]

    intervalos = np.linspace(minimo, maximo+1, num_classes)
    return intervalos

'''
Pega o array de salarios e atribui a cada salario uma classe
'''
def categoriza_salarios(dados_salarios, classes):
    for i in range(len(dados_salarios)):
        dados_salarios[i][1] = descobre_classe(classes, dados_salarios[i][1])
    return dados_salarios


with open("SalariesTreino.txt", "r") as i:
    dados_salarios = []
    i.readline()

    for linha in i:
        dados_linha = formata_dados(linha)
        dados_salarios.append(dados_linha)

    dados_salarios = np.array(dados_salarios)
    classes = gera_classes(dados_salarios, num_classes)
    dados_salarios = categoriza_salarios(dados_salarios, classes)

    pesos = treina(dados_salarios, classes, num_treino)

    if plotar == 1:
        for d in dados_salarios:
            cmap = cm.autumn
            norm = Normalize(vmin=-10, vmax=10)
            color = cmap(norm(d[1]))

            val1 = d[0][0] + d[0][2]
            val2 = d[0][1] + d[0][3] + d[0][4]

            plt.figure(1)
            plt.plot(['sx', 'rk', 'yr', 'dg', 'yd', 'sal'], np.append(d[0],d[1]), color = color)

            plt.figure(2)
            plt.plot(val1, val2, 'o', color = color)
        plt.show()


with open("SalariesValidacao.txt", "r") as s:
    dados_treino = []
    for linha in s:
        dados_linha = formata_dados(linha)
        dados_treino.append(dados_linha)

    dados_treino = np.array(dados_treino)

    total_erro = 0
    for i, d in enumerate(dados_treino):
        sala = d[1]
        prev = preve(d[0], pesos, classes)
        erro = sala-prev
        total_erro += erro
        print("Previsto: " + str(prev) + " - Salario: " + str(sala) + " || Erro: " + str(erro))
    print("Media de erro: " + str(total_erro/len(dados_treino)))

with open("SalariesGeneros.txt", "r") as s:
    dados_treino = []
    for linha in s:
        dados_linha = formata_dados(linha)
        dados_treino.append(dados_linha)

    dados_treino = np.array(dados_treino)

    total_erro = 0
    for i, d in enumerate(dados_treino):
        sala = d[1]
        prev = preve(d[0], pesos, classes)
        erro = sala-prev
        total_erro += erro
        print("Genero: " + str(d[0][0]) + " - Previsto: " + str(prev))
    print("Media de erro: " + str(total_erro/len(dados_treino)))