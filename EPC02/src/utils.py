import os
import matplotlib
import matplotlib.pyplot as plt   

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def carregar_dados(caminho_relativo : str) -> list:
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    dados = []
    with open(caminho, "r") as arquivo:

        for linha in arquivo:

            valores = linha.split()

            x1 = float(valores[0])
            x2 = float(valores[1])
            d = float(valores[2])

            dados.append([x1, x2, d])
        
    
    return dados


def limpar_arquivo(caminho_relativo:str):
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    open(caminho, "w").close()
    

def escrever_result_treinamento(caminho_relativo: str, epocas: int, pesos_iniciais: list, pesos_finais: list, taxa_aprendizagem : float, taxa_erro : float ) -> None:
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    with open(caminho, "a") as arquivo:

        arquivo.write("Pesos iniciais:\n")
        arquivo.write(f"{pesos_iniciais[0]:.4f}, {pesos_iniciais[1]:.4f}, {pesos_iniciais[2]:.4f} \n")

        arquivo.write("Pesos finais:\n")
        arquivo.write(f"{pesos_finais[0]:.4f}, {pesos_finais[1]:.4f}, {pesos_finais[2]:.4f}\n")

        arquivo.write(f"Épocas: {epocas}\n")

        arquivo.write(f"Taxa de aprendizagem: {taxa_aprendizagem}\n")

        arquivo.write(f"Taxa de erro: {taxa_erro}\n")

        arquivo.write("\n")  
        

def escrever_result_teste(caminho_relativo: str, resultados: list ):
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    with open(caminho, "a") as arquivo:
        for i in range(len(resultados)):
            arquivo.write(f"treinamento {i+1}: \n ")
            arquivo.write(f"{resultados[0]}\n")
        arquivo.write("\n")  



def plotar_eqm(historicos : list, caminho_relativo):
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    _, axs = plt.subplots(5, 1, figsize=(8, 12))  
    for i, historico in enumerate(historicos):
        epocas = list(range(1, len(historico) + 1))

        axs[i].plot(epocas, historico)
        axs[i].set_title(f'Treinamento T{i+1}')
        axs[i].set_xlabel("Épocas")
        axs[i].set_ylabel("EQM")
        axs[i].grid()

    plt.tight_layout()

    plt.savefig(caminho)



def plotar_separacao(dados, modelos, caminho_relativo):
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    _, axs = plt.subplots(5, 1, figsize=(8, 20))

    for i, modelo in enumerate(modelos):
        w = modelo.get_weights()

        # separar pontos por classe
        x1_c0 = [x1 for x1, x2, d in dados if d == 0]
        x2_c0 = [x2 for x1, x2, d in dados if d == 0]
        x1_c1 = [x1 for x1, x2, d in dados if d == 1]
        x2_c1 = [x2 for x1, x2, d in dados if d == 1]

        axs[i].scatter(x1_c0, x2_c0, color='blue', label='Classe 0', marker='o')
        axs[i].scatter(x1_c1, x2_c1, color='red', label='Classe 1', marker='x')

        # reta de separação
        x1_vals = [min(x1_c0 + x1_c1) - 0.5, max(x1_c0 + x1_c1) + 0.5]
        x2_vals = [(w[0] - w[1] * x1) / w[2] for x1 in x1_vals]     

        axs[i].plot(x1_vals, x2_vals, color='green', label='Reta de separação')
        axs[i].set_title(f'Treinamento T{i+1}')
        axs[i].set_xlabel("x1")
        axs[i].set_ylabel("x2")
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()
    plt.savefig(caminho)
