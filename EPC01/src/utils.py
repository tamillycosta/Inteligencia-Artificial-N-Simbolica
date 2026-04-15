<<<<<<< HEAD

def carregar_dados(caminho : str) -> list:
=======
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def carregar_dados(caminho_relativo : str) -> list:
    caminho = os.path.join(BASE_DIR, caminho_relativo)
>>>>>>> cfcc0a595d669bf76e9c8981e374bf001fcde7f1
    dados = []
    arquivo = open(caminho, "r")

    for linha in arquivo:

        valores = linha.split()

        x1 = float(valores[0])
        x2 = float(valores[1])
        d = float(valores[2])

        dados.append([x1, x2, d])
        
    arquivo.close()
    return dados

def limpar_arquivo(caminho_relativo:str):
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    open(caminho, "w").close()
    

def escrever_result_treinamento(caminho_relativo: str, epocas: int, pesos_iniciais: list, pesos_finais: list) -> None:
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    with open(caminho, "a") as arquivo:

        arquivo.write("Pesos iniciais:\n")
        arquivo.write(f"{pesos_iniciais[0]:.4f}, {pesos_iniciais[1]:.4f}, {pesos_iniciais[2]:.4f} \n")

        arquivo.write("Pesos finais:\n")
        arquivo.write(f"{pesos_finais[0]:.4f}, {pesos_finais[1]:.4f}, {pesos_finais[2]:.4f}\n")

        arquivo.write(f"Épocas: {epocas}\n")


        arquivo.write("\n")  

def escrever_result_teste(caminho_relativo: str, resultados: list ):
    caminho = os.path.join(BASE_DIR, caminho_relativo)
    with open(caminho, "a") as arquivo:
        for i in range(len(resultados)):
            arquivo.write(f"treinamento {i+1}: \n ")
            arquivo.write(f"{resultados[0]}\n")


