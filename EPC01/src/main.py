from src import utils
from src.Perceptron import Perceptron


def start():
    utils.limpar_arquivo("results/resultado_treinamento.txt")
    utils.limpar_arquivo("results/resultado_teste.txt")

    dados_treinamento =  utils.carregar_dados("data/dados-tra.txt")
    modelos = []
   
    # treinamento 
    for _ in range(5):
        perceptron = Perceptron()
        pesos_iniciais = list(perceptron.get_weights())

        epocas, pesos_finais = perceptron.train(dados_treinamento)
        utils.escrever_result_treinamento("results/resultado_treinamento.txt",epocas, pesos_iniciais, pesos_finais)
        modelos.append(perceptron)

    #classificação
    dados_teste = utils.carregar_dados("data/dados-tst.txt")
    resultados = []
    for modelo in modelos:
        resultados.append(modelo.operacao(dados_teste))
    utils.escrever_result_teste("results/resultado_teste.txt",resultados)
    
    
