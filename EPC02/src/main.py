from src import utils
from src.Adaline import Adaline



TAXA_APRENDIZADO = 0.01
TAXA_ERRO = 0.000001

def main():

    datasets = [
    ("data/dados-tra.txt", "data/dados-tst.txt"),
    ("data/dados2-tra.txt", "data/dados2-tst.txt"),
    ]
    cont = 0
   
    utils.limpar_arquivo("results/resultado_treinamento.txt")
    utils.limpar_arquivo("results/resultado_teste.txt") 

    for data_train , data_test in datasets:
       

        dados_treinamento =  utils.carregar_dados(data_train)
        dados_teste = utils.carregar_dados(data_test)

        modelos = []
        historico_eqm_treinamentos = []

        for _ in range(5):
            
            adaline = Adaline(TAXA_APRENDIZADO)
            pesos_iniciais = list(adaline.get_weights())

            # treinamento 
            epocas, pesos_finais, historico_eqm = adaline.train(dados_treinamento, TAXA_ERRO)
            utils.escrever_result_treinamento("results/resultado_treinamento.txt",epocas, pesos_iniciais, pesos_finais, TAXA_APRENDIZADO, TAXA_ERRO)

            modelos.append(adaline)
            historico_eqm_treinamentos.append(historico_eqm)
         
        cont += 1
       
        utils.plotar_eqm(historico_eqm_treinamentos,f"results/graficos_eqm_{cont}.png")
      
        resultados = []
        for modelo in modelos:
            #classificação
            resultados.append(modelo.operacao(dados_teste))

        utils.escrever_result_teste("results/resultado_teste.txt",resultados)
        utils.plotar_separacao(dados_teste, modelos, f"results/graficos_sep_{cont}.png")

        modelo = []

   
    

