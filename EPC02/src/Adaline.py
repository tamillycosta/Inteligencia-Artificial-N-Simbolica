import random
import numpy as np
class Adaline:

    def __init__(self, taxa_aprendizado ):
        self.taxa_aprendizado = taxa_aprendizado
        self.w = np.array([[random.uniform(0, 1)] for _ in range(3)])

    def get_weights(self)->list:
        return self.w.flatten() 
    
    def __atualiza_pesos(self, x, error):
        self.w = self.w + self.taxa_aprendizado * error * x


    def calc_u(self, x):
        return np.dot(self.w.T, x)[0][0]
    

    def ativacao(self, u):
        return 1 if u >= 0 else -1  


    def calc_eqm(self, amostras):
        p = len(amostras)
        erro_quadrastico = 0

        for  x1, x2, d in amostras:
            x = self.__montar_x(x1,x2)
            u = self.calc_u(x)
            erro_quadrastico += (d - u) ** 2  
        return erro_quadrastico / p

    # O algortimo converge quando o erro quadrastico medio entre duas epocas
    # for suficientemente pequeno 
    def train(self, amostras: list, taxa_erro) -> tuple:
        amostras_bipolar = self.__converter(amostras)

        epocas = 0
        eqm = self.calc_eqm(amostras_bipolar)
        historico_eqm = []

        while True:
          
            for x1, x2 , d in amostras_bipolar:
                x = self.__montar_x(x1, x2)
                u = self.calc_u(x)
              
                self.__atualiza_pesos(x, d - u)

            epocas += 1
            eqm_atual = self.calc_eqm(amostras_bipolar)
            historico_eqm.append(eqm_atual)

            if abs(eqm_atual - eqm) <= taxa_erro :
                break
            else :
                eqm = eqm_atual 

      
        return  epocas, self.get_weights(), historico_eqm

    
    def operacao(self, amostras:list):
        resultados = []
        for x1, x2 , _ in amostras:
            x = self.__montar_x(x1, x2)
            u = self.calc_u(x)
            y = self.ativacao(u)

           
            resultados.append(1 if y == 1 else 0)

        return resultados

    def __montar_x(self, x1, x2):
        return np.array([[-1], [x1], [x2]])
   

    def __converter(self, amostras):
        resultado = []
        for x1, x2, d in amostras:
            if d == 1:
                resultado.append((x1, x2, 1.0))
            else:
                resultado.append((x1, x2, -1.0))
        return resultado

