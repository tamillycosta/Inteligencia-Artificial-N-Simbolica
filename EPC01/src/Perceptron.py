import random

class Perceptron:

    def __init__(self, learning_rate=0.01):
      self.learning_rate = learning_rate 
      self.w = [random.uniform(0,1) for _ in range(3)]

              
    def get_weights(self)->list:
        return self.w

    def ativacao(self, x1: float, x2: float ) -> int:
        x = [-1, x1, x2]
        u = 0

        for i in range(len(self.w)):
            u += self.w[i] * x[i]

        if u >= 0:
            return 1
        else:
            return 0 
       
    def __atualiza_pesos(self, x1: float, x2: float, error: int ):
        x = [-1, x1, x2]

        for i in range(len(self.w)):
            self.w[i] = self.w[i] + self.learning_rate * error * x[i]

 
    def train(self, amostras: list) -> tuple:
       
        epocas = 0

        while True:
            erro_total = 0

            for x1, x2, d in amostras:

                y = self.ativacao(x1, x2)
                
                if d - y != 0:
                    self.__atualiza_pesos(x1, x2, d - y)
                    erro_total += 1

            epocas += 1

            if erro_total == 0:
                break

        return epocas, self.w

    #classificação 
    def operacao(self, amostras: list)-> list:
        resultados = []

        for x1, x2 ,_ in amostras:
            y = self.ativacao(x1, x2)
            resultados.append(y)

        return resultados



   

    
    