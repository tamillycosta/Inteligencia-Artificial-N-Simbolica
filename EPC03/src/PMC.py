import numpy as np


class PMC:

    '''
    Abstrai uma rede Perceptron Multicamadas (PMC)

    Parâmetros:
        topologia : lista com o número de neurônios por camada
                    Ex: [4, 5, 3]
        taxa_aprendizado : taxa de aprendizado (η)
    '''
    class PMC:
        def __init__(self, topologia : list, taxa_aprendizado: float):
            self.taxa_aprendizado = taxa_aprendizado
            self.camadas = [] #lits[matrizes_peso de cada camada]

            for i in range(len(topologia) - 1):
                n_origem = topologia[i]
                n_destino = topologia[i + 1]

                W = np.random.uniform(0, 1, (n_destino, n_origem + 1)) #+ 1 do bias

                # cada camada guarda seus próprios pesos
                self.camadas.append({
                    "W": W
                })


    def sigmoid(self, u):
        return 1 / (1 + np.exp(-0.5 * u))
    

    def sigmoid_deriv(self, y):
        return 0.5 * y * (1 - y)
    

    def forward(self, x: list) -> list[list]:
        # lista que guarda a saída de cada camada 
        saidas = [x]
        for W in self.camadas:
            x = np.append(x, 1)
        
            # soma ponderada: u = W * x
            u = np.dot(W, x)
            x = self.sigmoid(u)
            saidas.append(x)
        
        return saidas
    

    def backward(self, saidas, y_real):
    """
    saidas: lista do forward
            [entrada, camada1, ..., saída_final]

    y_real: saída esperada
    """

    deltas = []

    # =========================
    # 🔹 1. DELTA DA SAÍDA
    # =========================

    y_pred = saidas[-1]

    erro = y_real - y_pred

    delta_saida = erro * self.sigmoid_deriv(y_pred)

    deltas.append(delta_saida)

    # =========================
    # 🔹 2. DELTAS OCULTOS
    # =========================

    # percorre as camadas de trás pra frente
    for i in range(len(self.pesos) - 1, 0, -1):
        W = self.pesos[i]

        # remove a coluna do bias
        W_sem_bias = W[:, :-1]

        delta_prox = deltas[0]     # delta da camada seguinte
        saida_atual = saidas[i]    # saída da camada atual

        # propaga erro pra trás
        delta = np.dot(W_sem_bias.T, delta_prox) * self.sigmoid_deriv(saida_atual)

        deltas.insert(0, delta)

    # =========================
    # 🔹 3. ATUALIZAÇÃO DOS PESOS
    # =========================

    for i in range(len(self.pesos)):
        entrada = saidas[i]

        # adiciona bias
        entrada = np.append(entrada, 1)

        delta = deltas[i]

        # transforma em formato correto
        delta = delta.reshape(-1, 1)
        entrada = entrada.reshape(1, -1)

        # regra de atualização
        self.pesos[i] += self.taxa_aprendizado * np.dot(delta, entrada)



    def calc_eqm(self, amostras):
        p = len(amostras)
        erro_quadrastico = 0

        for  x1, x2, d in amostras:
            x = self.__montar_x(x1,x2)
            u = self.calc_u(x)
            erro_quadrastico += (d - u) ** 2  
        return erro_quadrastico / p


    def train(self, entradas_x : list, saidas_d : list, taxa_erro :float):
        """
        X: entradas (lista de padrões)
        Y: saídas esperadas

        algoritimo :
        Para cada época:
        calcular EQM anterior
        para cada amostra:
            forward completo
            calcular deltas (saída → ocultas)
            atualizar pesos camada por camada
        calcular EQM atual
        parar quando |EQM_atual - EQM_anterior| < ε

        """
        epoca = 0
        eqm_anterior = 0

        while True:
            for x, d in zip(entradas_x, saidas_d):

                # forward
                saidas_neuronios = self.forward(x)
                y_final = saidas_neuronios[-1]

                #eqm
                erro = d - y_final
                erro_total += np.sum(erro ** 2)

                # backward 
                self.backward(saidas_neuronios, d)

            eqm_atual = erro_total / len(x)

            print(f"Época {epoca} | EQM: {eqm_atual:.6f}")

            if(abs(eqm_atual - eqm_anterior < taxa_erro)){
                break
            }

            eqm_anterior = eqm_atual
            epoca += 1

        return epoca, eqm_atual




    


  