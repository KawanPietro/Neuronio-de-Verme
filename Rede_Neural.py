import math
import random

def tanh(x):
    x = max(-500.0, min(500.0, x))
    return math.tanh(x)

class WormBrain:

    def __init__(self):
        def xavier_init(fan_in, fan_out):
            limit = math.sqrt(6 / (fan_in + fan_out))
            return random.uniform(-limit, limit)

        # Inicializar pesos usando Xavier Initialization
        self.w_input_hidden = [
            [xavier_init(2, 4) for _ in range(4)]  # 2 entradas, 4 neurônios na camada oculta
            for _ in range(2)
        ]
        self.w_hidden_output = [
            [xavier_init(4, 3) for _ in range(3)]  # 4 neurônios na camada oculta, 3 saídas
            for _ in range(4)
        ]

        # Inicializar vieses aleatoriamente (mantendo a inicialização simples para vieses)
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(4)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(3)]

        # Variáveis para armazenar os últimos valores
        self.last_hidden = [0.0] * 4
        self.last_output = [0.0] * 3
        self.last_inputs = [0.0] * 2

    def forward(self, light: float, touch: float) -> tuple:
        """
        Realiza a propagação direta na rede neural.

        Args:
            light (float): Entrada representando a intensidade da luz.
            touch (float): Entrada representando o estímulo tátil.

        Returns:
            tuple: Saídas da rede neural (3 valores normalizados entre -1 e 1).
        """
        
        inputs = [light, touch]
        self.last_inputs = inputs
        hidden = []
        for j in range(4):
            soma = self.bias_hidden[j]
            for i in range(2):
                soma += inputs[i] * self.w_input_hidden[i][j]
            hidden.append(tanh(soma))
        self.last_hidden = hidden
        output = []
        for k in range(3):
            soma = self.bias_output[k]
            for j in range(4):
                soma += hidden[j] * self.w_hidden_output[j][k]
            output.append(tanh(soma))
        self.last_output = output
        return (output[0], output[1], output[2])

    def reinforce(self, reward: float, learning_rate: float = 0.01, regularization: float = 0.001):
        """
        Atualiza os pesos e vieses com base na recompensa recebida, usando L2 Regularization.

        Args:
            reward (float): Recompensa recebida.
            learning_rate (float): Taxa de aprendizado.
            regularization (float): Fator de regularização (L2).
        """
        # Atualizar pesos da camada oculta para a saída
        for j in range(4):
            for k in range(3):
                self.w_hidden_output[j][k] += (
                    learning_rate * reward * self.last_hidden[j] * self.last_output[k]
                    - regularization * self.w_hidden_output[j][k]  # L2 Regularization
                )

        # Atualizar pesos da entrada para a camada oculta
        for i in range(2):
            for j in range(4):
                self.w_input_hidden[i][j] += (
                    learning_rate * reward * self.last_inputs[i] * self.last_hidden[j]
                    - regularization * self.w_input_hidden[i][j]  # L2 Regularization
                )

        # Atualizar vieses da camada oculta
        for j in range(4):
            self.bias_hidden[j] += learning_rate * reward * self.last_hidden[j]

        # Atualizar vieses da camada de saída
        for k in range(3):
            self.bias_output[k] += learning_rate * reward * self.last_output[k]
    def get_weights(self) -> dict:
        return {
            'w_input_hidden' : self.w_input_hidden,
            'w_hidden_output': self.w_hidden_output,
            'bias_hidden'    : self.bias_hidden,
            'bias_output'    : self.bias_output,
        }

    def set_weights(self, weights: dict):
        self.w_input_hidden  = weights['w_input_hidden']
        self.w_hidden_output = weights['w_hidden_output']
        self.bias_hidden     = weights['bias_hidden']
        self.bias_output     = weights['bias_output']

brain = WormBrain()

# Simular algumas atualizações com reforço
for _ in range(10):
    reward = random.uniform(-1, 1)  # Recompensa aleatória
    brain.reinforce(reward=reward, learning_rate=0.01, regularization=0.001)

# Verificar os pesos após as atualizações
print("Pesos (Entrada -> Oculta):", brain.w_input_hidden)
print("Pesos (Oculta -> Saída):", brain.w_hidden_output)