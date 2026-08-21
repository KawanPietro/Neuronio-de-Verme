import math
import random

from config import CONFIG

# NOTA: este módulo é o cérebro LEGADO (regra Hebbian-like, Fase 0/1), mantido
# como referência didática. O cérebro usado pelo jogo agora é o `PolicyNetwork`
# de `mlp.py` (MLP com backprop + política estocástica, Fase 2+).


def tanh(x):
    x = max(-500.0, min(500.0, x))
    return math.tanh(x)


class WormBrain:
    """MLP feedforward simples, implementado do zero, com arquitetura configurável."""

    def __init__(self, n_inputs=None, n_hidden=None, n_outputs=None):
        self.n_inputs  = CONFIG['n_inputs']  if n_inputs  is None else n_inputs
        self.n_hidden  = CONFIG['n_hidden']  if n_hidden  is None else n_hidden
        self.n_outputs = CONFIG['n_outputs'] if n_outputs is None else n_outputs
        self.reset()

    def reset(self):
        """Reinicializa todos os pesos e vieses do zero (Xavier)."""
        def xavier_init(fan_in, fan_out):
            limit = math.sqrt(6 / (fan_in + fan_out))
            return random.uniform(-limit, limit)

        self.w_input_hidden = [
            [xavier_init(self.n_inputs, self.n_hidden) for _ in range(self.n_hidden)]
            for _ in range(self.n_inputs)
        ]
        self.w_hidden_output = [
            [xavier_init(self.n_hidden, self.n_outputs) for _ in range(self.n_outputs)]
            for _ in range(self.n_hidden)
        ]

        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(self.n_hidden)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(self.n_outputs)]

        self.last_hidden = [0.0] * self.n_hidden
        self.last_output = [0.0] * self.n_outputs
        self.last_inputs = [0.0] * self.n_inputs

    def forward(self, inputs: list) -> tuple:
        """
        Realiza a propagação direta na rede neural.

        Args:
            inputs (list): lista com `n_inputs` valores de estado.

        Returns:
            tuple: saídas da rede neural (n_outputs valores entre -1 e 1).
        """
        self.last_inputs = list(inputs)
        hidden = []
        for j in range(self.n_hidden):
            soma = self.bias_hidden[j]
            for i in range(self.n_inputs):
                soma += inputs[i] * self.w_input_hidden[i][j]
            hidden.append(tanh(soma))
        self.last_hidden = hidden
        output = []
        for k in range(self.n_outputs):
            soma = self.bias_output[k]
            for j in range(self.n_hidden):
                soma += hidden[j] * self.w_hidden_output[j][k]
            output.append(tanh(soma))
        self.last_output = output
        return tuple(output)

    def reinforce(self, reward: float, learning_rate: float = None, regularization: float = None):
        """
        Atualiza os pesos e vieses com base na recompensa recebida, usando L2 Regularization.

        Args:
            reward (float): Recompensa recebida.
            learning_rate (float): Taxa de aprendizado (padrão: CONFIG).
            regularization (float): Fator de regularização L2 (padrão: CONFIG).
        """
        if learning_rate is None:
            learning_rate = CONFIG['learning_rate']
        if regularization is None:
            regularization = CONFIG['regularization']

        # Atualizar pesos da camada oculta para a saída
        for j in range(self.n_hidden):
            for k in range(self.n_outputs):
                self.w_hidden_output[j][k] += (
                    learning_rate * reward * self.last_hidden[j] * self.last_output[k]
                    - regularization * self.w_hidden_output[j][k]  # L2 Regularization
                )

        # Atualizar pesos da entrada para a camada oculta
        for i in range(self.n_inputs):
            for j in range(self.n_hidden):
                self.w_input_hidden[i][j] += (
                    learning_rate * reward * self.last_inputs[i] * self.last_hidden[j]
                    - regularization * self.w_input_hidden[i][j]  # L2 Regularization
                )

        # Atualizar vieses da camada oculta
        for j in range(self.n_hidden):
            self.bias_hidden[j] += learning_rate * reward * self.last_hidden[j]

        # Atualizar vieses da camada de saída
        for k in range(self.n_outputs):
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


if __name__ == '__main__':
    # Demo de linha de comando: cria um cérebro e aplica 10 reforços aleatórios.
    brain = WormBrain()
    for _ in range(10):
        reward = random.uniform(-1, 1)  # Recompensa aleatória
        brain.reinforce(reward=reward)

    print(f"Arquitetura: {brain.n_inputs} -> {brain.n_hidden} -> {brain.n_outputs}")
    print("Pesos (Entrada -> Oculta):", brain.w_input_hidden)
    print("Pesos (Oculta -> Saída):", brain.w_hidden_output)