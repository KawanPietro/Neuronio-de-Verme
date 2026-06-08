import math
import random

def tanh(x):
    x = max(-500.0, min(500.0, x))
    return math.tanh(x)

class WormBrain:

    def __init__(self):
        self.w_input_hidden = [
            [random.uniform(-1, 1) for _ in range(4)]
            for _ in range(2)
        ]
        self.w_hidden_output = [
            [random.uniform(-1, 1) for _ in range(3)]
            for _ in range(4)
        ]
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(4)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(3)]
        self.last_hidden  = [0.0] * 4
        self.last_output  = [0.0] * 3
        self.last_inputs  = [0.0] * 2

    def forward(self, light: float, touch: float) -> tuple:
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

    def reinforce(self, reward: float, learning_rate: float = 0.01):
        for j in range(4):
            for k in range(3):
                self.w_hidden_output[j][k] += (
                    learning_rate * reward * self.last_hidden[j] * self.last_output[k]
                )
        for i in range(2):
            for j in range(4):
                self.w_input_hidden[i][j] += (
                    learning_rate * reward * self.last_inputs[i] * self.last_hidden[j]
                )
        for j in range(4):
            self.bias_hidden[j] += learning_rate * reward * self.last_hidden[j]
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