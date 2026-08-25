import json
import math
import random

from config import CONFIG


# ─── Funções de ativação ──────────────────────────────────────────────────────

def tanh(x):
    x = max(-500.0, min(500.0, x))
    return math.tanh(x)


def tanh_deriv(y):
    """Derivada da tanh em função da SAÍDA: 1 − tanh²(x)."""
    return 1.0 - y * y


def relu(x):
    return x if x > 0 else 0.0


def relu_deriv(x):
    return 1.0 if x > 0 else 0.0


def softmax(logits):
    """Softmax estável (subtrai o máximo para evitar overflow)."""
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    total = sum(e)
    return [v / total for v in e]


def entropy(probs):
    """Entropia de uma distribuição: H = −Σ p·log p."""
    return -sum(p * math.log(p + 1e-12) for p in probs)


def compute_returns(rewards, gamma):
    """
    Retornos descontados G_t = Σ_{k≥0} γ^k · r_{t+k} (REINFORCE/Fase 3).

    Percorre do fim para o início: G_t = r_t + γ·G_{t+1}.
    """
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


# ─── MLP com backpropagation ──────────────────────────────────────────────────
#
# Implementado camada a camada (SEM loops genéricos de matriz), para que cada
# regra da cadeia fique visível e didática:
#
#   forward:   z1 = W1·x + b1 → h = act(z1) → z2 = W2·h + b2 → out = out_act(z2)
#   backward:  dL/dW2 = dL/dz2·hᵀ ;  dL/dz1 = W2ᵀ·dL/dz2 ⊙ act'(z1) ;  dL/dW1 = dL/dz1·xᵀ

class MLP:

    def __init__(self, n_inputs, n_hidden, n_outputs,
                 hidden_activation='tanh', output_activation='identity'):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.hidden_activation = hidden_activation   # 'tanh' ou 'relu'
        self.output_activation = output_activation   # 'identity' ou 'tanh'
        self.reset()

    # ── Inicialização ────────────────────────────────────────────────────────

    def reset(self):
        def xavier_init(fan_in, fan_out):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return random.uniform(-limit, limit)

        self.w_input_hidden = [
            [xavier_init(self.n_inputs, self.n_hidden) for _ in range(self.n_hidden)]
            for _ in range(self.n_inputs)
        ]
        self.bias_hidden = [0.0] * self.n_hidden
        self.w_hidden_output = [
            [xavier_init(self.n_hidden, self.n_outputs) for _ in range(self.n_outputs)]
            for _ in range(self.n_hidden)
        ]
        self.bias_output = [0.0] * self.n_outputs

        # Ativações intermediárias guardadas no forward (para o backward)
        self.last_input = None
        self.last_pre_hidden = None
        self.last_hidden = None
        self.last_pre_output = None
        self.last_output = None

        self.zero_grad()

    def zero_grad(self):
        self.grad_w_input_hidden = None
        self.grad_bias_hidden = None
        self.grad_w_hidden_output = None
        self.grad_bias_output = None
        self.grad_acc = None

    # ── Ativações ────────────────────────────────────────────────────────────

    def _act(self, x):
        return relu(x) if self.hidden_activation == 'relu' else tanh(x)

    def _act_deriv(self, pre, act):
        return relu_deriv(pre) if self.hidden_activation == 'relu' else tanh_deriv(act)

    def _out(self, s):
        if self.output_activation == 'tanh':
            return tanh(s)
        return s

    def output_activation_derivs(self):
        """d(out_k)/d(z_k) — usado para compor o gradiente da saída."""
        if self.output_activation == 'tanh':
            return [tanh_deriv(v) for v in self.last_output]
        return [1.0] * self.n_outputs

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, inputs):
        self.last_input = list(inputs)

        self.last_pre_hidden = []
        self.last_hidden = []
        for j in range(self.n_hidden):
            s = self.bias_hidden[j]
            for i in range(self.n_inputs):
                s += inputs[i] * self.w_input_hidden[i][j]
            self.last_pre_hidden.append(s)
            self.last_hidden.append(self._act(s))

        self.last_pre_output = []
        self.last_output = []
        for k in range(self.n_outputs):
            s = self.bias_output[k]
            for j in range(self.n_hidden):
                s += self.last_hidden[j] * self.w_hidden_output[j][k]
            self.last_pre_output.append(s)
            self.last_output.append(self._out(s))

        return list(self.last_output)

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward_from_output_grad(self, grad_output):
        """
        Propaga dL/dz (gradiente em relação aos logits/saídas pré-ativação).

        Camada a camada:
          camada 2: dL/dW2[j][k] = dL/dz2[k] · h[j] ;  dL/db2[k] = dL/dz2[k]
          oculta:   dL/dh[j] = Σ_k dL/dz2[k] · W2[j][k]
          camada 1: dL/dz1[j] = dL/dh[j] · act'(z1[j])
                    dL/dW1[i][j] = dL/dz1[j] · x[i] ;  dL/db1[j] = dL/dz1[j]
        """
        # Camada saída (oculta → saída)
        self.grad_w_hidden_output = [[0.0] * self.n_outputs for _ in range(self.n_hidden)]
        self.grad_bias_output = [0.0] * self.n_outputs
        grad_hidden = [0.0] * self.n_hidden
        for k in range(self.n_outputs):
            gk = grad_output[k]
            self.grad_bias_output[k] = gk
            for j in range(self.n_hidden):
                self.grad_w_hidden_output[j][k] = gk * self.last_hidden[j]
                grad_hidden[j] += gk * self.w_hidden_output[j][k]

        # Camada oculta (entrada → oculta)
        self.grad_w_input_hidden = [[0.0] * self.n_hidden for _ in range(self.n_inputs)]
        self.grad_bias_hidden = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            local = grad_hidden[j] * self._act_deriv(self.last_pre_hidden[j], self.last_hidden[j])
            self.grad_bias_hidden[j] = local
            for i in range(self.n_inputs):
                self.grad_w_input_hidden[i][j] = local * self.last_input[i]

    # ── Parâmetros ───────────────────────────────────────────────────────────

    def params(self):
        return {
            'w_input_hidden' : self.w_input_hidden,
            'bias_hidden'    : self.bias_hidden,
            'w_hidden_output': self.w_hidden_output,
            'bias_output'    : self.bias_output,
        }

    def grads(self):
        return {
            'w_input_hidden' : self.grad_w_input_hidden,
            'bias_hidden'    : self.grad_bias_hidden,
            'w_hidden_output': self.grad_w_hidden_output,
            'bias_output'    : self.grad_bias_output,
        }

    def set_params(self, params):
        self.w_input_hidden  = params['w_input_hidden']
        self.bias_hidden     = params['bias_hidden']
        self.w_hidden_output = params['w_hidden_output']
        self.bias_output     = params['bias_output']

    # ── Memória (Fase 5): salvar/carregar o cérebro em JSON ──────────────────

    def save_weights(self, path):
        """Exporta todos os pesos/vieses para um arquivo JSON."""
        with open(path, 'w') as f:
            json.dump(self.params(), f)

    def load_weights(self, path):
        """Importa pesos/vieses de um arquivo JSON (sobrescreve os atuais)."""
        with open(path) as f:
            self.set_params(json.load(f))

    def apply_gradients(self, learning_rate, regularization=0.0):
        """θ ← θ + lr·∇ − lr·λ·θ (decay L2)."""
        for i in range(self.n_inputs):
            for j in range(self.n_hidden):
                self.w_input_hidden[i][j] += (
                    learning_rate * self.grad_w_input_hidden[i][j]
                    - learning_rate * regularization * self.w_input_hidden[i][j]
                )
        for j in range(self.n_hidden):
            self.bias_hidden[j] += learning_rate * self.grad_bias_hidden[j]
        for j in range(self.n_hidden):
            for k in range(self.n_outputs):
                self.w_hidden_output[j][k] += (
                    learning_rate * self.grad_w_hidden_output[j][k]
                    - learning_rate * regularization * self.w_hidden_output[j][k]
                )
        for k in range(self.n_outputs):
            self.bias_output[k] += learning_rate * self.grad_bias_output[k]

    # ── Gradientes em lote (para updates episódicos) ─────────────────────────

    def accumulate_grads(self):
        """Soma os gradientes do último `backward` num acumulador (lote)."""
        g = self.grads()
        if self.grad_acc is None:
            self.grad_acc = {
                k: [row[:] for row in v] if isinstance(v[0], list) else v[:]
                for k, v in g.items()
            }
            return
        for k, v in g.items():
            if isinstance(v[0], list):
                for i, row in enumerate(v):
                    for j, val in enumerate(row):
                        self.grad_acc[k][i][j] += val
            else:
                for i, val in enumerate(v):
                    self.grad_acc[k][i] += val

    def apply_accumulated(self, learning_rate, regularization=0.0):
        """Aplica o gradiente acumulado: θ ← θ + lr·∇ − lr·λ·θ; zera o lote."""
        acc = self.grad_acc
        for i in range(self.n_inputs):
            for j in range(self.n_hidden):
                self.w_input_hidden[i][j] += (
                    learning_rate * acc['w_input_hidden'][i][j]
                    - learning_rate * regularization * self.w_input_hidden[i][j]
                )
        for j in range(self.n_hidden):
            self.bias_hidden[j] += learning_rate * acc['bias_hidden'][j]
        for j in range(self.n_hidden):
            for k in range(self.n_outputs):
                self.w_hidden_output[j][k] += (
                    learning_rate * acc['w_hidden_output'][j][k]
                    - learning_rate * regularization * self.w_hidden_output[j][k]
                )
        for k in range(self.n_outputs):
            self.bias_output[k] += learning_rate * acc['bias_output'][k]
        self.grad_acc = None


# ─── Funções de save/load combinado (actor + critic, Fase 8) ─────────────────

def save_brain(path, actor, critic=None):
    """Salva actor (e opcionalmente critic) num JSON."""
    data = {'actor': actor.params()}
    if critic is not None:
        data['critic'] = critic.params()
    with open(path, 'w') as f:
        json.dump(data, f)


def load_brain(path, actor, critic=None):
    """Carrega actor (e opcionalmente critic) de um JSON."""
    with open(path) as f:
        data = json.load(f)
    if 'actor' in data:
        actor.set_params(data['actor'])
    else:
        actor.set_params(data)  # compatibilidade com pesos antigos (só actor)
    if critic is not None and 'critic' in data:
        critic.set_params(data['critic'])
#
# π(a|s) = softmax(W2·h + b2)  →  distribuição de probabilidade sobre ações.
# A ação é AMOSTRADA (não é o argmax) — é isso que permite aprender por tentativa.

ACTIONS = ['esquerda', 'frente_esquerda', 'frente', 'frente_direita', 'direita']


# ─── Crítico (Fase 8 — A2C) ─────────────────────────────────────────────────
#
# V(s) estima o valor esperado do estado: "quanto recompensa total espero daqui?"
# Usado para calcular advantage via GAE:  A_t = δ_t + γλ·δ_{t+1} + ...
# onde δ_t = r_t + γV(s_{t+1}) − V(s_t)  (erro TD).
#
# Arquitetura: idêntica ao actor (8→16→1), mas com saída identidade (escalar).

class CriticNetwork(MLP):

    def __init__(self, n_inputs, n_hidden=16):
        super().__init__(n_inputs, n_hidden, 1, 'tanh', 'identity')
        self.learning_rate = CONFIG['learning_rate']

    def value(self, inputs):
        """V(s) — retorna o valor escalar do estado."""
        return self.forward(inputs)[0]


# ─── Cálculo de GAE (Generalized Advantage Estimation) ──────────────────────
#
# Advantage = "o que aconteceu vs. o que eu esperava".
# GAE suaviza entre TD(0) (baixa variância, alto viés) e REINFORCE (alta variância,
# baixo viés):  A_t = Σ_{l=0} (γλ)^l · δ_{t+l}
# onde δ_t = r_t + γ·V(s_{t+1}) − V(s_t).
#
# λ (gae_lambda) controla o trade-off: λ=0 → puro TD, λ=1 → REINFORCE.

def compute_gae(rewards, values, gamma, gae_lambda):
    """
    Calcula GAE advantage para um episódio.

    rewards: lista de recompensas [r_0, r_1, ..., r_{T-1}]
    values:  lista de valores [V(s_0), V(s_1), ..., V(s_{T-1})]
             (o último V(s_T) é 0 — estado terminal)
    Retorna: lista de advantages [A_0, A_1, ..., A_{T-1}]
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in range(T - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    return advantages


class PolicyNetwork(MLP):

    def __init__(self, n_inputs, n_hidden, n_actions,
                 hidden_activation='tanh', temperature=1.0):
        super().__init__(n_inputs, n_hidden, n_actions, hidden_activation, 'identity')
        self.n_actions = n_actions
        self.temperature = temperature
        self.learning_rate = CONFIG['learning_rate']  # α, com decay por episódio
        self.last_probs = None
        self.last_logits = None
        self.last_temp_scale = 1.0
        self.old_probs = None  # Fase 9: π_old(a|s) para PPO clipping

    def reset(self):
        super().reset()
        self.last_probs = None
        self.last_logits = None
        self.last_temp_scale = 1.0
        self.old_probs = None

    # ── Distribuição e amostragem ────────────────────────────────────────────

    def probabilities(self, inputs, temperature=None):
        """π(a|s) = softmax(logits / T). Guarda logits/probs para o gradiente."""
        t = self.temperature if temperature is None else temperature
        logits = self.forward(inputs)
        self.last_logits = list(logits)
        self.last_temp_scale = 1.0 / t
        probs = softmax([z / t for z in logits])
        self.last_probs = probs
        return probs

    def sample_action(self, inputs, temperature=None):
        """Amostra uma ação da distribuição π (exploração estocástica)."""
        probs = self.probabilities(inputs, temperature)
        r = random.random()
        acc = 0.0
        for a, p in enumerate(probs):
            acc += p
            if r < acc:
                return a
        return self.n_actions - 1

    def log_prob(self, action):
        return math.log(self.last_probs[action] + 1e-12)

    # ── Gradientes da política ───────────────────────────────────────────────

    def grad_log_prob_z(self, action):
        """
        ∇log π(a|s) em relação aos logits: (δ_{a,k} − π_k) / T.
        (derivada da softmax: ∂π_k/∂z_j = π_k·(δ_{jk} − π_j))
        """
        return [self.last_temp_scale * ((1.0 if k == action else 0.0) - p)
                for k, p in enumerate(self.last_probs)]

    def grad_entropy_z(self):
        """∇H(π) em relação aos logits: π_k·(−log π_k − H) / T."""
        H = entropy(self.last_probs)
        return [self.last_temp_scale * p * (-math.log(p + 1e-12) - H)
                for p in self.last_probs]

    # ── PPO: gradiente clipped (Fase 9) ─────────────────────────────────────
    #
    # PPO clipped surrogate:  L = min(r·A, clip(r, 1−ε, 1+ε)·A)
    # onde r = π_new(a|s) / π_old(a|s)  (razão de probabilidades).
    #
    # O gradiente em relação aos logits:
    #   dL/dz_k = A · [δ_{ak} − π_k] · min(1, r/A)   se A > 0
    #           = A · [δ_{ak} − π_k] · max(1, r/A)   se A < 0
    #
    # Isso equivale a "limitar quanto o gradiente pode mudar a política" —
    # previne update grande demais que destrói o que foi aprendido.

    def ppo_grad_log_prob_z(self, action, old_probs_a, advantage, clip_eps):
        """
        Gradiente do PPO clipped surrogate em relação aos logits.

        action:     ação tomada
        old_probs_a: π_old(a|s) — probabilidade da ação na política antiga
        advantage:  A_t (GAE)
        clip_eps:   ε do clipping (ex: 0.2)
        Retorna: lista de gradientes em relação aos logits (5 valores).
        """
        pi_a = self.last_probs[action]
        ratio = pi_a / (old_probs_a + 1e-12)

        # Clipping da razão
        if advantage > 0:
            clipped_ratio = min(ratio, 1.0 + clip_eps)
        else:
            clipped_ratio = max(ratio, 1.0 - clip_eps)

        # Gradiente: A · (δ_{ak} − π_k) · clipped_ratio / T
        # (derivada de log π em relação ao logits, composta com o clipping)
        return [self.last_temp_scale * clipped_ratio * advantage * (
                    (1.0 if k == action else 0.0) - p
                )
                for k, p in enumerate(self.last_probs)]

    # ── Atualização (ponte provisória para a Fase 3) ─────────────────────────
    #
    # Policy gradient por passo (REINFORCE-lite): usa a recompensa imediata como
    # "vantagem". A Fase 3 substitui por retornos descontados + baseline + episódios.
    #
    #   ∇ = reward · ∇logπ(a|s) + β · ∇H(π)      e      θ ← θ + lr·∇ − lr·λ·θ

    def update(self, reward, action, inputs,
               learning_rate=None, entropy_coef=None, regularization=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        if entropy_coef is None:
            entropy_coef = CONFIG['entropy_coef']
        if regularization is None:
            regularization = CONFIG['regularization']

        self.probabilities(inputs)
        grad_z = [
            reward * gl + entropy_coef * ge
            for gl, ge in zip(self.grad_log_prob_z(action), self.grad_entropy_z())
        ]
        self.backward_from_output_grad(grad_z)
        self.apply_gradients(learning_rate, regularization)

    # ── Imitação supervisionada (Estágio A da Fase 4) ────────────────────────
    #
    # Cross-entropy:  loss = −log π(a_professor|s)
    # Como aqui 'grad' é aplicado como θ ← θ + lr·grad, descer a CE equivale a
    # subir o log-prob da ação do professor:  grad = +∇logπ(a_professor|s).

    def imitate(self, inputs, action, learning_rate=None, regularization=None):
        """Passo de imitation learning: CE supervisionada com a ação do professor."""
        if learning_rate is None:
            learning_rate = self.learning_rate
        if regularization is None:
            regularization = CONFIG['regularization']

        self.probabilities(inputs)
        grad_z = self.grad_log_prob_z(action)  # θ ← θ + lr·∇logπ = descida na CE
        self.backward_from_output_grad(grad_z)
        self.apply_gradients(learning_rate, regularization)

    # ── REINFORCE episódico (Fase 3) + A2C (Fase 8) ──────────────────────────
    #
    # Modo REINFORCE (critic=None):
    #   advantage = G_t − baseline   (baseline = média dos retornos)
    #
    # Modo A2C (critic != None):
    #   advantage = GAE(δ_t, γ, λ)   (δ_t = r_t + γV(s_{t+1}) − V(s_t))
    #   + critic_loss = MSE(V(s_t), G_t)
    #
    # Em ambos os modos:  ∇ = advantage · ∇logπ(a|s) + β · ∇H(π)

    def update_episode(self, episode, gamma=None, learning_rate=None,
                       entropy_coef=None, regularization=None, reward_scale=None,
                       imitation_weight=0.0, critic=None, value_coef=0.5,
                       gae_lambda=0.95, ppo_clip=None):
        """
        Update episódico: REINFORCE (Fase 3), A2C (Fase 8) ou PPO (Fase 9).

        Se ppo_clip for fornecido: usa PPO clipped surrogate (Fase 9).
        Se critic for fornecido (sem ppo_clip): usa A2C com GAE (Fase 8).
        Caso contrário: usa REINFORCE com baseline (Fase 3).
        """
        if gamma is None:
            gamma = CONFIG['gamma']
        if learning_rate is None:
            learning_rate = self.learning_rate
        if entropy_coef is None:
            entropy_coef = CONFIG['entropy_coef']
        if regularization is None:
            regularization = CONFIG['regularization']
        if reward_scale is None:
            reward_scale = CONFIG['reward_scale']

        scaled_rewards = [r * reward_scale for _, _, r, *_ in episode]

        # ── Modo A2C/PPO: GAE advantage + critic loss ─────────────────────────
        if critic is not None:
            # Coleta V(s_t) para cada passo
            values = []
            for sensors, *_ in episode:
                values.append(critic.value(sensors))

            # GAE advantage
            advantages = compute_gae(scaled_rewards, values, gamma, gae_lambda)

            # Retornos para o critic loss: G_t = A_t + V(s_t)
            returns = [a + v for a, v in zip(advantages, values)]

            # PPO (Fase 9): coleta π_old(a|s) antes do update
            old_probs_list = None
            if ppo_clip is not None:
                old_probs_list = []
                for sensors, action, *_ in episode:
                    self.probabilities(sensors)
                    old_probs_list.append(self.last_probs[action])

            # Acumula gradientes do actor E do critic
            self.grad_acc = None
            critic.grad_acc = None
            total_entropy = 0.0
            total_critic_loss = 0.0
            action_counts = [0] * self.n_actions

            for t, item in enumerate(episode):
                sensors, action, _, teacher_action = (item + (None, None))[:4]
                advantage = advantages[t]

                # ── Actor: ∇logπ · advantage + β·∇H ─────────────────────────
                self.probabilities(sensors)
                total_entropy += entropy(self.last_probs)
                action_counts[action] += 1

                # PPO (Fase 9): gradiente clipped em vez de advantage puro
                if ppo_clip is not None and old_probs_list is not None:
                    grad_z = [
                        g + entropy_coef * ge
                        for g, ge in zip(
                            self.ppo_grad_log_prob_z(action, old_probs_list[t],
                                                     advantage, ppo_clip),
                            self.grad_entropy_z()
                        )
                    ]
                else:
                    # A2C (Fase 8): advantage puro
                    grad_z = [
                        advantage * gl + entropy_coef * ge
                        for gl, ge in zip(self.grad_log_prob_z(action),
                                          self.grad_entropy_z())
                    ]
                if imitation_weight > 0 and teacher_action is not None:
                    grad_z = [
                        gz + imitation_weight * gt
                        for gz, gt in zip(grad_z, self.grad_log_prob_z(teacher_action))
                    ]
                self.backward_from_output_grad(grad_z)
                self.accumulate_grads()

                # ── Critic: ∇MSE(V(s), G) ───────────────────────────────────
                v_pred = values[t]
                v_target = returns[t]
                total_critic_loss += (v_pred - v_target) ** 2
                # dMSE/dV = 2·(V − G) → backward com grad_output = (V − G)
                critic.forward(sensors)
                critic.backward_from_output_grad([v_pred - v_target])
                critic.accumulate_grads()

            # Aplica gradientes
            self.apply_accumulated(learning_rate, regularization)
            critic.apply_accumulated(learning_rate, regularization)

            return {
                'mean_reward'      : sum(r for _, _, r, *_ in episode) / len(episode),
                'mean_return'      : sum(returns) / len(returns),
                'mean_entropy'     : total_entropy / len(episode),
                'action_counts'    : action_counts,
                'mean_advantage'   : sum(advantages) / len(advantages),
                'critic_loss'      : total_critic_loss / len(episode),
            }

        # ── Modo REINFORCE: baseline = média dos retornos ────────────────────
        returns = compute_returns(scaled_rewards, gamma)
        baseline = sum(returns) / len(returns)

        self.grad_acc = None
        total_entropy = 0.0
        action_counts = [0] * self.n_actions
        for item, G in zip(episode, returns):
            sensors, action, _, teacher_action = (item + (None, None))[:4]
            advantage = G - baseline
            self.probabilities(sensors)
            total_entropy += entropy(self.last_probs)
            action_counts[action] += 1
            grad_z = [
                advantage * gl + entropy_coef * ge
                for gl, ge in zip(self.grad_log_prob_z(action), self.grad_entropy_z())
            ]
            if imitation_weight > 0 and teacher_action is not None:
                grad_z = [
                    gz + imitation_weight * gt
                    for gz, gt in zip(grad_z, self.grad_log_prob_z(teacher_action))
                ]
            self.backward_from_output_grad(grad_z)
            self.accumulate_grads()

        self.apply_accumulated(learning_rate, regularization)

        return {
            'mean_reward'   : sum(r for _, _, r, *_ in episode) / len(episode),
            'mean_return'   : sum(returns) / len(returns),
            'mean_entropy'  : total_entropy / len(episode),
            'action_counts' : action_counts,
        }