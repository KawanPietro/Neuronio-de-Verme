"""
Testes do MLP, política e A2C — critérios de aceite das Fases 2/3/8.

Verifica que o `backward` calcula gradientes corretos (diferenças finitas)
e que os componentes A2C (Critic, GAE) funcionam.

Rode com: python test_mlp.py
"""
import math
import random

from mlp import MLP, PolicyNetwork, CriticNetwork, compute_returns, compute_gae, save_brain, load_brain


# ─── Acesso genérico a parâmetros ─────────────────────────────────────────────

def param_indices(net):
    for name in ('w_input_hidden', 'bias_hidden', 'w_hidden_output', 'bias_output'):
        tensor = getattr(net, name)
        if name.startswith('w'):
            for i, row in enumerate(tensor):
                for j in range(len(row)):
                    yield name, (i, j)
        else:
            for i in range(len(tensor)):
                yield name, (i,)


def get_param(holder, name, idx):
    """Lê um parâmetro de um objeto (atributo) ou de um dict de tensores."""
    t = holder[name] if isinstance(holder, dict) else getattr(holder, name)
    for i in idx:
        t = t[i]
    return t


def set_param(net, name, idx, value):
    if len(idx) == 1:
        getattr(net, name)[idx[0]] = value
    else:
        getattr(net, name)[idx[0]][idx[1]] = value


def numeric_grad(net, loss_fn, eps=1e-6):
    """Gradiente numérico de loss_fn por diferenças finitas centrais."""
    grads = {}
    for name, idx in param_indices(net):
        orig = get_param(net, name, idx)
        set_param(net, name, idx, orig + eps)
        plus = loss_fn()
        set_param(net, name, idx, orig - eps)
        minus = loss_fn()
        set_param(net, name, idx, orig)
        grads[(name, idx)] = (plus - minus) / (2 * eps)
    return grads


def compare(analytic_grads, numeric_grads, tol=2e-3):
    """Retorna (maior erro relativo, lista de falhas)."""
    worst = 0.0
    failed = []
    for (name, idx), num in numeric_grads.items():
        ana = get_param(analytic_grads, name, idx)
        denom = max(1e-6, abs(num))
        rel = abs(ana - num) / denom
        worst = max(worst, rel)
        if rel > tol:
            failed.append((name, idx, ana, num))
    return worst, failed


def check(title, net, loss_fn, analytic_grad_fn):
    analytic_grads = analytic_grad_fn()
    numeric = numeric_grad(net, loss_fn)
    worst, failed = compare(analytic_grads, numeric)
    status = "OK" if not failed else "FALHOU"
    print(f"[{status}] {title}  (maior erro relativo: {worst:.2e})")
    for name, idx, ana, num in failed[:5]:
        print(f"        {name}{idx}: analitico={ana:.6f} numerico={num:.6f}")
    return not failed


# ─── Objetivos de teste ───────────────────────────────────────────────────────

def mse_loss_fn(net, inputs, targets):
    def f():
        out = net.forward(inputs)
        return 0.5 * sum((o - t) ** 2 for o, t in zip(out, targets))
    return f


def mse_grad(net, inputs, targets):
    out = net.forward(inputs)
    derivs = net.output_activation_derivs()
    grad_z = [(o - t) * d for o, t, d in zip(out, targets, derivs)]
    net.backward_from_output_grad(grad_z)
    return net.grads()


def policy_loss_fn(net, inputs, action):
    def f():
        probs = net.probabilities(inputs)
        return -math.log(probs[action] + 1e-12)
    return f


def policy_grad(net, inputs, action):
    net.probabilities(inputs)
    # L = -log pi(a|s)  =>  dL/dz = -grad_log_prob_z
    grad_z = [-g for g in net.grad_log_prob_z(action)]
    net.backward_from_output_grad(grad_z)
    return net.grads()


# ─── Execução ─────────────────────────────────────────────────────────────────

def test_returns():
    """G_t = r_t + γ·G_{t+1}: verifica a conta manual de retornos descontados."""
    r = compute_returns([1, 0, 0], 0.5)
    assert r == [1.0, 0.0, 0.0], r
    r = compute_returns([1, 1, 1], 0.9)
    assert abs(r[0] - (1 + 0.9 + 0.81)) < 1e-12
    assert abs(r[1] - (1 + 0.9)) < 1e-12
    assert abs(r[2] - 1) < 1e-12
    print("[OK] compute_returns -- retornos descontados G_t")
    return True


def test_update_episode():
    """REINFORCE episódico: roda e provoca mudança nos pesos (gradiente real)."""
    random.seed(42)
    net = PolicyNetwork(8, 16, 5, hidden_activation='tanh', temperature=1.0)

    sensors = [random.uniform(-1, 1) for _ in range(8)]
    episode = [(sensors, random.randrange(5), random.uniform(-0.5, 0.5)) for _ in range(30)]

    before = net.w_hidden_output[0][0]
    stats = net.update_episode(episode, gamma=0.9, learning_rate=0.01,
                               entropy_coef=0.01, regularization=0.0,
                               reward_scale=1.0)
    after = net.w_hidden_output[0][0]
    assert before != after, "gradiente nulo: update_episode não mudou os pesos"
    assert len(stats['action_counts']) == 5
    assert sum(stats['action_counts']) == len(episode)
    assert stats['mean_entropy'] > 0
    print("[OK] update_episode -- REINFORCE com baseline aplica gradiente")
    return True


def test_imitate():
    """Imitação (CE supervisionada): reduz −log π(a_professor|s) num passo."""
    random.seed(42)
    net = PolicyNetwork(8, 16, 5, hidden_activation='tanh', temperature=1.0)
    sensors = [random.uniform(-1, 1) for _ in range(8)]
    action = 4
    before = -math.log(net.probabilities(sensors)[action] + 1e-12)
    net.imitate(sensors, action, learning_rate=0.05, regularization=0.0)
    after = -math.log(net.probabilities(sensors)[action] + 1e-12)
    assert after < before, f"CE subiu: {before:.4f} -> {after:.4f}"
    print("[OK] imitate -- imitacao supervisionada reduz a perda CE")
    return True


def test_save_load():
    """Persistência (Fase 5): salvar e carregar pesos preserva a política."""
    import json
    import tempfile
    import os
    random.seed(42)
    net = PolicyNetwork(8, 16, 5, hidden_activation='tanh', temperature=1.0)
    path = os.path.join(tempfile.gettempdir(), 'teste_pesos.json')

    sensors = [random.uniform(-1, 1) for _ in range(8)]
    before = net.probabilities(sensors)
    net.save_weights(path)
    net.reset()  # destrói os pesos

    net.load_weights(path)
    after = net.probabilities(sensors)
    os.remove(path)

    assert before == after, "pesos nao preservados no ciclo salvar/carregar"
    print("[OK] save/load -- pesos preservados em JSON (ciclo completo)")
    return True


def main():
    random.seed(42)
    inputs  = [random.uniform(-1, 1) for _ in range(8)]
    targets = [random.uniform(-1, 1) for _ in range(3)]

    results = [test_returns(), test_imitate(), test_save_load()]

    # 1 e 2 — Regressão (MSE) com tanh e relu na camada oculta
    for act in ('tanh', 'relu'):
        net = MLP(8, 16, 3, hidden_activation=act, output_activation='tanh')
        ok = check(
            f"MLP {act} -- regressao (MSE, saida tanh)",
            net,
            mse_loss_fn(net, inputs, targets),
            lambda: mse_grad(net, inputs, targets),
        )
        results.append(ok)

    # 3 e 4 — Política estocástica (−log π) com tanh e relu
    for act in ('tanh', 'relu'):
        net = PolicyNetwork(8, 16, 5, hidden_activation=act, temperature=1.0)
        ok = check(
            f"Policy {act} -- softmax 5 acoes (L = -log pi)",
            net,
            policy_loss_fn(net, inputs, action=2),
            lambda: policy_grad(net, inputs, action=2),
        )
        results.append(ok)

    # 5 — REINFORCE episódico com baseline (Fase 3)
    results.append(test_update_episode())

    # 6 — CriticNetwork: V(s) responde ao backward (Fase 8)
    results.append(test_critic())

    # 7 — GAE: advantage computa corretamente (Fase 8)
    results.append(test_gae())

    # 8 — A2C update_episode: actor + critic mudam pesos (Fase 8)
    results.append(test_a2c_update())

    # 9 — save/load combinado (actor + critic, Fase 8)
    results.append(test_save_load_brain())

    print()
    if all(results):
        print("Todos os testes conferem.")
        return 0
    print("Ha divergencias — revise o codigo.")
    return 1


# ─── Testes da Fase 8 (A2C) ─────────────────────────────────────────────────

def test_critic():
    """CriticNetwork: V(s) muda após backward (gradiente real)."""
    random.seed(42)
    critic = CriticNetwork(8, n_hidden=16)
    sensors = [random.uniform(-1, 1) for _ in range(8)]

    v_before = critic.value(sensors)
    # Forward + backward com grad = (V − target)
    critic.forward(sensors)
    critic.backward_from_output_grad([v_before - 1.0])  # target = 1.0
    # Aplica um passo grande para garantir mudança
    for i in range(critic.n_inputs):
        for j in range(critic.n_hidden):
            critic.w_input_hidden[i][j] -= 0.1 * critic.grad_w_input_hidden[i][j]
    for j in range(critic.n_hidden):
        critic.bias_hidden[j] -= 0.1 * critic.grad_bias_hidden[j]
    for j in range(critic.n_hidden):
        critic.w_hidden_output[j][0] -= 0.1 * critic.grad_w_hidden_output[j][0]
    for k in range(critic.n_outputs):
        critic.bias_output[k] -= 0.1 * critic.grad_bias_output[k]

    v_after = critic.value(sensors)
    assert v_before != v_after, "Critic nao mudou apos backward"
    print("[OK] CriticNetwork -- V(s) responde ao backward")
    return True


def test_gae():
    """GAE: advantage calcula corretamente para episodio simples."""
    # Episodio de 3 passos: r=[1, 1, 1], V=[0.5, 0.5, 0.5], gamma=0.99, lambda=0.95
    rewards = [1.0, 1.0, 1.0]
    values  = [0.5, 0.5, 0.5]
    gamma   = 0.99
    lam     = 0.95

    advantages = compute_gae(rewards, values, gamma, lam)

    # δ_2 = 1.0 + 0.99*0 − 0.5 = 0.5 ;  A_2 = 0.5
    # δ_1 = 1.0 + 0.99*0.5 − 0.5 = 0.995 ;  A_1 = 0.995 + 0.99*0.95*0.5 = 1.46525
    # δ_0 = 1.0 + 0.99*0.5 − 0.5 = 0.995 ;  A_0 = 0.995 + 0.99*0.95*1.46525 ≈ 2.33087
    assert len(advantages) == 3
    assert abs(advantages[2] - 0.5) < 0.01, f"A[2] errado: {advantages[2]}"
    assert advantages[1] > advantages[2], "A[1] deveria ser > A[2]"
    assert advantages[0] > advantages[1], "A[0] deveria ser > A[1]"
    print("[OK] GAE -- advantage computa corretamente")
    return True


def test_a2c_update():
    """A2C update_episode: actor + critic mudam pesos."""
    random.seed(42)
    actor  = PolicyNetwork(8, 16, 5, temperature=1.0)
    critic = CriticNetwork(8, n_hidden=16)

    sensors = [random.uniform(-1, 1) for _ in range(8)]
    episode = [(sensors, random.randrange(5), random.uniform(-0.5, 0.5)) for _ in range(30)]

    w_actor_before  = actor.w_hidden_output[0][0]
    w_critic_before = critic.w_hidden_output[0][0]

    stats = actor.update_episode(episode, gamma=0.9, learning_rate=0.01,
                                 entropy_coef=0.01, regularization=0.0,
                                 reward_scale=1.0, critic=critic,
                                 value_coef=0.5, gae_lambda=0.95)

    w_actor_after  = actor.w_hidden_output[0][0]
    w_critic_after = critic.w_hidden_output[0][0]

    assert w_actor_before != w_actor_after, "Actor nao mudou"
    assert w_critic_before != w_critic_after, "Critic nao mudou"
    assert 'mean_advantage' in stats, "Stats sem mean_advantage"
    assert 'critic_loss' in stats, "Stats sem critic_loss"
    print("[OK] A2C update_episode -- actor + critic atualizam pesos")
    return True


def test_save_load_brain():
    """Save/load combinado: actor + critic preservados em JSON."""
    import tempfile
    import os
    random.seed(42)
    actor  = PolicyNetwork(8, 16, 5, temperature=1.0)
    critic = CriticNetwork(8, n_hidden=16)
    path = os.path.join(tempfile.gettempdir(), 'teste_brain.json')

    sensors = [random.uniform(-1, 1) for _ in range(8)]
    probs_before = actor.probabilities(sensors)
    v_before = critic.value(sensors)

    save_brain(path, actor, critic)
    actor.reset()
    critic.reset()

    load_brain(path, actor, critic)
    probs_after = actor.probabilities(sensors)
    v_after = critic.value(sensors)

    ok_actor  = all(abs(a - b) < 1e-9 for a, b in zip(probs_before, probs_after))
    ok_critic = abs(v_before - v_after) < 1e-9
    os.remove(path)

    assert ok_actor, "Actor probs divergem apos load"
    assert ok_critic, "Critic value diverge apos load"
    print("[OK] save/load brain -- actor + critic preservados")
    return True


if __name__ == '__main__':
    raise SystemExit(main())