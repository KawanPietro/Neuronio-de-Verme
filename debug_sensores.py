"""
Debug do contrato de comportamento (Fase 1).

Imprime o estado 8-dim e a recompensa por progresso a cada passo, mostrando que
`r` é MAIOR quando o verme se aproxima da chuva (e menor quando se afasta).

Headless: usa entidades falsas em vez de abrir a janela do Ursina.
Rode com: python debug_sensores.py
"""
from ursina import Vec3

from config import CONFIG
from perception import calculate_reward, get_sensor_inputs


class FakeEntity:
    def __init__(self, x=0.0, y=1.0, z=0.0):
        self.position = Vec3(x, y, z)


class FakeWorm:
    def __init__(self, x=0.0, y=1.25, z=0.0):
        self.head = FakeEntity(x, y, z)


class FakeEnv:
    def __init__(self, rain, light):
        self.rain_sources = rain
        self.light_sources = light


def new_state():
    return {
        'total_reward'    : 0.0,
        'rain_pulse'      : 0.0,
        'prev_dist_light' : None,
        'prev_dist_rain'  : None,
        'prev_position'   : None,
    }


def run_phase(worm, env, state, steps, dir_x, label):
    """Move o verme em linha reta no eixo X por `steps` passos, imprimindo sensores + r."""
    dt = 1 / 60
    rain = env.rain_sources[0]
    print(f"\n-- FASE {label}: {'aproximando' if dir_x < 0 else 'afastando'} da chuva --")
    rewards = []
    for step in range(steps):
        worm.head.position.x = worm.head.position.x + dir_x * CONFIG['speed'] * dt
        sensors = get_sensor_inputs(worm, env, state)
        r = calculate_reward(worm, env, state)
        rewards.append(r)
        dist = abs(rain.position.x - worm.head.position.x)
        sens = ", ".join(f"{s:+.2f}" for s in sensors)
        print(f"passo {step:3d} | sensores=[{sens}] | r={r:+.3f} | dist_chuva={dist:6.1f}")
    return rewards


def main():
    rain  = FakeEntity(x=-10, y=1, z=0)
    light = FakeEntity(x=+10, y=1, z=0)
    env   = FakeEnv(rain=[rain], light=[light])
    worm  = FakeWorm(x=0, y=1.25, z=0)
    state = new_state()

    rewards_away   = run_phase(worm, env, state, 60,  dir_x=+1, label="A (afastando)")
    rewards_toward = run_phase(worm, env, state, 180, dir_x=-1, label="B (aproximando)")

    mean_away   = sum(rewards_away) / len(rewards_away)
    mean_toward = sum(rewards_toward) / len(rewards_toward)

    print("\n" + "=" * 60)
    print(f"Recompensa média afastando da chuva : {mean_away:+.4f}")
    print(f"Recompensa média aproximando da chuva: {mean_toward:+.4f}")
    ok_contract = mean_toward > mean_away
    print("OK: a recompensa é maior quando o verme se aproxima da chuva."
          if ok_contract else "FALHOU: revise a recompensa por progresso.")

    # ── Casos-limite (Fase 6) ─────────────────────────────────────────────────
    ok_cases = test_casos_limite()

    return 0 if (ok_contract and ok_cases) else 1


def test_casos_limite():
    """Casos-limite: sem fontes, fonte no limite do mapa e worm parado (idle)."""
    all_ok = True

    # C1 — Sem fontes no mapa: sensores zerados e recompensa neutra (sem crash).
    env = FakeEnv(rain=[], light=[])
    worm = FakeWorm(x=5, y=1.25, z=5)
    state = new_state()
    sensors = get_sensor_inputs(worm, env, state)
    r = calculate_reward(worm, env, state)
    c1 = (all(abs(s) < 1e-9 for s in sensors)) and (abs(r) < 1e-9)
    all_ok &= c1
    print(f"[{'OK' if c1 else 'FALHOU'}] sem fontes: sensores zerados, r neutro={r:+.3f}")

    # C2 — Fonte além do alcance dos sensores: features saturam em [0,1], sem crash.
    rain  = FakeEntity(x=100, y=1, z=0)
    light = FakeEntity(x=-100, y=1, z=0)
    env = FakeEnv(rain=[rain], light=[light])
    worm = FakeWorm(x=0, y=1.25, z=0)
    state = new_state()
    sensors = get_sensor_inputs(worm, env, state)
    r = calculate_reward(worm, env, state)
    c2 = (sensors[2] == 1.0 and sensors[6] == 1.0)
    all_ok &= c2
    print(f"[{'OK' if c2 else 'FALHOU'}] fonte no limite: dist satura em 1.0, r={r:+.3f}")

    # C3 — Worm parado: idle_cost desativado (Fase 7) → r >= 0 (sem penalidade)
    rain  = FakeEntity(x=-10, y=1, z=0)
    light = FakeEntity(x=+10, y=1, z=0)
    env = FakeEnv(rain=[rain], light=[light])
    worm = FakeWorm(x=0, y=1.25, z=0)
    state = new_state()
    calculate_reward(worm, env, state)   # inicializa prev_*
    r = calculate_reward(worm, env, state)  # segundo passo, worm parado
    c3 = r >= 0  # idle_cost = 0, mas proximity pode dar r > 0
    all_ok &= c3
    print(f"[{'OK' if c3 else 'FALHOU'}] idle: worm parado sem penalidade, r={r:+.3f}")

    return all_ok


if __name__ == '__main__':
    raise SystemExit(main())