"""
Debug do modo alimento 11-dim (seção F8 do README.md).

Verifica sem abrir a janela do Ursina:
- contrato: r maior aproximando do alimento do que afastando
- C1 sem alimento / C2 longe / C3 parado / C4 faixas / C5 fome / C6 evento comer

Rode com: python debug_food.py
Nao toca no legado 17-dim (debug_sensores.py continua válido).
"""
from ursina import Vec3

from config import CONFIG
from perception import (
    FOOD_DIM,
    calculate_food_reward,
    get_food_sensor_inputs,
)


class FakeEntity:
    def __init__(self, x=0.0, y=1.0, z=0.0):
        self.position = Vec3(x, y, z)


class FakeWorm:
    def __init__(self, x=0.0, y=1.25, z=0.0):
        self.head = FakeEntity(x, y, z)
        self.direction = Vec3(0, 0, 1)


class FakeEnv:
    def __init__(self, foods, obstacles=None):
        self.food_sources = foods
        self.rain_sources = []  # legado vazio p/ provar que food é independente
        self.light_sources = []
        self.obstacles = obstacles if obstacles is not None else []


def new_state(hunger=0.0):
    return {
        'hunger': hunger,
        'prev_dist_food': None,
        'prev_dist_obs': None,
        'prev_position': None,
    }


def run_phase(worm, env, state, steps, dir_x, label):
    dt = 1 / 60
    print(f"\n-- FASE {label} --")
    rewards = []
    for step in range(steps):
        worm.head.position.x = worm.head.position.x + dir_x * CONFIG['speed'] * dt
        sensors = get_food_sensor_inputs(worm, env, state)
        r = calculate_food_reward(worm, env, state)
        rewards.append(r)
        assert len(sensors) == FOOD_DIM == 11, f"dim errada: {len(sensors)}"
        if step % 30 == 0:
            sens = ", ".join(f"{s:+.2f}" for s in sensors)
            print(f"passo {step:3d} | sensores=[{sens}] | r={r:+.3f} hunger={state['hunger']:.2f}")
    return rewards


def main():
    food = FakeEntity(x=-10, y=1, z=0)
    env = FakeEnv(foods=[food])
    worm = FakeWorm(x=0, y=1.25, z=0)

    # Contrato principal: aproximar rende mais que afastar
    state = new_state()
    away = run_phase(worm, env, state, 60, dir_x=+1, label="A (afastando)")
    toward = run_phase(worm, env, state, 120, dir_x=-1, label="B (aproximando)")
    mean_away = sum(away) / len(away)
    mean_toward = sum(toward) / len(toward)
    print("\n" + "=" * 60)
    print(f"Media afastando : {mean_away:+.4f}")
    print(f"Media aproximando: {mean_toward:+.4f}")
    ok_contract = mean_toward > mean_away
    print("OK: aproximar rende mais." if ok_contract else "FALHOU: contrato alimento.")

    ok_cases = True

    # C1 — sem alimento: sensores food zerados, r neutro, sem crash
    env0 = FakeEnv(foods=[])
    worm0 = FakeWorm(x=5, y=1.25, z=5)
    s0 = new_state()
    sens = get_food_sensor_inputs(worm0, env0, s0)
    r = calculate_food_reward(worm0, env0, s0)
    c1 = (len(sens) == 11 and sens[0] == 0.0 and sens[1] == 0.0
          and sens[2] == 0.0 and sens[3] == 0.0 and abs(r) < 1e-9)
    print(f"[{'OK' if c1 else 'FALHOU'}] C1 sem alimento: food zerado, r={r:+.3f}")
    ok_cases &= c1

    # C2 — alimento longe: dist satura, smell ~0, sem NaN
    far = FakeEntity(x=100, y=1, z=0)
    env2 = FakeEnv(foods=[far])
    worm2 = FakeWorm(x=0, y=1.25, z=0)
    s2 = new_state()
    sens2 = get_food_sensor_inputs(worm2, env2, s2)
    r2 = calculate_food_reward(worm2, env2, s2)
    c2 = (sens2[2] == 1.0 and 0.0 <= sens2[3] < 0.05 and abs(r2) < 2.0)
    print(f"[{'OK' if c2 else 'FALHOU'}] C2 longe: dist=1.0 smell={sens2[3]:.3f} r={r2:+.3f}")
    ok_cases &= c2

    # C3 — parado: sem penalidade (idle_cost=0)
    env3 = FakeEnv(foods=[FakeEntity(x=-10, y=1, z=0)])
    worm3 = FakeWorm(x=0, y=1.25, z=0)
    s3 = new_state()
    calculate_food_reward(worm3, env3, s3)
    r3 = calculate_food_reward(worm3, env3, s3)
    c3 = r3 >= -0.05
    print(f"[{'OK' if c3 else 'FALHOU'}] C3 parado: r={r3:+.3f} (sem punicao)")
    ok_cases &= c3

    # C4 — faixas 11-dim
    sens4 = get_food_sensor_inputs(FakeWorm(), FakeEnv([FakeEntity(x=-10, y=1, z=0)]), new_state())
    c4 = (len(sens4) == 11 and -1.0 <= sens4[7] <= 1.0 and -1.0 <= sens4[8] <= 1.0
          and 0.0 <= sens4[9] <= 1.0 and 0.0 <= sens4[10] <= 1.0
          and 0.0 <= sens4[2] <= 1.0 and 0.0 <= sens4[3] <= 1.0)
    print(f"[{'OK' if c4 else 'FALHOU'}] C4 faixas 11-dim OK")
    ok_cases &= c4

    # C5 — fome multiplica (mesmo passo, hunger 1 > hunger 0 quando r>0)
    env5 = FakeEnv(foods=[FakeEntity(x=-5, y=1, z=0)])
    worm5a = FakeWorm(x=0, y=1.25, z=0)
    worm5b = FakeWorm(x=0, y=1.25, z=0)
    sa = new_state(hunger=0.0)
    sb = new_state(hunger=1.0)
    calculate_food_reward(worm5a, env5, sa)  # init prev
    calculate_food_reward(worm5b, env5, sb)
    worm5a.head.position.x -= 0.1  # passo realista (speed*dt=0.1): evita clip [-1,1]
    worm5b.head.position.x -= 0.1
    ra = calculate_food_reward(worm5a, env5, sa)
    rb = calculate_food_reward(worm5b, env5, sb)
    c5 = rb > ra  # progresso positivo amplificado pela fome
    print(f"[{'OK' if c5 else 'FALHOU'}] C5 fome: saciado={ra:+.3f} faminto={rb:+.3f}")
    ok_cases &= c5

    # C6 — evento comer: cruzar food_radius dá salto de recompensa
    fr = CONFIG.get('food_radius', 3.0)
    env6 = FakeEnv(foods=[FakeEntity(x=0, y=1, z=0)])
    worm6 = FakeWorm(x=fr + 0.5, y=1.25, z=0)
    s6 = new_state()
    calculate_food_reward(worm6, env6, s6)  # prev fora
    worm6.head.position.x = 0.0  # entra no raio
    r6 = calculate_food_reward(worm6, env6, s6)
    c6 = r6 > 0.3  # eat_bonus domina
    print(f"[{'OK' if c6 else 'FALHOU'}] C6 comer: r_entrada={r6:+.3f} (bonus esperado)")
    ok_cases &= c6

    all_ok = ok_contract and ok_cases
    print("=" * 60)
    print("debug_food: TUDO OK" if all_ok else "debug_food: FALHOU")
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
