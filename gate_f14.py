"""
Gate F14 (legado 15-dim, curto e sem alta demanda).

Treina 35eps A10/B20/C5 seed 42 headless (PPO puro, sem buffer) e avalia
20eps em seeds 42/99/123 medindo % seguro (chegada>0 e perigo==0).

Uso:
  python gate_f14.py
"""
import math
import random
import time

from ursina import Vec3

from config import CONFIG
from mlp import PolicyNetwork, CriticNetwork, save_brain
from perception import get_sensor_inputs, calculate_reward

H = CONFIG['episode_steps']
TURNS = [-1.0, -0.5, 0.0, 0.5, 1.0]


class FakeEnt:
    def __init__(self, x, z):
        self.position = Vec3(x, 1, z)


class FakeWorm:
    def __init__(self, x, z):
        self.head = FakeEnt(x, z)
        self.direction = Vec3(0, 0, 1)


class FakeEnv:
    def __init__(self):
        self.light_sources = []
        self.rain_sources = []
        self.obstacles = []

    def randomize(self, lim=15):
        self.light_sources = [FakeEnt(random.uniform(-lim, lim), random.uniform(-lim, lim))]
        self.rain_sources = [FakeEnt(random.uniform(-lim, lim), random.uniform(-lim, lim))]
        self.obstacles = [FakeEnt(random.uniform(-lim, lim), random.uniform(-lim, lim)) for _ in range(4)]


def _sang(d, t):
    cur = math.atan2(d.z, d.x)
    tgt = math.atan2(t.z, t.x)
    a = tgt - cur
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _turnv(d, ang):
    c, s = math.cos(ang), math.sin(ang)
    return Vec3(d.x * c - d.z * s, 0, d.x * s + d.z * c)


def teacher_dir(worm, env):
    cx = cz = 0.0
    for rs in env.rain_sources:
        tox = rs.position.x - worm.head.position.x
        toz = rs.position.z - worm.head.position.z
        d = math.hypot(tox, toz)
        if d > 0.01:
            w = max(0.0, 1.0 - d / CONFIG['sensor_max_dist'])
            cx += tox / d * w
            cz += toz / d * w
    for ls in env.light_sources:
        fx = worm.head.position.x - ls.position.x
        fz = worm.head.position.z - ls.position.z
        d = math.hypot(fx, fz)
        if d > 0.01:
            w = max(0.0, 1.0 - d / CONFIG['light_repel_dist'])
            cx += fx / d * w
            cz += fz / d * w
    lim = CONFIG['map_limit']
    m = CONFIG['wall_margin']
    p = worm.head.position
    if p.x < -lim + m:
        cx += (1.0 + (-lim + m - p.x) / m)
    if p.x > lim - m:
        cx -= (1.0 + (p.x - (lim - m)) / m)
    if p.z < -lim + m:
        cz += (1.0 + (-lim + m - p.z) / m)
    if p.z > lim - m:
        cz -= (1.0 + (p.z - (lim - m)) / m)
    if abs(cx) + abs(cz) > 0.01:
        return Vec3(cx, 0, cz).normalized()
    return worm.direction


def run_episode(brain, env, train, autonomy, lam, critic, dt=1 / 60):
    worm = FakeWorm(random.uniform(-10, 10), random.uniform(-10, 10))
    worm.direction = Vec3(random.uniform(-1, 1), 0, random.uniform(-1, 1)).normalized()
    state = {'rain_pulse': 0.0, 'prev_dist_light': None, 'prev_dist_rain': None,
             'prev_dist_obs': None, 'prev_position': None}
    traj = []
    steps_rain = 0
    steps_danger = 0
    tot = 0.0
    mt = CONFIG['turn_rate'] * dt
    for _ in range(H):
        s = get_sensor_inputs(worm, env, state)
        a = brain.sample_action(s)
        td = teacher_dir(worm, env)
        tt = max(-mt, min(mt, _sang(worm.direction, td)))
        turn = tt * (1 - autonomy) + TURNS[a] * mt * autonomy
        worm.direction = _turnv(worm.direction, turn).normalized()
        worm.head.position.x += worm.direction.x * CONFIG['speed'] * dt
        worm.head.position.z += worm.direction.z * CONFIG['speed'] * dt
        lim = CONFIG['map_limit']
        worm.head.position.x = max(-lim, min(lim, worm.head.position.x))
        worm.head.position.z = max(-lim, min(lim, worm.head.position.z))
        r = calculate_reward(worm, env, state)
        td2 = teacher_dir(worm, env)
        des = _sang(worm.direction, td2)
        best = min(range(5), key=lambda k: abs(TURNS[k] * mt - des))
        traj.append((s, a, r, best))
        tot += r
        dr = min(math.hypot(worm.head.position.x - rs.position.x,
                            worm.head.position.z - rs.position.z) for rs in env.rain_sources)
        dl = min(math.hypot(worm.head.position.x - ls.position.x,
                            worm.head.position.z - ls.position.z) for ls in env.light_sources)
        if dr < CONFIG['arrival_radius']:
            steps_rain += 1
        if dl < CONFIG['arrival_radius']:
            steps_danger += 1
    if train:
        brain.update_episode(traj, imitation_weight=lam, critic=critic,
                             value_coef=CONFIG['value_coef'],
                             gae_lambda=CONFIG['gae_lambda'],
                             ppo_clip=CONFIG['ppo_clip'])
    seguro = 1 if (steps_rain > 0 and steps_danger == 0) else 0
    return tot, seguro


def main():
    t0 = time.time()
    random.seed(CONFIG['seed'])
    brain = PolicyNetwork(CONFIG['n_inputs'], CONFIG['n_hidden'], CONFIG['n_actions'],
                          temperature=CONFIG['temperature'], n_hidden2=CONFIG.get('n_hidden2'))
    critic = CriticNetwork(CONFIG['n_inputs'], CONFIG['n_hidden'],
                           n_hidden2=CONFIG.get('n_hidden2'))
    env = FakeEnv()
    A = CONFIG['stage_a_episodes']
    B = CONFIG['stage_b_episodes']
    for ep in range(35):
        env.randomize()
        if ep < A:
            aut, lam = 0.0, 0.0
        elif ep < A + B:
            k = ep - A
            aut = min(1.0, (k + 1) / B)
            lam = CONFIG['lambda_start'] * (CONFIG['lambda_decay'] ** k)
        else:
            aut, lam = 1.0, float(CONFIG.get('lambda_c', 0.0) or 0.0)
        run_episode(brain, env, True, aut, lam, critic)
        brain.learning_rate *= CONFIG['lr_decay']
        brain.temperature = max(CONFIG['min_temperature'],
                                brain.temperature * CONFIG['temperature_decay'])
    save_brain('pesos_gate15.json', brain, critic)
    print(f"TREINO 35eps 15-dim ok em {time.time()-t0:.1f}s -> pesos_gate15.json")
    brain.temperature = CONFIG['min_temperature']
    res = {}
    for seed in (42, 99, 123):
        random.seed(seed)
        segs = []
        for _ in range(20):
            env.randomize()
            _, s = run_episode(brain, env, False, 1.0, 0.0, critic)
            segs.append(s)
        res[seed] = sum(segs) / len(segs) * 100
        print(f"eval seed {seed}: seguro={res[seed]:.1f}% ({sum(segs)}/20)")
    gap = max(res.values()) - min(res.values())
    media = sum(res.values()) / len(res)
    print(f"MEDIA seguro={media:.1f}% GAP={gap:.1f}pp DoD>=80%: {'SIM' if media >= 80 else 'NAO'}")
    print(f"OVERFIT: {'SIM' if gap > 15 else 'NAO'} (gate >15pp)")
    return 0 if media >= 80 else 1


if __name__ == '__main__':
    raise SystemExit(main())
