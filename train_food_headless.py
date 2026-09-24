"""
Trainer headless do modo alimento (T5, Lean: feedback rápido sem GUI).

Reusa o cérebro real (PolicyNetwork 11-dim + Critic + PPO) e a percepção real
(get_food_sensor_inputs / calculate_food_reward), mas com física simplificada
em grade (sem Ursina window). Serve para estimar a curva da pergunta-base
"épocas até o alimento no menor tempo" em segundos, antes de runs GUI caros.

Uso:
  python train_food_headless.py --episodes=10 --maze=A --seed=42
  python train_food_headless.py --episodes=35 --maze=A --seed=42 --eval-only --weights=pesos_food.json
"""
import json
import math
import random
import sys

from ursina import Vec3

from config import CONFIG
from mlp import PolicyNetwork, CriticNetwork, save_brain, load_brain
from perception import get_food_sensor_inputs, calculate_food_reward

TURN_MULTS = [-1.0, -0.5, 0.0, 0.5, 1.0]


class FakeEnt:
    def __init__(self, x, z):
        self.position = Vec3(x, 1, z)
        self.x, self.z = x, z


class HeadlessWorm:
    def __init__(self, x, z):
        self.head = FakeEnt(x, z)
        self.direction = Vec3(0, 0, 1)


class HeadlessMaze:
    """Muros como células bloqueadas (mesma regra do debug_maze)."""
    def __init__(self, key, cell=1.0):
        with open(CONFIG.get('maze_file', 'labirintos.json')) as f:
            data = json.load(f)
        aliases = {'a': 'a_treino', 'b': 'b_teste'}
        key = aliases.get(key.lower(), key.lower())
        self.maze = data[key]
        self.key = key
        self.limit = self.maze.get('map_limit', 16)
        self.cell = cell
        self.blocked = set()
        self._build_blocked()
        # obstáculos p/ percepção: pedras + pontos amostrados dos muros
        self.obstacles = []
        for p in self.maze.get('pedras', []):
            self.obstacles.append(FakeEnt(p['x'], p['z']))
        for w in self.maze.get('muros', []):
            for t in (-0.4, 0.0, 0.4):
                ang = math.radians(w.get('angle', 0))
                dx, dz = math.cos(ang), math.sin(ang)
                L = w.get('length', 9.0)
                self.obstacles.append(
                    FakeEnt(w['x'] + dx * L * t, w['z'] - dz * L * t))
        self.food_sources = []
        self.spawn = self.maze.get('spawn', [0, 1.25, -12])
        self.respawn_food()

    def _rect(self, w):
        m = 1.0
        L, th, ang = w.get('length', 9.0), w.get('thickness', 1.4), w.get('angle', 0)
        if ang in (0, 180, -180):
            return w['x'], w['z'], L / 2 + m, th / 2 + m
        if ang in (90, -90, 270, -270):
            return w['x'], w['z'], th / 2 + m, L / 2 + m
        r = (abs(math.cos(math.radians(ang))) * L + abs(math.sin(math.radians(ang))) * th) / 2 + m
        return w['x'], w['z'], r, r

    def _build_blocked(self):
        def tg(x, z):
            return (int(round((x + self.limit) / self.cell)),
                    int(round((z + self.limit) / self.cell)))
        for w in self.maze.get('muros', []):
            cx, cz, hx, hz = self._rect(w)
            a, b = tg(cx - hx, cz - hz)
            c, d = tg(cx + hx, cz + hz)
            for gx in range(min(a, c), max(a, c) + 1):
                for gz in range(min(b, d), max(b, d) + 1):
                    self.blocked.add((gx, gz))

    def _grid(self, x, z):
        return (int(round((x + self.limit) / self.cell)),
                int(round((z + self.limit) / self.cell)))

    def free(self, x, z):
        if abs(x) > self.limit or abs(z) > self.limit:
            return False
        return self._grid(x, z) not in self.blocked

    def respawn_food(self, cx=None, cz=None, max_dist=None):
        """T8: sem args = uniforme legado; com (cx,cz,max) = currículo proximal."""
        if cx is not None and max_dist is not None:
            _min_d = CONFIG.get('food_radius', 3.0) + 1.0  # anti-sorte
            for _ in range(20):
                ang = random.uniform(0, 2 * math.pi)
                dist = random.uniform(_min_d, max(_min_d + 0.5, max_dist))
                x, z = cx + math.cos(ang) * dist, cz + math.sin(ang) * dist
                if self.free(x, z):
                    self.food_sources = [FakeEnt(x, z)]
                    return
        for _ in range(40):
            x = random.uniform(-self.limit + 2, self.limit - 2)
            z = random.uniform(-self.limit + 2, self.limit - 2)
            if self.free(x, z):
                self.food_sources = [FakeEnt(x, z)]
                return
        self.food_sources = [FakeEnt(0, 0)]

    def food_max_dist(self, ep):
        if not CONFIG.get('food_curriculum', 0):
            return CONFIG.get('food_limit', 12)
        return min(CONFIG.get('food_limit', 12),
                   CONFIG.get('food_start_dist', 5.0) + CONFIG.get('food_growth', 0.15) * ep)


def signed_angle(d, t):
    cur = math.atan2(d.z, d.x)
    tgt = math.atan2(t.z, t.x)
    a = tgt - cur
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def turn_vec(d, ang):
    c, s = math.cos(ang), math.sin(ang)
    return Vec3(d.x * c - d.z * s, 0, d.x * s + d.z * c)


def teacher_dir(worm, env):
    foods = env.food_sources
    cx = cz = 0.0
    for fs in foods:
        to = Vec3(fs.position.x - worm.head.position.x, 0,
                  fs.position.z - worm.head.position.z)
        dist = (to.x ** 2 + to.z ** 2) ** 0.5
        if dist > 0.01:
            w = max(0.0, 1.0 - dist / CONFIG['sensor_max_dist'])
            cx += to.x / dist * w
            cz += to.z / dist * w
    if abs(cx) + abs(cz) > 0.01:
        return Vec3(cx, 0, cz).normalized()
    return worm.direction


def run(episodes, maze_key, seed, train=True, epsilon=0.0, force_autonomy0=False,
        eval_mode=False, weights_path=None, save_path=None):
    random.seed(seed)
    env = HeadlessMaze(maze_key)
    n_in = CONFIG.get('n_inputs_food', 11)
    brain = PolicyNetwork(n_in, CONFIG['n_hidden'], CONFIG['n_actions'],
                          temperature=CONFIG['temperature'],
                          n_hidden2=CONFIG.get('n_hidden2'))
    critic = CriticNetwork(n_in, CONFIG['n_hidden'],
                           n_hidden2=CONFIG.get('n_hidden2'))
    if weights_path:
        ok = load_brain(weights_path, brain, critic)
        print(f"  pesos {weights_path}: {'OK' if ok else 'IGNORADOS (dimensão?)'}")
    if eval_mode:
        brain.temperature = CONFIG['min_temperature']  # política decisiva
        train = False
    H = CONFIG['episode_steps']
    speed, turn_rate = CONFIG['speed'], CONFIG['turn_rate']
    dt = 1.0 / 60.0
    fr = CONFIG.get('food_radius', 3.0)
    results = []
    for ep in range(episodes):
        worm = HeadlessWorm(env.spawn[0], env.spawn[2])
        worm.direction = Vec3(0, 0, 1)
        if eval_mode:
            env.respawn_food()  # Fase 14: dificuldade total, sem currículo
        else:
            # T8: episódio começa com alimento a schedule(ep) do spawn (perto→longe)
            env.respawn_food(cx=env.spawn[0], cz=env.spawn[2],
                             max_dist=env.food_max_dist(ep))
        state = {'hunger': 0.0, 'prev_dist_food': None,
                 'prev_dist_obs': None, 'prev_position': None,
                 'hunger_rate': CONFIG.get('hunger_rate', 0.002)}
        traj = []
        encontros = 0
        primeiro = None
        ent_acc = 0.0
        act_count = [0] * 5
        # estágio currículo (mesma regra do main; teacher-only força A sempre)
        if eval_mode:
            stage, autonomy, lam = 'EVAL', 1.0, 0.0
        elif force_autonomy0 or ep < CONFIG['stage_a_episodes']:
            stage, autonomy, lam = 'A', 0.0, 0.0
        elif ep < CONFIG['stage_a_episodes'] + CONFIG['stage_b_episodes']:
            k = ep - CONFIG['stage_a_episodes']
            stage, autonomy = 'B', min(1.0, (k + 1) / CONFIG['stage_b_episodes'])
            lam = CONFIG['lambda_start'] * (CONFIG['lambda_decay'] ** k)
        else:
            stage, autonomy, lam = 'C', 1.0, float(CONFIG.get('lambda_c', 0.0) or 0.0)
        for step in range(H):
            sensors = get_food_sensor_inputs(worm, env, state)
            action = brain.sample_action(sensors)
            if epsilon > 0 and random.random() < epsilon:
                action = random.randrange(5)  # test-hook ε-greedy (T6)
            from mlp import entropy as _ent
            ent_acc += _ent(brain.last_probs)
            act_count[action] += 1
            # professor + mistura
            td = teacher_dir(worm, env)
            mt = turn_rate * dt
            tturn = max(-mt, min(mt, signed_angle(worm.direction, td)))
            turn = tturn * (1 - autonomy) + TURN_MULTS[action] * mt * autonomy
            worm.direction = turn_vec(worm.direction, turn).normalized()
            nx = worm.head.position.x + worm.direction.x * speed * dt
            nz = worm.head.position.z + worm.direction.z * speed * dt
            if env.free(nx, nz):
                worm.head.position.x, worm.head.position.z = nx, nz
            # recompensa + fome + comer
            r = calculate_food_reward(worm, env, state)
            d = min(((worm.head.position.x - f.position.x) ** 2
                     + (worm.head.position.z - f.position.z) ** 2) ** 0.5
                    for f in env.food_sources)
            if d < fr:
                encontros += 1
                if primeiro is None:
                    primeiro = step + 1
                state['hunger'] = 0.0
                if eval_mode:
                    env.respawn_food()
                else:
                    # T8: respawn mantém schedule do episódio (perto do worm)
                    env.respawn_food(cx=worm.head.position.x, cz=worm.head.position.z,
                                     max_dist=env.food_max_dist(ep))
            else:
                state['hunger'] = min(1.0, state['hunger'] + state['hunger_rate'])
            # ação professor p/ imitação
            td2 = teacher_dir(worm, env)
            des = signed_angle(worm.direction, td2)
            best = min(range(5), key=lambda a: abs(TURN_MULTS[a] * mt - des))
            traj.append((sensors, action, r, best))
        if train:
            brain.update_episode(traj, imitation_weight=lam, critic=critic,
                                 value_coef=CONFIG['value_coef'],
                                 gae_lambda=CONFIG['gae_lambda'],
                                 ppo_clip=CONFIG['ppo_clip'])
            brain.learning_rate *= CONFIG['lr_decay']
            brain.temperature = max(CONFIG['min_temperature'],
                                    brain.temperature * CONFIG['temperature_decay'])
        results.append((encontros, primeiro, ent_acc / H, list(act_count)))
        top = max(range(5), key=lambda a: act_count[a])
        print(f"ep {ep+1:3d} [{stage}] encontros={encontros} passos1o={primeiro} "
              f"ent={ent_acc/H:.3f} top={top}({act_count[top]}/{H})")
    # resumo pergunta-base
    enc = [e for e, _, _, _ in results]
    fst = [p for _, p, _, _ in results if p is not None]
    print("=" * 50)
    print(f"maze={maze_key} eps={episodes} seed={seed} train={train} eps_greedy={epsilon}")
    print(f"% com encontro: {100*sum(1 for e in enc if e>0)/len(enc):.1f}%  "
          f"media encontros/ep: {sum(enc)/len(enc):.2f}")
    # foco no defeito: estágio C (últimos 5) + entropia final
    c_enc = enc[-5:] if len(enc) >= 5 else enc
    c_ent = [en for _, _, en, _ in results[-5:]]
    print(f"ESTAGIO C: encontros={c_enc} ent_media={sum(c_ent)/len(c_ent):.3f}")
    if fst:
        print(f"passos ate 1o (mediana): {sorted(fst)[len(fst)//2]}  "
              f"min: {min(fst)}  media: {sum(fst)/len(fst):.1f}")
    else:
        print("passos ate 1o: NENHUM encontro (política aleatória ainda)")
    if save_path and train:
        save_brain(save_path, brain, critic)
        print(f"pesos salvos em {save_path}")
    return results


if __name__ == '__main__':
    eps, maze, seed = 10, 'A', CONFIG['seed']
    epsilon = 0.0  # Lean test-hook: 0 = desligado; >0 simula ε-greedy sem tocar mlp.py
    teacher_only = False  # T8: teto do professor (sem aprendizado, autonomy=0)
    eval_mode, weights_path, save_path = False, None, None
    for a in sys.argv[1:]:
        if a.startswith('--episodes='):
            eps = int(a.split('=')[1])
        elif a == '--eval':
            eval_mode = True
        elif a.startswith('--weights='):
            weights_path = a.split('=', 1)[1]
        elif a.startswith('--save='):
            save_path = a.split('=', 1)[1]
        elif a.startswith('--teacher-only'):
            teacher_only = True
        elif a.startswith('--maze='):
            maze = a.split('=')[1]
        elif a.startswith('--seed='):
            seed = int(a.split('=')[1])
        elif a.startswith('--epsilon='):
            epsilon = float(a.split('=')[1])
        elif a.startswith('--set='):
            key, value = a[len('--set='):].split('=', 1)
            try:
                value = float(value) if ('e' in value.lower() or '.' in value) else int(value)
            except ValueError:
                pass
            CONFIG[key] = value
            print(f"  CONFIG['{key}'] = {CONFIG[key]}")
    run(eps, maze, seed, train=not teacher_only and not eval_mode, epsilon=epsilon,
        force_autonomy0=teacher_only, eval_mode=eval_mode,
        weights_path=weights_path, save_path=save_path)
