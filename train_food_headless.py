"""
Trainer headless do modo alimento (T5, Lean: feedback rápido sem GUI).

Reusa o cérebro real (PolicyNetwork 8-dim + Critic + PPO) e a percepção real
(get_food_sensor_inputs / calculate_food_reward), mas com física simplificada
em grade (sem Ursina window). Serve para estimar a curva da pergunta-base
"épocas até o alimento no menor tempo" em segundos, antes de runs GUI caros.

Uso:
   python train_food_headless.py --episodes=10 --seed=42
   python train_food_headless.py --episodes=35 --seed=42 --eval --weights=pesos.json
"""
import math
import random
import sys

from ursina import Vec3

from config import CONFIG
from mlp import PolicyNetwork, CriticNetwork, QNetwork, save_brain, load_brain
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
    """Limpo default; com maze opt-in (maze_enabled=1) valida colisao."""
    def __init__(self):
        self.limit = CONFIG.get('map_limit', 32)
        self.food_sources = []
        self.spawn = [0, 1.25, -12]
        self.maze = None
        if CONFIG.get('maze_enabled', 0):
            try:
                from maze import Maze as _Maze
                self.maze = _Maze.load(CONFIG.get('maze_file', 'labirintos.json'),
                                       CONFIG.get('maze_key', 'gigante'))
                self.limit = self.maze.limit
            except Exception as e:
                print(f'  AVISO maze headless sem gigante ({e}) — limpo')
                self.maze = None
        self.respawn_food()

    def free(self, x, z, clearance=None):
        if abs(x) > self.limit or abs(z) > self.limit:
            return False
        if self.maze is not None:
            clr = clearance if clearance is not None else CONFIG.get('worm_radius', 1.5)
            return self.maze.free(x, z, clearance=clr)
        return True

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
            return CONFIG.get('food_limit', 24)
        return min(CONFIG.get('food_limit', 24),
                   CONFIG.get('food_start_dist', 10.0) + CONFIG.get('food_growth', 0.3) * ep)


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
    # Terreno limpo: igual ao GUI — atração comida + fuga de borda (sem muros).
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
    # ── Fuga de borda (mesma fórmula do main.py) ──
    # L16-refaz: limit vem do CONFIG como no GUI (paridadeTOTAL mapa 32).
    margin = CONFIG.get('wall_margin', 4.0)
    limit = CONFIG.get('map_limit', 32)
    px, pz = worm.head.position.x, worm.head.position.z
    if px < -limit + margin:
        cx += 1.0 * (1.0 + (-limit + margin - px) / margin)
    if px > limit - margin:
        cx += -1.0 * (1.0 + (px - (limit - margin)) / margin)
    if pz < -limit + margin:
        cz += 1.0 * (1.0 + (-limit + margin - pz) / margin)
    if pz > limit - margin:
        cz += -1.0 * (1.0 + (pz - (limit - margin)) / margin)
    # Terreno limpo: sem contorno de obstáculos.
    if abs(cx) + abs(cz) > 0.01:
        # L16-refaz: zona-morta em norma L2 como no GUI (antes L1 abs+abs;
        # só divergia na casca 0.005–0.01, desempate irrelevante).
        n = (cx ** 2 + cz ** 2) ** 0.5
        if n > 0.01:
            return Vec3(cx / n, 0, cz / n)
    return worm.direction


def run(episodes, seed, train=True, epsilon=0.0, force_autonomy0=False,
        eval_mode=False, weights_path=None, save_path=None,
        bc_mode=False, dqn_mode=False, dqn_epsilon=0.2, target_sync=10,
        csv_path=None, target_sync_steps=0):
    random.seed(seed)
    # L15: --bc e --dqn são mutuamente exclusivos (supervisionado vs TD).
    # Se ambos forem passados, o DQN vence e o BC é ignorado com aviso.
    if bc_mode and dqn_mode:
        print("  AVISO L15: --bc + --dqn juntos → DQN vence, BC ignorado no treino")
        bc_mode = False
    env = HeadlessMaze()
    n_in = CONFIG.get('n_inputs_food', 8)
    if dqn_mode:
        # L14b: value-based — Q online + Q alvo (sync a cada N eps)
        brain = QNetwork(n_in, CONFIG['n_hidden'], CONFIG['n_actions'],
                         n_hidden2=CONFIG.get('n_hidden2'))
        target = QNetwork(n_in, CONFIG['n_hidden'], CONFIG['n_actions'],
                          n_hidden2=CONFIG.get('n_hidden2'))
        target.sync_from(brain)
        critic = None
    else:
        brain = PolicyNetwork(n_in, CONFIG['n_hidden'], CONFIG['n_actions'],
                              temperature=CONFIG['temperature'],
                              n_hidden2=CONFIG.get('n_hidden2'))
        critic = CriticNetwork(n_in, CONFIG['n_hidden'],
                               n_hidden2=CONFIG.get('n_hidden2'))
        target = None
    if weights_path:
        ok = load_brain(weights_path, brain, critic)
        print(f"  pesos {weights_path}: {'OK' if ok else 'IGNORADOS (dimensão?)'}")
    if eval_mode:
        # L15: temperatura só existe na Policy (softmax). Q age por argmax +
        # epsilon externo, então não há temperatura a ajustar no DQN.
        if not dqn_mode:
            brain.temperature = CONFIG['min_temperature']  # política decisiva
        train = False
    H = CONFIG['episode_steps']
    speed, turn_rate = CONFIG['speed'], CONFIG['turn_rate']
    dt = 1.0 / 60.0
    fr = CONFIG.get('food_radius', 3.0)
    results = []
    # CSV com as mesmas 13 colunas do GUI (terreno limpo, sem maze).
    # DQN não tem entropia softmax: registra epsilon como proxy (documentado).
    csv_f, csv_w = None, None
    if csv_path:
        import csv as _csv
        csv_f = open(csv_path, 'w', newline='')
        csv_w = _csv.writer(csv_f)
        csv_w.writerow(['episodio', 'estagio', 'recompensa_total', 'recompensa_media',
                        'retorno_medio', 'entropia_media', 'learning_rate',
                        'lambda_imitacao', 'autonomia_forcada',
                        'encontros_food', 'passos_ate_comer',
                        'acao_principal', 'semente'])
    for ep in range(episodes):
        worm = HeadlessWorm(env.spawn[0], env.spawn[2])
        # Movimento livre: heading aleatorio por episodio (fix "sempre sobe").
        # Legado norte: --set=spawn_heading=fixed.
        if CONFIG.get('spawn_heading', 'random') == 'random':
            _a0 = random.uniform(0, 2 * math.pi)
            worm.direction = Vec3(math.cos(_a0), 0, math.sin(_a0))
        else:
            worm.direction = Vec3(0, 0, 1)
        if eval_mode:
            env.respawn_food()  # Fase 14: dificuldade total, sem currículo
        else:
            # T8: episódio começa com alimento a schedule(ep) do spawn (perto→longe)
            env.respawn_food(cx=env.spawn[0], cz=env.spawn[2],
                             max_dist=env.food_max_dist(ep))
        state = {'hunger': 0.0, 'prev_dist_food': None,
                 'prev_position': None,
                  'hunger_rate': CONFIG.get('hunger_rate', 0.001)}
        traj = []
        encontros = 0
        primeiro = None
        ent_acc = 0.0
        act_count = [0] * 5
        # estágio currículo (mesma regra do main; teacher-only força A sempre)
        if eval_mode:
            stage, autonomy, lam = 'EVAL', 1.0, 0.0
        elif bc_mode and train:
            # L14a: BC puro — segue o professor p/ coletar pares corretos
            stage, autonomy, lam = 'BC', 0.0, 0.0
        elif force_autonomy0 or ep < CONFIG['stage_a_episodes']:
            stage, autonomy, lam = 'A', 0.0, 0.0
        elif ep < CONFIG['stage_a_episodes'] + CONFIG['stage_b_episodes']:
            k = ep - CONFIG['stage_a_episodes']
            stage, autonomy = 'B', min(1.0, (k + 1) / CONFIG['stage_b_episodes'])
            lam = CONFIG['lambda_start'] * (CONFIG['lambda_decay'] ** k)
        else:
            stage, autonomy, lam = 'C', 1.0, float(CONFIG.get('lambda_c', 0.0) or 0.0)
        if dqn_mode and train:
            # L14b: sem professor no movimento (autonomia total, explora via epsilon)
            autonomy = 1.0
            stage = 'DQN'
        dqn_trans = []  # (s, a, r_scaled, s_next, done)
        total_r = 0.0  # soma p/ CSV (recompensa_total)
        for step in range(H):
            sensors = get_food_sensor_inputs(worm, env, state)
            if dqn_mode:
                eps_now = dqn_epsilon if train else 0.0
                action = brain.act(sensors, epsilon=eps_now)
                ent_acc += eps_now  # proxy: taxa de exploracao (nao entropia softmax)
            else:
                action = brain.sample_action(sensors)
                # L15: epsilon só no treino — no eval a política deve ser decisiva,
                # senão o eval mede sorte, não aprendizado.
                if epsilon > 0 and train and random.random() < epsilon:
                    action = random.randrange(5)
                from mlp import entropy as _ent
                ent_acc += _ent(brain.last_probs)
            # NOTA L15: act_count movido p/ depois do override BC (ação executada,
            # não amostrada) — antes mentia no modo --bc.
            # professor + mistura
            td = teacher_dir(worm, env)
            mt = turn_rate * dt
            tturn = max(-mt, min(mt, signed_angle(worm.direction, td)))
            turn = tturn * (1 - autonomy) + TURN_MULTS[action] * mt * autonomy
            worm.direction = turn_vec(worm.direction, turn).normalized()
            nx = worm.head.position.x + worm.direction.x * speed * dt
            nz = worm.head.position.z + worm.direction.z * speed * dt
            # Colisao deslizante (igual worm.step): X depois Z — permite
            # contornar e descer por corredores em qualquer direcao.
            _clr = CONFIG.get('worm_radius', 1.5) if env.maze is not None else None
            _cx, _cz = worm.head.position.x, worm.head.position.z
            if _clr is not None:
                if env.free(nx, _cz, clearance=_clr):
                    _cx = nx
                if env.free(_cx, nz, clearance=_clr):
                    _cz = nz
                worm.head.position.x, worm.head.position.z = _cx, _cz
            elif env.free(nx, nz):
                worm.head.position.x, worm.head.position.z = nx, nz
            # recompensa + fome + comer
            # L15: s_next ANTES do respawn — a transição TD (s,a,r,s') deve ser do
            # MESMO objetivo. Se ler s' depois de teleportar a comida, o agente
            # aprende a mentira "depois de comer a comida já está longe".
            r = calculate_food_reward(worm, env, state, action=action)
            d = min(((worm.head.position.x - f.position.x) ** 2
                     + (worm.head.position.z - f.position.z) ** 2) ** 0.5
                    for f in env.food_sources)
            ate = (d < fr)
            s_next_pre = None
            if dqn_mode:
                s_next_pre = get_food_sensor_inputs(worm, env, state)
            if ate:
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
            if bc_mode and train:
                action = best  # executa o professor; par (s, best) é o alvo
            # L15: conta a ação EXECUTADA (pós-override), não a amostrada.
            act_count[action] += 1
            total_r += r  # acumula p/ CSV
            traj.append((sensors, action, r, best))
            if dqn_mode:
                # L15: done=True ao comer (subgoal terminou: target=r, sem
                # bootstrap p/ a comida NOVA). Timeout (step==H-1 sem comer) é
                # truncamento, não morte: done=False (mantém bootstrap).
                done = bool(ate)
                dqn_trans.append((list(sensors), action, r, list(s_next_pre), done))
        if train:
            if bc_mode:
                # L14a: só cross-entropy supervisionada (sem PPO/A2C)
                for s_bc, a_bc in ((s, t) for s, _, _, t in traj):
                    brain.imitate(s_bc, a_bc)
                brain.learning_rate *= CONFIG['lr_decay']
            elif dqn_mode:
                # L14b: TD batch do episodio + sync periodica do alvo.
                # L16: reward_scale default 1.0 p/ Q (não 5.0 do PPO) — Q em
                # escala crua [-1,1] é estável; 5x explodia o target e o
                # clipping [-10,10] mentia. Override via --set=reward_scale=X.
                _rs = CONFIG.get('q_reward_scale', 1.0)
                brain.td_update(dqn_trans, target_net=target,
                                gamma=CONFIG['gamma'],
                                learning_rate=brain.learning_rate,
                                regularization=CONFIG['regularization'],
                                reward_scale=_rs)
                brain.learning_rate *= CONFIG['lr_decay']
                # L16: sync por steps (opcional) ou por episódio (legado).
                # P/ iniciante: rede-alvo fixa = gabarito congelado; sync copia.
                # Restrição: S deve ser múltiplo de H (ex: H=200, S=2000); se
                # S<H, sincroniza todo episódio (nunca intra-ep); se não for
                # múltiplo, há deriva de fase — avisado abaixo, não quebrado.
                if target_sync_steps > 0:
                    if target_sync_steps < H or (target_sync_steps % H) != 0:
                        print(f"  AVISO L16: --target-sync-steps={target_sync_steps} não-múltiplo de H={H} — sync só em fronteira de episódio")
                    if (ep * H + H) % max(1, target_sync_steps) < H:
                        target.sync_from(brain)
                elif (ep + 1) % max(1, target_sync) == 0:
                    target.sync_from(brain)
            else:
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
        # mesma linha do GUI — retorno_medio = media de r (BC/DQN sem GAE; documentado).
        if csv_w:
            from mlp import ACTIONS as _ACTS
            mean_r = total_r / max(1, H)
            auto = 1 if (eval_mode or stage in ('C', 'EVAL', 'DQN')) else 0
            csv_w.writerow([
                ep + 1, stage, round(total_r, 3), round(mean_r, 4),
                round(mean_r, 3), round(ent_acc / H, 3),
                round(brain.learning_rate, 5),
                round(lam, 3) if lam > 0 else 0.0, auto,
                encontros, primeiro if primeiro is not None else '',
                _ACTS[top], seed,
            ])
    # resumo pergunta-base
    enc = [e for e, _, _, _ in results]
    fst = [p for _, p, _, _ in results if p is not None]
    print("=" * 50)
    print(f"eps={episodes} seed={seed} train={train} eps_greedy={epsilon} bc={bc_mode} dqn={dqn_mode}")
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
    if csv_f:
        csv_f.close()
        print(f"CSV salvo em {csv_path} (13 cols = GUI)")
    return results


if __name__ == '__main__':
    eps, seed = 10, CONFIG['seed']
    epsilon = 0.0  # Lean test-hook: 0 = desligado; >0 simula ε-greedy sem tocar mlp.py
    teacher_only = False  # T8: teto do professor (sem aprendizado, autonomy=0)
    eval_mode, weights_path, save_path = False, None, None
    bc_mode = False  # L14a: behavioral cloning puro (só imitate, autonomy=0)
    dqn_mode = False  # L14b: DQN value-based (Q + epsilon-greedy + alvo)
    dqn_epsilon = 0.2
    target_sync = 10
    target_sync_steps = 0  # L16: >0 = sync por steps (ex: 2000); 0 = por episódio
    csv_path = None  # L16: --csv=episodios_headless.csv (13 cols = GUI)
    for a in sys.argv[1:]:
        if a.startswith('--episodes='):
            eps = int(a.split('=')[1])
        elif a == '--eval':
            eval_mode = True
        elif a.startswith('--weights='):
            weights_path = a.split('=', 1)[1]
        elif a.startswith('--save='):
            save_path = a.split('=', 1)[1]
        elif a == '--teacher-only':
            teacher_only = True
        elif a.startswith('--seed='):
            seed = int(a.split('=')[1])
        elif a.startswith('--epsilon='):
            epsilon = float(a.split('=', 1)[1])
        elif a == '--bc':
            bc_mode = True
        elif a == '--dqn':
            dqn_mode = True
        elif a.startswith('--dqn-epsilon='):
            dqn_epsilon = float(a.split('=', 1)[1])
        elif a.startswith('--target-sync='):
            target_sync = int(a.split('=', 1)[1])
        elif a.startswith('--target-sync-steps='):
            target_sync_steps = int(a.split('=', 1)[1])
        elif a.startswith('--csv='):
            csv_path = a.split('=', 1)[1]
        elif a.startswith('--set='):
            key, value = a[len('--set='):].split('=', 1)
            try:
                value = float(value) if ('e' in value.lower() or '.' in value) else int(value)
            except ValueError:
                pass
            CONFIG[key] = value
            print(f"  CONFIG['{key}'] = {CONFIG[key]}")
    run(eps, seed, train=not teacher_only and not eval_mode, epsilon=epsilon,
        force_autonomy0=teacher_only, eval_mode=eval_mode,
        weights_path=weights_path, save_path=save_path,
        bc_mode=bc_mode, dqn_mode=dqn_mode, dqn_epsilon=dqn_epsilon,
        target_sync=target_sync, csv_path=csv_path,
        target_sync_steps=target_sync_steps)
