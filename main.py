import csv
import math
import os
import random
import sys

from ursina import *

from config import CONFIG
from environment import Environment
from mlp import ACTIONS, PolicyNetwork, CriticNetwork, entropy, save_brain, load_brain
from perception import calculate_reward, get_sensor_inputs
from worm import Worm

app = Ursina()

# ─── MODO DE EXECUÇÃO (Fase 5) ────────────────────────────────────────────────
# python main.py --eval [--episodes=N] [--set chave=valor ...]
#   --eval          → avaliação: professor sempre desligado, sem treino.
#   --episodes=N    → (com --eval) roda N episódios, salva pesos e fecha.
#   --set k=v       → sobrescreve um valor do CONFIG (tabela de experimentos).
EVAL_MODE = '--eval' in sys.argv
EVAL_EPISODES = None
NUM_SEEDS = 1
for _arg in sys.argv:
    if _arg.startswith('--episodes='):
        EVAL_EPISODES = int(_arg.split('=')[1])
    elif _arg.startswith('--seeds='):
        NUM_SEEDS = int(_arg.split('=')[1])
    elif _arg.startswith('--set='):
        key, value = _arg[len('--set='):].split('=', 1)
        try:
            value = float(value) if ('e' in value.lower() or '.' in value) else int(value)
        except ValueError:
            pass
        CONFIG[key] = value
        print(f"  CONFIG['{key}'] = {CONFIG[key]}")

random.seed(CONFIG['seed'])  # execução reproduzível

# ─── ILUMINAÇÃO ───────────────────────────────────────────────────────────────
DirectionalLight(y=2, z=-1)
AmbientLight(color=Color(0.6, 0.6, 0.6, 1))  # Iluminação mais clara e vibrante

# ─── CHÃO E CÉU ───────────────────────────────────────────────────────────────
Entity(
    model='plane',
    scale=40,
    texture='grass',  # Textura mais detalhada
    texture_scale=(40, 40),
    color=color.white,  # Cor mais clara para destacar o verme
)

sky = Entity(
    model='sphere',
    scale=500,
    double_sided=True,
    texture='sky_sunset',  # Textura de céu
    color=color.white.tint(-0.2),
)

# ─── PLANO INVISÍVEL DE POSICIONAMENTO ────────────────────────────────────────
ground_collider = Entity(
    model='plane',
    scale=40,
    collider='box',
    visible=False,
    y=0,
)

# ─── AMBIENTE (FONTES DE LUZ E CHUVA) ─────────────────────────────────────────
env = Environment()
env.place_light(Vec3(10, 1, 10))
env.place_rain(Vec3(-10, 1, -10))

# ─── VERME ────────────────────────────────────────────────────────────────────
worm = Worm()

# Luz suave que segue o verme
worm_light = PointLight(
    parent=worm.head,
    color=color.cyan,  # Luz azul brilhante
    position=(0, 5, 0),
    intensity=1.5,
)

# ─── CÂMERA ───────────────────────────────────────────────────────────────────
cam_pivot = Entity()
cam_pivot.y = 10

camera.parent = cam_pivot
camera.position = (0, 20, -50)
camera.rotation = (20, 0, 0)

# ─── CÉREBRO: política estocástica (8 → 16 → 5 ações) ─────────────────────────
brain = PolicyNetwork(
    CONFIG['n_inputs'],
    CONFIG['n_hidden'],
    CONFIG['n_actions'],
    temperature=CONFIG['temperature'],
)

# ─── CRÍTICO (Fase 8 — A2C): V(s) 8 → 16 → 1 ──────────────────────────────
critic = CriticNetwork(
    CONFIG['n_inputs'],
    CONFIG['n_hidden'],
)

# ─── MODO AVALIAÇÃO: testa a política SALVA (pesos.json) ───────────────────
# A avaliação carrega o cérebro treinado e usa a temperatura de decisão (piso
# do treino): política determinística, representativa do que foi aprendido.
if EVAL_MODE:
    if os.path.exists(CONFIG['weights_file']):
        load_brain(CONFIG['weights_file'], brain, critic)
        print(f"  Avaliando pesos salvos em {CONFIG['weights_file']}")
    else:
        print(f"  ATENCAO: {CONFIG['weights_file']} nao existe — avaliando cerebro aleatorio")
    brain.temperature = CONFIG['min_temperature']

# ─── MODO DE EDIÇÃO ───────────────────────────────────────────────────────────
editor = {'mode': 'none'}

# ─── ESTADO DO VERME ──────────────────────────────────────────────────────────
state = {
    'total_reward'    : 0.0,
    'log_timer'       : 0.0,
    'rain_pulse'      : 0.0,
    'pulse_timer'     : 0.0,
    'prev_dist_light' : None,
    'prev_dist_rain'  : None,
    'prev_position'   : None,
    'episode'         : [],     # (sensors, action, reward, acao_professor) — Fases 3/4
    'episode_count'   : 0,      # quantos episódios já treinaram
    'force_autonomy'  : False,  # tecla A: professor desligado (autonomia forçada = 1)
    'steps_in_rain'   : 0,      # métricas do episódio (critério de aceite da Fase 4)
    'steps_in_danger' : 0,
    'episode_rewards' : [],     # recompensas totais por episódio (rolling no HUD)
    'action_history'  : [],     # Fase 7: últimas N ações (anti-colapso)
}

if EVAL_MODE:
    state['force_autonomy'] = True

# ─── LOG DE EPISÓDIOS (curva de aprendizado, critério de aceite da Fase 3/4) ─
log_csv = open(CONFIG['log_csv'], 'w', newline='')
csv_writer = csv.writer(log_csv)
csv_writer.writerow(['episodio', 'estagio', 'recompensa_total', 'recompensa_media',
                     'retorno_medio', 'entropia_media', 'learning_rate',
                     'lambda_imitacao', 'autonomia_forcada',
                     'chegada_chuva', 'perigo_luz', 'acao_principal', 'semente'])

# ─── HUD (Fase 5) ─────────────────────────────────────────────────────────────
hud = Text(
    text='',
    position=(-0.85, 0.47),
    origin=(0, 0),
    scale=1,
    color=color.white,
    background_color=Color(0, 0, 0, 0.55),
)


# ─── MOVIMENTO POR GIRO (ações discretas) ─────────────────────────────────────
# Cada ação vira o verme no plano XZ por um múltiplo da taxa máxima:
#   esquerda(−1)  frente_esquerda(−0.5)  frente(0)  frente_direita(+0.5)  direita(+1)
TURN_MULTIPLIERS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def signed_angle(direction, target):
    """Ângulo orientado (em radianos) de `direction` até `target`, em [-pi, pi]."""
    cur = math.atan2(direction.z, direction.x)
    tgt = math.atan2(target.z, target.x)
    d = tgt - cur
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def turn_vector(direction, angle):
    """Rotaciona um vetor no plano XZ por `angle` radianos."""
    c = math.cos(angle)
    s = math.sin(angle)
    return Vec3(
        direction.x * c - direction.z * s,
        0,
        direction.x * s + direction.z * c,
    )


# ─── PROFESSOR (CAMPOS DE POTENCIAL) ──────────────────────────────────────────
def get_target_direction():
    """
    Direção ideal de movimento ("professor"), por campos de potencial.

    Contrato da Fase 1: o comportamento desejado é CHEGAR à chuva e FUGIR da
    luz. A "órbita" ao redor da chuva foi removida — a meta é chegar e
    permanecer perto, não circular.
    """
    combined = Vec3(0, 0, 0)

    # ── Atração pelas fontes de chuva ─────────────────────────────────────────
    for rs in env.rain_sources:
        to_rain = rs.position - worm.head.position
        dist    = to_rain.length()
        if dist > 0.01:
            # Peso inversamente proporcional à distância: fontes próximas
            # atraem mais do que fontes distantes
            weight    = max(0.0, 1.0 - (dist / CONFIG['sensor_max_dist']))
            combined += to_rain.normalized() * weight

    # ── Repulsão pelas fontes de luz ──────────────────────────────────────────
    for ls in env.light_sources:
        from_light = worm.head.position - ls.position
        dist       = from_light.length()
        if dist > 0.01:
            weight    = max(0.0, 1.0 - (dist / CONFIG['light_repel_dist']))
            combined += from_light.normalized() * weight

    # ── Fuga das bordas (anti-encalhe, Fase 6) ────────────────────────────────
    # Se o verme estiver perto de uma parede, o professor empurra de volta para
    # o centro — evita ficar "engessado" contra o limite do mapa.
    margin = CONFIG['wall_margin']
    limit  = CONFIG['map_limit']
    p = worm.head.position
    if p.x < -limit + margin:
        combined += Vec3(1, 0, 0) * (1.0 + (-limit + margin - p.x) / margin)
    if p.x > limit - margin:
        combined += Vec3(-1, 0, 0) * (1.0 + (p.x - (limit - margin)) / margin)
    if p.z < -limit + margin:
        combined += Vec3(0, 0, 1) * (1.0 + (-limit + margin - p.z) / margin)
    if p.z > limit - margin:
        combined += Vec3(0, 0, -1) * (1.0 + (p.z - (limit - margin)) / margin)

    if Vec3(combined).length() > 0.01:
        return Vec3(combined).normalized()

    return worm.direction


# ─── CURRÍCULO DE AUTONOMIA (Fase 4) ─────────────────────────────────────────
# Estágio A — Imitação (warm-up): professor demonstra, rede aprende por CE.
# Estágio B — Híbrido: REINFORCE + λ·CE (λ decai), autonomia sobe 0→1.
# Estágio C — Autonomia plena: λ = 0, professor desligado (tecla A confirma).
STAGE_NAMES = {'A': 'IMITACAO', 'B': 'HIBRIDO', 'C': 'AUTONOMO'}


def current_stage():
    """Estágio do currículo com base no número de episódios já treinados."""
    n = state['episode_count']
    if n < CONFIG['stage_a_episodes']:
        return 'A'
    if n < CONFIG['stage_a_episodes'] + CONFIG['stage_b_episodes']:
        return 'B'
    return 'C'


def lambda_imitation():
    """Peso λ da imitação no híbrido (Estágio B), decaindo até 0 no Estágio C."""
    stage = current_stage()
    if stage == 'B':
        k = state['episode_count'] - CONFIG['stage_a_episodes']
        return CONFIG['lambda_start'] * (CONFIG['lambda_decay'] ** k)
    return 0.0


def teacher_action():
    """A ação discreta ideal do professor — alvo da imitação (CE)."""
    target_dir = get_target_direction()
    desired = signed_angle(worm.direction, target_dir)
    max_turn = CONFIG['turn_rate'] * time.dt
    best, best_err = 0, float('inf')
    for a, m in enumerate(TURN_MULTIPLIERS):
        err = abs(m * max_turn - desired)
        if err < best_err:
            best_err, best = err, a
    return best


def _nearest_dist(position, sources):
    """Distância da posição até a fonte mais próxima (ou None)."""
    best = None
    for s in sources:
        d = (position - s.position).length()
        if best is None or d < best:
            best = d
    return best


def episode_stats(brain, episode):
    """Estatísticas de um episódio para o log (usado no Estágio A / imitação)."""
    counts = [0] * CONFIG['n_actions']
    total_r = 0.0
    ent = 0.0
    for sensors, action, reward, _ in episode:
        counts[action] += 1
        total_r += reward
        ent += entropy(brain.probabilities(sensors))
    n = max(1, len(episode))
    return {
        'mean_reward'   : total_r / n,
        'mean_return'   : 0.0,
        'mean_entropy'  : ent / n,
        'action_counts' : counts,
    }


# ─── VISUALIZAÇÃO DA POLÍTICA (Fase 5, tecla P) ───────────────────────────────
# Grade de setas sobre o mapa: em cada célula, mostra para onde a política
# "quer ir" (ação mais provável, assumindo o verme virado para +Z) e pinta de
# verde/vermelho conforme a direção aproxime da chuva ou da luz.
policy_grid = {'visible': False, 'pivots': []}


def _fake_worm_at(x, z):
    class _Head:
        position = Vec3(x, 0, z)
    class _Worm:
        head = _Head()
    return _Worm()


def build_policy_grid():
    """Cria os pivôs (seta + cabo) da grade, uma vez."""
    n = CONFIG['grid_cells']
    step = 32 / (n - 1)
    for gx in range(n):
        for gz in range(n):
            x, z = -16 + gx * step, -16 + gz * step
            pivot = Entity(position=(x, 0.4, z))
            Entity(model='cube', scale=(0.12, 0.12, 1.0), color=color.gray,
                   parent=pivot, y=0)
            Entity(model='cone', scale=(0.3, 0.3, 0.4), position=(0, 0, 1.0),
                   rotation_x=90, parent=pivot)
            pivot.visible = False
            policy_grid['pivots'].append(pivot)


def refresh_policy_grid():
    """Recomputa a seta e a cor de cada célula com a política atual."""
    n = CONFIG['grid_cells']
    step = 32 / (n - 1)
    for gx in range(n):
        for gz in range(n):
            pivot = policy_grid['pivots'][gx * n + gz]
            x, z = -16 + gx * step, -16 + gz * step

            sensors = get_sensor_inputs(_fake_worm_at(x, z), env, state)
            probs = brain.probabilities(sensors)
            action = max(range(len(probs)), key=lambda a: probs[a])

            # Direção resultante da ação mais provável (virado para +Z)
            max_turn = CONFIG['turn_rate'] * 0.016
            newdir = turn_vector(Vec3(0, 0, 1), TURN_MULTIPLIERS[action] * max_turn)
            pivot.rotation_y = math.degrees(math.atan2(newdir.x, newdir.z))

            # Cor: verde se aponta p/ chuva, vermelho se p/ luz, cinza se neutro
            to_rain = _nearest_dist(pivot.position, env.rain_sources)
            to_light = _nearest_dist(pivot.position, env.light_sources)
            color = color.gray
            if to_rain is not None:
                d = Vec3(pivot.position.x, 0, pivot.position.z)
                r = min(env.rain_sources, key=lambda s: (Vec3(s.x, 0, s.z) - d).length())
                toward_rain = (Vec3(r.x, 0, r.z) - d).normalized()
                if newdir.dot(toward_rain) > 0.25:
                    color = color.lime
            if to_light is not None:
                d = Vec3(pivot.position.x, 0, pivot.position.z)
                l = min(env.light_sources, key=lambda s: (Vec3(s.x, 0, s.z) - d).length())
                toward_light = (Vec3(l.x, 0, l.z) - d).normalized()
                if newdir.dot(toward_light) > 0.25:
                    color = color.red.tint(-0.2)
            for child in pivot.children:
                child.color = color


def toggle_policy_grid():
    """Mostra/esconde a grade de setas (recalculada ao ligar)."""
    if not policy_grid['pivots']:
        build_policy_grid()
    policy_grid['visible'] = not policy_grid['visible']
    if policy_grid['visible']:
        refresh_policy_grid()
    for pivot in policy_grid['pivots']:
        pivot.visible = policy_grid['visible']
    print(f"  Grade da politica {'LIGADA' if policy_grid['visible'] else 'DESLIGADA'}")


# ─── HUD (Fase 5) ─────────────────────────────────────────────────────────────
def update_hud():
    """Atualiza o texto na tela com as métricas do aprendizado."""
    window = CONFIG['rolling_window']
    rolling = state['episode_rewards'][-window:]
    media = sum(rolling) / len(rolling) if rolling else 0.0
    stage = current_stage()
    ent = entropy(brain.last_probs) if brain.last_probs else 0.0
    hud.text = (
        f"episodio : {state['episode_count']}\n"
        f"estagio  : {stage} ({STAGE_NAMES[stage]})\n"
        f"recompensa media (roll {len(rolling)}/{window}): {media:+.2f}\n"
        f"entropia : {ent:.3f}\n"
        f"learning rate: {brain.learning_rate:.5f}\n"
        f"temperatura  : {brain.temperature:.3f}\n"
        f"professor: {'desligado' if state['force_autonomy'] else 'ligado'}\n"
        f"modo     : {'AVALIACAO' if EVAL_MODE else 'TREINO'}"
    )


# ─── REINICIAR ────────────────────────────────────────────────────────────────
def reset():
    """Reinicia o verme e o cérebro sem fechar o programa."""
    worm.reset()
    state.update({
        'total_reward'    : 0.0,
        'log_timer'       : 0.0,
        'rain_pulse'      : 0.0,
        'pulse_timer'     : 0.0,
        'prev_dist_light' : None,
        'prev_dist_rain'  : None,
        'prev_position'   : None,
        'episode'         : [],
        'episode_count'   : 0,
        'steps_in_rain'   : 0,
        'steps_in_danger' : 0,
        'episode_rewards' : [],
    })
    brain.reset()
    print("\n-- Reiniciado -----------------------------------------\n")


# ─── FIM DE EPISÓDIO (treino por estágio do currículo — Fase 4) ──────────────
def finish_episode():
    """
    Treina a política conforme o estágio do currículo e monta uma cena nova.

    Estágio A — Imitação: cross-entropy supervisionada com a ação do professor.
    Estágio B — Híbrido:  REINFORCE + λ·CE (λ decai), autonomia sobe 0→1.
    Estágio C — Autônomo: REINFORCE puro (λ = 0), professor desligado.
    """
    stage = current_stage()
    lam = lambda_imitation()

    if EVAL_MODE:
        # Avaliação (--eval): professor desligado e SEM treino — só mede.
        stats = episode_stats(brain, state['episode'])
    elif stage == 'A':
        # Imitação pura: um passo de CE por passo do episódio (ação do professor)
        for sensors, _, _, t_action in state['episode']:
            brain.imitate(sensors, t_action)
        stats = episode_stats(brain, state['episode'])
    else:
        # B: REINFORCE + λ·CE ; C: REINFORCE puro (λ = 0)
        # Fase 9: PPO — GAE advantage + clipped surrogate + critic
        stats = brain.update_episode(state['episode'], imitation_weight=lam,
                                     critic=critic, value_coef=CONFIG['value_coef'],
                                     gae_lambda=CONFIG['gae_lambda'],
                                     ppo_clip=CONFIG['ppo_clip'])

    if not EVAL_MODE:
        brain.learning_rate *= CONFIG['lr_decay']
        # Exploração estruturada (boltzmann): temperatura decai a cada episódio,
        # mas nunca zera — a política nunca fica 100% greedy (anti-colapso).
        brain.temperature = max(
            CONFIG['min_temperature'],
            brain.temperature * CONFIG['temperature_decay'],
        )

    state['episode_count'] += 1
    total = sum(r for _, _, r, _ in state['episode'])
    state['episode_rewards'].append(total)
    h = max(1, len(state['episode']))
    principal = max(range(len(stats['action_counts'])),
                    key=lambda a: stats['action_counts'][a])
    csv_writer.writerow([
        state['episode_count'],
        stage,
        round(total, 3),
        round(stats['mean_reward'], 4),
        round(stats['mean_return'], 3),
        round(stats['mean_entropy'], 3),
        round(brain.learning_rate, 5),
        round(lam, 3) if lam > 0 else 0.0,
        int(state['force_autonomy']),
        round(state['steps_in_rain'] / h, 3),
        round(state['steps_in_danger'] / h, 3),
        ACTIONS[principal],
        CONFIG['seed'],
    ])
    log_csv.flush()

    actions = ' '.join(f"{ACTIONS[a][:6]}:{c}" for a, c in enumerate(stats['action_counts']))
    print(
        f"\n-- Episodio {state['episode_count']} [estagio {stage}: {STAGE_NAMES[stage]}] -----\n"
        f"  total={total:+.1f}  media={stats['mean_reward']:+.4f}  "
        f"entropia={stats['mean_entropy']:.3f}  lr={brain.learning_rate:.5f}\n"
        f"  lambda_imit={lam:.3f}  chegada_chuva={state['steps_in_rain']/h:.2f}  "
        f"perigo_luz={state['steps_in_danger']/h:.2f}\n"
        f"  acoes: {actions}\n"
    )

    # Limite de episódios (treino com --episodes=N ou avaliação): salva e fecha
    if EVAL_EPISODES and state['episode_count'] >= EVAL_EPISODES:
        save_brain(CONFIG['weights_file'], brain, critic)
        print(f"\n-- Concluido ({EVAL_EPISODES} episodios): pesos salvos em {CONFIG['weights_file']}")
        quit()

    env.randomize_sources()
    worm.reset()
    worm.head.position = Vec3(random.uniform(-8, 8), CONFIG['segment_size'] / 2, random.uniform(-8, 8))
    state['episode'] = []
    state['steps_in_rain'] = 0
    state['steps_in_danger'] = 0
    state['action_history'] = []
    state['prev_position'] = Vec3(worm.head.position)


# ─── UPDATE (loop único por frame) ────────────────────────────────────────────
def update():
    # ── Câmera segue o verme ──────────────────────────────────────────────────
    cam_pivot.position = lerp(cam_pivot.position, worm.head.position, 0.1)
    worm_light.position = worm.head.position + Vec3(0, 5, 0)
    worm.animate()

    # ── Câmera: órbita ────────────────────────────────────────────────────────
    if mouse.right:
        cam_pivot.rotation_y += mouse.velocity[0] * CONFIG['cam']['rot_speed']
        cam_pivot.rotation_x -= mouse.velocity[1] * CONFIG['cam']['rot_speed']
        cam_pivot.rotation_x  = clamp(cam_pivot.rotation_x, -80, 80)

    # ── Câmera: pan ───────────────────────────────────────────────────────────
    if held_keys['w']: cam_pivot.y += CONFIG['cam']['pan_speed'] * time.dt
    if held_keys['s']: cam_pivot.y -= CONFIG['cam']['pan_speed'] * time.dt
    if held_keys['a']: cam_pivot.x -= CONFIG['cam']['pan_speed'] * time.dt
    if held_keys['d']: cam_pivot.x += CONFIG['cam']['pan_speed'] * time.dt

    # ── Pulso de chuva ────────────────────────────────────────────────────────
    state['pulse_timer'] += time.dt
    if state['pulse_timer'] > 0.3:
        state['pulse_timer'] = 0.0
        state['rain_pulse']  = 1.0
    else:
        state['rain_pulse'] *= (1 - time.dt * 8)

    # ── Partículas de chuva ───────────────────────────────────────────────────
    env.update_rain_particles()

    # ── Estado (8-dim) e amostragem de ação ───────────────────────────────────
    sensors = get_sensor_inputs(worm, env, state)
    action = brain.sample_action(sensors)

# ── Direção: currículo professor + política (curriculum learning) ────────
    #
    # 'autonomy' controla quanto do movimento vem da rede neural vs do professor
    # determinístico. Na Fase 4 ela é ditada pelo ESTÁGIO do currículo:
    #   A → 0 (professor demonstra) ; B → sobe 0→1 ; C → 1 (autonomia plena).
    # A tecla A força autonomia total (professor desligado) para avaliar o verme.
    stage = current_stage()
    if stage == 'A':
        autonomy = 0.0
    elif stage == 'B':
        k = state['episode_count'] - CONFIG['stage_a_episodes']
        autonomy = min(1.0, (k + 1) / CONFIG['stage_b_episodes'])
    else:
        autonomy = 1.0
    if state['force_autonomy']:
        autonomy = 1.0

    # Professor: vire em direção ao alvo (limitado à taxa máxima de giro)
    teacher_dir = get_target_direction()
    max_turn = CONFIG['turn_rate'] * time.dt
    teacher_turn = max(-max_turn, min(max_turn, signed_angle(worm.direction, teacher_dir)))

    # Política: a ação amostrada define um giro discreto
    policy_turn = TURN_MULTIPLIERS[action] * max_turn

    # Mistura conforme a autonomia (autonomy=0 → 100% professor)
    turn = lerp(teacher_turn, policy_turn, autonomy)

    # Aplica o giro e move
    new_dir = turn_vector(worm.direction, turn)
    if new_dir.length() > 0.01:
        worm.direction = new_dir.normalized()
    worm.step(time.dt)

    # ── Recompensa por passo e coleta do episódio (Fases 3/4) ─────────────────
    reward = calculate_reward(worm, env, state)

    # ── Penalidade por repetição de ação (Fase 7: anti-colapso) ──────────────
    # Se o verme escolhe a mesma ação muitas vezes seguidas, penaliza para
    # quebrar o loop "anda reto / sempre vira esquerda".
    state['action_history'].append(action)
    window = CONFIG['action_repeat_window']
    if len(state['action_history']) >= window:
        recent = state['action_history'][-window:]
        if len(set(recent)) == 1:
            reward -= CONFIG['action_repeat_penalty']

    teacher = teacher_action()
    state['episode'].append((sensors, action, reward, teacher))
    state['total_reward'] += reward

    # ── Métricas do episódio (critério de aceite da Fase 4) ───────────────────
    dist_rain  = _nearest_dist(worm.head.position, env.rain_sources)
    dist_light = _nearest_dist(worm.head.position, env.light_sources)
    if dist_rain is not None and dist_rain < CONFIG['arrival_radius']:
        state['steps_in_rain'] += 1
    if dist_light is not None and dist_light < CONFIG['arrival_radius']:
        state['steps_in_danger'] += 1

    # ── Treino episódico: chega em H passos → treina e troca a cena ───────────
    if len(state['episode']) >= CONFIG['episode_steps']:
        finish_episode()

    # ── Log periódico ─────────────────────────────────────────────────────────
    state['log_timer'] += time.dt
    if state['log_timer'] >= CONFIG['log_interval']:
        state['log_timer'] = 0.0
        print(
            f"ep={state['episode_count']+1}[{stage}]  acao={ACTIONS[action]:>14}  "
            f"recompensa={reward:+.3f}  "
            f"acumulada={state['total_reward']:+.1f}  "
            f"autonomia={autonomy*100:.0f}%"
        )

    # ── HUD (Fase 5): métricas na tela ────────────────────────────────────────
    update_hud()


# ─── INPUT ────────────────────────────────────────────────────────────────────
def input(key):

    # ── Zoom ──────────────────────────────────────────────────────────────────
    if key == 'scroll up':
        camera.z = min(CONFIG['cam']['min_zoom'] * -1, camera.z + CONFIG['cam']['zoom_speed'])
    if key == 'scroll down':
        camera.z = max(CONFIG['cam']['max_zoom'] * -1, camera.z - CONFIG['cam']['zoom_speed'])

    # ── Modos do editor ───────────────────────────────────────────────────────
    # Cada tecla alterna o modo ativo. Pressionar a mesma tecla duas vezes
    # cancela o modo (volta para 'none') — comportamento toggle.
    if key == '1':
        editor['mode'] = 'place_light' if editor['mode'] != 'place_light' else 'none'
        print(f"  Modo: {editor['mode']}")

    if key == '2':
        editor['mode'] = 'place_rain' if editor['mode'] != 'place_rain' else 'none'
        print(f"  Modo: {editor['mode']}")

    if key == '3':
        editor['mode'] = 'delete' if editor['mode'] != 'delete' else 'none'
        print(f"  Modo: {editor['mode']}")

    # ── Clique esquerdo: ação do modo atual ───────────────────────────────────
    if key == 'left mouse down' and editor['mode'] != 'none':

        if editor['mode'] == 'delete':
            # mouse.hovered_entity retorna a entidade sob o cursor, se houver.
            # Verificamos se é uma fonte conhecida antes de destruir.
            target = mouse.hovered_entity
            if target and (target in env.light_sources or target in env.rain_sources):
                env.delete_source(target)

        else:
            # camera.raycast dispara um raio da câmera até o mouse e retorna o
            # ponto de interseção com o collider atingido (ground_collider).
            hit = camera.raycast(distance=200, ignore=[worm.head] + worm.segments)
            if hit.hit:
                pos = hit.world_point
                if editor['mode'] == 'place_light':
                    env.place_light(pos)
                elif editor['mode'] == 'place_rain':
                    env.place_rain(pos)

    # ── Outras teclas ─────────────────────────────────────────────────────────
    if key == 'r':
        reset()

    if key == 'a':
        state['force_autonomy'] = not state['force_autonomy']
        print(f"  Professor {'DESLIGADO' if state['force_autonomy'] else 'LIGADO'} "
              f"(autonomia forçada = {state['force_autonomy']})")

    if key == 'p':
        toggle_policy_grid()

    if key == 's':
        save_brain(CONFIG['weights_file'], brain, critic)
        print(f"  Pesos salvos em {CONFIG['weights_file']}")

    if key == 'l':
        if os.path.exists(CONFIG['weights_file']):
            load_brain(CONFIG['weights_file'], brain, critic)
            print(f"  Pesos carregados de {CONFIG['weights_file']}")
        else:
            print(f"  Nao ha pesos em {CONFIG['weights_file']}")

    if key == 'escape':
        # ESC cancela o modo ativo primeiro; segundo ESC fecha o programa
        if editor['mode'] != 'none':
            editor['mode'] = 'none'
            print("  Modo cancelado.")
        else:
            quit()


print("\n-- Verme Neural ---------------------------------------")
if EVAL_MODE:
    print("  MODO AVALIACAO: professor desligado, sem treino")
    print("  carrega pesos.json (politica salva) e usa a temperatura de decisao")
    if EVAL_EPISODES:
        print(f"  Roda {EVAL_EPISODES} episodios, salva pesos e fecha")
elif EVAL_EPISODES:
    print(f"  MODO TREINO com limite: {EVAL_EPISODES} episodios, salva pesos e fecha")
print("  Botão direito + mouse -> orbitar câmera")
print("  WASD                  -> mover foco da câmera")
print("  Scroll                -> zoom")
print("  R                     -> reiniciar verme e cérebro")
print("  A                     -> ligar/desligar PROFESSOR (autonomia total)")
print("  P                     -> grade de setas da política aprendida")
print("  S / L                 -> salvar / carregar pesos (pesos.json)")
print("  ESC                   -> cancelar modo / sair")
print("  -- Editor -----------------------------------------")
print("  [1] + clique esquerdo -> colocar fonte de LUZ")
print("  [2] + clique esquerdo -> colocar fonte de CHUVA")
print("  [3] + clique esquerdo -> deletar bloco")
print("  (pressione a tecla de modo novamente para cancelar)")
print("---------------------------------------------------------\n")

app.run()