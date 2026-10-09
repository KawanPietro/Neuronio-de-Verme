import csv
import math
import os
import random
import sys

from ursina import *

from config import CONFIG
from environment import Environment
from mlp import (ACTIONS, PolicyNetwork, CriticNetwork, entropy, save_brain,
                  load_brain, ReplayBuffer, compute_gae)
from perception import get_sensor_inputs, calculate_reward
from worm import Worm

app = Ursina()

# ─── MODO DE EXECUÇÃO ─────────────────────────────────────────────────────────
# python main.py [--eval] [--visual] [--present] [--episodes=N] [--set k=v ...]
#   --eval          → avaliação: professor desligado, sem treino, carrega pesos
#   --visual        → passeio livre: sem treino, sem professor, sem CSV
#   --present       → apresentação: como o visual, mas HUD só didático
#                     (--apresentacao e --presentation são apelidos)
#   --episodes=N    → roda N episódios e fecha (treino ou eval)
#   --set k=v       → sobrescreve CONFIG (ex: --set=food_curriculum=1)
EVAL_MODE = '--eval' in sys.argv
VISUAL_MODE = '--visual' in sys.argv
PRESENT_MODE = ('--present' in sys.argv or '--apresentacao' in sys.argv
                or '--presentation' in sys.argv)
# Apresentação tem prioridade: nunca treina nem avalia, mesmo com --eval junto.
if PRESENT_MODE:
    EVAL_MODE = False
    VISUAL_MODE = False
EPISODE_LIMIT = None
for _arg in sys.argv:
    if _arg.startswith('--episodes='):
        EPISODE_LIMIT = int(_arg.split('=')[1])
    elif _arg.startswith('--set='):
        key, value = _arg[len('--set='):].split('=', 1)
        try:
            value = float(value) if ('e' in value.lower() or '.' in value) else int(value)
        except ValueError:
            pass
        CONFIG[key] = value
        print(f"  CONFIG['{key}'] = {CONFIG[key]}")
print(f"  Terreno limpo 8-dim (map_limit={CONFIG.get('map_limit', 32)})")

random.seed(CONFIG['seed'])

N_INPUTS_ACTIVE = CONFIG['n_inputs']  # 8 canonico (terreno limpo)

# ─── ILUMINAÇÃO ───────────────────────────────────────────────────────────────
DirectionalLight(y=2, z=-1)
AmbientLight(color=Color(0.6, 0.6, 0.6, 1))

# ─── CHÃO E CÉU ───────────────────────────────────────────────────────────────
gs = CONFIG['ground_scale']
Entity(
    model='plane',
    scale=gs,
    texture='grass',
    texture_scale=(gs, gs),
    color=color.white,
)

sky = Entity(
    model='sphere',
    scale=900,
    double_sided=True,
    texture='sky_sunset',
    color=color.white.tint(-0.2),
)

# ─── PLANO INVISÍVEL DE POSICIONAMENTO ────────────────────────────────────────
ground_collider = Entity(
    model='plane',
    scale=gs,
    collider='box',
    visible=False,
    y=0,
)

# ─── AMBIENTE (limpo default; gigante opt-in via --set=maze_enabled=1) ──────
env = Environment()
MAZE = None
if CONFIG.get('maze_enabled', 0):
    try:
        from maze import Maze as _Maze
        MAZE = _Maze.load(CONFIG.get('maze_file', 'labirintos.json'),
                          CONFIG.get('maze_key', 'gigante'))
        env.maze = MAZE
        print(f"  Labirinto '{CONFIG.get('maze_key')}' muros={len(MAZE.walls)}")
    except Exception as e:
        print(f"  AVISO maze_enabled=1 sem gigante valido ({e}) — limpo")
        MAZE = None
SPAWN = Vec3(0, 1.25, -12)
try:
    _d0 = env.food_max_dist(0)
    env.randomize_food_near(SPAWN.x, SPAWN.z, _d0, n_foods=1)
except Exception:
    pass

# ─── VERME ────────────────────────────────────────────────────────────────────
worm = Worm()

worm_light = PointLight(
    parent=worm.head,
    color=color.cyan,
    position=(0, 5, 0),
    intensity=1.5,
)

# ─── CÂMERA ───────────────────────────────────────────────────────────────────
# L17-mapa: distâncias dobradas p/ enquadrar o mapa 32 (era 16; scroll ajusta).
cam_pivot = Entity()
cam_pivot.y = 20

camera.parent = cam_pivot
camera.position = (0, 60, -125)
camera.rotation = (24, 0, 0)

# ─── CÉREBRO: política 8 → 32 → 16 → 5 ações ──────────────────────────────────
brain = PolicyNetwork(
    N_INPUTS_ACTIVE,
    CONFIG['n_hidden'],
    CONFIG['n_actions'],
    temperature=CONFIG['temperature'],
    n_hidden2=CONFIG.get('n_hidden2'),
)

# ─── CRÍTICO: V(s) 8 → 32 → 16 → 1 ───────────────────────────────────────────
critic = CriticNetwork(
    N_INPUTS_ACTIVE,
    CONFIG['n_hidden'],
    n_hidden2=CONFIG.get('n_hidden2'),
)

# ─── REPLAY BUFFER ────────────────────────────────────────────────────────────
replay_buffer = ReplayBuffer(max_episodes=CONFIG.get('buffer_max_episodes', 50))

# ─── MODO AVALIAÇÃO ───────────────────────────────────────────────────────────
if EVAL_MODE:
    if os.path.exists(CONFIG['weights_file']):
        load_brain(CONFIG['weights_file'], brain, critic)
        print(f"  Avaliando pesos salvos em {CONFIG['weights_file']}")
    else:
        print(f"  ATENCAO: {CONFIG['weights_file']} nao existe — avaliando cerebro aleatorio")
    brain.temperature = CONFIG['min_temperature']

if VISUAL_MODE:
    brain.temperature = CONFIG['min_temperature']
    print("  MODO VISUALIZACAO: sem treino, professor desligado, só passeio")
    print(f"  build diag-frio-1 (teleport+bounce+pos/dir/dt no HUD) arq={__file__}")

if PRESENT_MODE:
    brain.temperature = CONFIG['min_temperature']
    print("  MODO APRESENTACAO: HUD didático, sem treino, sem números de treino")

# ─── MODO DE EDIÇÃO ───────────────────────────────────────────────────────────
editor = {'mode': 'none'}

# ─── ESTADO ───────────────────────────────────────────────────────────────────
state = {
    'total_reward'    : 0.0,
    'log_timer'       : 0.0,
    'prev_dist_food'  : None,
    'prev_position'   : None,
    'hunger'          : 0.0,   # 0 saciado → 1 faminto; zera ao comer
    'hunger_rate'     : CONFIG.get('hunger_rate', 0.001),
    'encontros_food'  : 0,
    'passos_ate_comer': None,  # passo do 1º encontro (métrica primária)
    'episode'         : [],     # (sensors, action, reward, acao_professor)
    'episode_count'   : 0,
    'force_autonomy'  : False,  # tecla A: professor desligado
    'episode_rewards' : [],
    'action_history'  : [],
    'encontros_hist'  : [],     # encontros por episódio (taxa de sucesso)
    'foods_eaten_total': 0,     # contador didático (visual/apresentação)
    'present_eating_flash': 0.0,  # timer do "COMENDO!" na apresentação
}
# Última ação p/ HUD (evita KeyError no 1º frame antes do update).
_last_action = [2]

if EVAL_MODE or VISUAL_MODE or PRESENT_MODE:
    state['force_autonomy'] = True

# ─── LOG DE EPISÓDIOS ─────────────────────────────────────────────────────────
# Visualização e Apresentação são só passeio: não truncam nem escrevem CSV.
if VISUAL_MODE or PRESENT_MODE:
    log_csv = None
    csv_writer = None
else:
    log_csv = open(CONFIG['log_csv'], 'w', newline='')
    csv_writer = csv.writer(log_csv)
    csv_writer.writerow(['episodio', 'estagio', 'recompensa_total', 'recompensa_media',
                         'retorno_medio', 'entropia_media', 'learning_rate',
                         'lambda_imitacao', 'autonomia_forcada',
                         'encontros_food', 'passos_ate_comer',
                         'acao_principal', 'semente'])

# ─── HUD (3 camadas: selo do modo + painel principal + ajuda) ───────────────
# Foco é sempre o verme: painéis pequenos, semi-transparentes, nos cantos.
MODE_META = {
    'TREINO'      : {'cor': color.orange, 'simbolo': '●'},
    'AVALIACAO'   : {'cor': color.azure,  'simbolo': '◆'},
    'VISUALIZACAO': {'cor': color.lime,   'simbolo': '○'},
    'APRESENTACAO': {'cor': color.yellow, 'simbolo': '★'},
}


def _mode_key():
    if PRESENT_MODE:
        return 'APRESENTACAO'
    if VISUAL_MODE:
        return 'VISUALIZACAO'
    if EVAL_MODE:
        return 'AVALIACAO'
    return 'TREINO'


badge = Text(
    text='',
    position=(-0.85, 0.47),
    origin=(-0.5, 0.5),
    scale=1.6,
    color=color.white,
    background_color=Color(0, 0, 0, 0.65),
)
hud = Text(
    text='',
    position=(-0.85, 0.36),
    origin=(-0.5, 0.5),
    scale=1,
    color=color.white,
    background_color=Color(0, 0, 0, 0.55),
)
footer = Text(
    text='',
    position=(-0.85, -0.44),
    origin=(-0.5, -0.5),
    scale=0.85,
    color=Color(1, 1, 1, 0.85),
    background_color=Color(0, 0, 0, 0.45),
)

ACTION_ARROW = ['◀', '◁', '▲', '▷', '▶']
ACTION_PT = ['virando à esquerda', 'virando de leve à esquerda',
             'andando em frente', 'virando de leve à direita',
             'virando à direita']


def _bar(frac, width=10):
    """Barra textual 0..1, ex: ████░░░░░░."""
    frac = max(0.0, min(1.0, float(frac or 0.0)))
    full = int(round(frac * width))
    return '█' * full + '░' * (width - full)


def _food_info():
    """(distância, cheiro) até o alimento mais próximo — didático e útil."""
    d = _nearest_dist(worm.head.position, getattr(env, 'food_sources', []))
    if d is None:
        return None, 0.0
    return d, 1.0 / (1.0 + d)


# ─── MOVIMENTO POR GIRO (ações discretas) ─────────────────────────────────────
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


# ─── PROFESSOR (campo de potencial unimodal: só atração) ──────────────────────
def _add_edge_avoidance(combined):
    margin = CONFIG['wall_margin']
    limit = CONFIG['map_limit']
    p = worm.head.position
    if p.x < -limit + margin:
        combined += Vec3(1, 0, 0) * (1.0 + (-limit + margin - p.x) / margin)
    if p.x > limit - margin:
        combined += Vec3(-1, 0, 0) * (1.0 + (p.x - (limit - margin)) / margin)
    if p.z < -limit + margin:
        combined += Vec3(0, 0, 1) * (1.0 + (-limit + margin - p.z) / margin)
    if p.z > limit - margin:
        combined += Vec3(0, 0, -1) * (1.0 + (p.z - (limit - margin)) / margin)


def get_target_direction():
    """Direção ideal: atração ao alimento + fuga de bordas (terreno limpo)."""
    combined = Vec3(0, 0, 0)
    for fs in getattr(env, 'food_sources', []):
        to_food = fs.position - worm.head.position
        dist = to_food.length()
        if dist > 0.01:
            weight = max(0.0, 1.0 - (dist / CONFIG['sensor_max_dist']))
            combined += to_food.normalized() * weight
    _add_edge_avoidance(combined)
    if Vec3(combined).length() > 0.01:
        return Vec3(combined).normalized()
    return worm.direction


def get_teacher_dir():
    return get_target_direction()


# ─── CURRÍCULO DE AUTONOMIA ───────────────────────────────────────────────────
STAGE_NAMES = {'A': 'IMITACAO', 'B': 'HIBRIDO', 'C': 'AUTONOMO'}


def current_stage():
    n = state['episode_count']
    if n < CONFIG['stage_a_episodes']:
        return 'A'
    if n < CONFIG['stage_a_episodes'] + CONFIG['stage_b_episodes']:
        return 'B'
    return 'C'


def lambda_imitation():
    stage = current_stage()
    if stage == 'B':
        k = state['episode_count'] - CONFIG['stage_a_episodes']
        return CONFIG['lambda_start'] * (CONFIG['lambda_decay'] ** k)
    if stage == 'C':
        return float(CONFIG.get('lambda_c', 0.0) or 0.0)
    return 0.0


def teacher_action():
    target_dir = get_teacher_dir()
    desired = signed_angle(worm.direction, target_dir)
    max_turn = CONFIG['turn_rate'] * time.dt
    best, best_err = 0, float('inf')
    for a, m in enumerate(TURN_MULTIPLIERS):
        err = abs(m * max_turn - desired)
        if err < best_err:
            best_err, best = err, a
    return best


def _nearest_dist(position, sources):
    best = None
    for s in sources:
        d = (position - s.position).length()
        if best is None or d < best:
            best = d
    return best


def episode_stats(brain, episode):
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


# ─── VISUALIZAÇÃO DA POLÍTICA (tecla P) ───────────────────────────────────────
policy_grid = {'visible': False, 'pivots': []}


def _fake_worm_at(x, z):
    class _Head:
        position = Vec3(x, 0, z)
    class _Worm:
        head = _Head()
        direction = Vec3(0, 0, 1)
    return _Worm()


def build_policy_grid():
    n = CONFIG['grid_cells']
    span = CONFIG['map_limit'] * 1.85
    step = span / max(1, n - 1)
    off = span / 2
    for gx in range(n):
        for gz in range(n):
            x, z = -off + gx * step, -off + gz * step
            pivot = Entity(position=(x, 0.4, z))
            Entity(model='cube', scale=(0.12, 0.12, 1.0), color=color.gray,
                   parent=pivot, y=0)
            Entity(model='cone', scale=(0.3, 0.3, 0.4), position=(0, 0, 1.0),
                   rotation_x=90, parent=pivot)
            pivot.visible = False
            policy_grid['pivots'].append(pivot)


def refresh_policy_grid():
    n = CONFIG['grid_cells']
    span = CONFIG['map_limit'] * 1.85
    step = span / max(1, n - 1)
    off = span / 2
    for gx in range(n):
        for gz in range(n):
            pivot = policy_grid['pivots'][gx * n + gz]
            x, z = -off + gx * step, -off + gz * step

            sensors = get_sensor_inputs(_fake_worm_at(x, z), env, state)
            probs = brain.probabilities(sensors)
            action = max(range(len(probs)), key=lambda a: probs[a])

            max_turn = CONFIG['turn_rate'] * 0.016
            newdir = turn_vector(Vec3(0, 0, 1), TURN_MULTIPLIERS[action] * max_turn)
            pivot.rotation_y = math.degrees(math.atan2(newdir.x, newdir.z))

            c = color.gray
            foods = getattr(env, 'food_sources', [])
            if foods:
                d = Vec3(pivot.position.x, 0, pivot.position.z)
                f = min(foods, key=lambda s: (Vec3(s.x, 0, s.z) - d).length())
                toward = (Vec3(f.x, 0, f.z) - d).normalized()
                if newdir.dot(toward) > 0.25:
                    c = color.lime
            for child in pivot.children:
                child.color = c


def toggle_policy_grid():
    if not policy_grid['pivots']:
        build_policy_grid()
    policy_grid['visible'] = not policy_grid['visible']
    if policy_grid['visible']:
        refresh_policy_grid()
    for pivot in policy_grid['pivots']:
        pivot.visible = policy_grid['visible']
    print(f"  Grade da politica {'LIGADA' if policy_grid['visible'] else 'DESLIGADA'}")


# ─── HUD ──────────────────────────────────────────────────────────────────────
def _paint_badge():
    key = _mode_key()
    meta = MODE_META[key]
    badge.color = meta['cor']
    if key == 'APRESENTACAO':
        badge.text = f"{meta['simbolo']}  APRESENTAÇÃO  —  Neurônio de Verme"
    elif key == 'VISUALIZACAO':
        badge.text = f"{meta['simbolo']}  VISUALIZAÇÃO"
    elif key == 'AVALIACAO':
        badge.text = f"{meta['simbolo']}  AVALIAÇÃO  ·  sem treino"
    else:
        badge.text = f"{meta['simbolo']}  TREINO"


def _success_rate(window=10):
    hist = state.get('encontros_hist', [])
    tail = hist[-window:] if hist else []
    if not tail:
        return None
    return sum(1 for e in tail if e > 0) / len(tail)


def update_hud():
    """Painel de trabalho: treino ou avaliação. Compacto, só o útil."""
    _paint_badge()
    window = CONFIG['rolling_window']
    rolling = state['episode_rewards'][-window:]
    media = sum(rolling) / len(rolling) if rolling else 0.0
    stage = current_stage()
    passo = len(state['episode'])
    total_steps = CONFIG['episode_steps']
    dist, smell = _food_info()
    dist_txt = f"{dist:.1f}m" if dist is not None else "—"
    taxa = _success_rate(window)
    p1 = state['passos_ate_comer']
    fome = state.get('hunger', 0.0)

    if EVAL_MODE:
        linha_ep = f"ep {state['episode_count']+1}  ·  passo {passo}/{total_steps}"
        linha_sucesso = (f"encontros {state['encontros_food']}  ·  "
                         f"1º em {p1 if p1 is not None else '—'}  ·  "
                         f"sucesso({len(rolling)}/{window}): "
                         f"{taxa*100:.0f}%" if taxa is not None else
                         f"encontros {state['encontros_food']}")
        hud.text = (
            f"{linha_ep}\n"
            f"{linha_sucesso}\n"
            f"recomp. média: {media:+.2f}  ·  total ep: {state['total_reward']:+.1f}\n"
            f"comida {dist_txt}  ·  faro {smell:.2f}  ·  {ACTION_ARROW[_last_action[0]]} {ACTIONS[_last_action[0]]}\n"
        )
        footer.text = "scroll zoom  ·  botão direito orbita  ·  R reinicia  ·  ESC sai"
    else:
        # Autonomia atual (mesma conta do update, p/ exibir sem duplicar lógica lá)
        if stage == 'A':
            autonomia = 0.0
        elif stage == 'B':
            k = state['episode_count'] - CONFIG['stage_a_episodes']
            autonomia = min(1.0, (k + 1) / CONFIG['stage_b_episodes'])
        else:
            autonomia = 1.0
        if state['force_autonomy']:
            autonomia = 1.0
        prof = 'DESLIGADO (A p/ ligar)' if state['force_autonomy'] else 'LIGADO (A p/ desligar)'
        hud.text = (
            f"ep {state['episode_count']+1}  ·  {stage} {STAGE_NAMES[stage]}  ·  autonomia {autonomia*100:.0f}%\n"
            f"passo {passo}/{total_steps}  ·  encontros {state['encontros_food']}  ·  1º em {p1 if p1 is not None else '—'}\n"
            f"recomp. média({len(rolling)}/{window}): {media:+.2f}  ·  fome {_bar(fome, 6)} {fome*100:.0f}%\n"
            f"comida {dist_txt}  ·  faro {smell:.2f}  ·  {ACTION_ARROW[_last_action[0]]} {ACTIONS[_last_action[0]]}\n"
            f"professor {prof}\n"
        )
        footer.text = "A professor  ·  P grade  ·  S/L salva/carrega  ·  1/3 editor  ·  ESC sai"


def update_hud_aquarium(action):
    """Painel da Visualização: observar o comportamento, sem números de treino."""
    _paint_badge()
    probs = brain.last_probs
    conf = max(probs) * 100 if probs else 0.0
    dist, smell = _food_info()
    dist_txt = f"{dist:.1f}m" if dist is not None else "—"
    # Diagnóstico de movimento (some quando o bug congelar estiver resolvido):
    # se pos não muda com dir≠0 e dt>0, o problema é fora da lógica (cena/engine).
    _dd = worm.direction
    hud.text = (
        f"{ACTION_ARROW[action]}  {ACTION_PT[action]}  (certeza {conf:.0f}%)\n"
        f"comida verde a {dist_txt}  ·  faro {smell:.2f}\n"
        f"comidas até aqui: {state.get('foods_eaten_total', 0)}\n"
        f"pos x{worm.head.position.x:+.1f} z{worm.head.position.z:+.1f}"
        f"  dir({_dd.x:+.2f},{_dd.z:+.2f})|d|={_dd.length():.2f}"
        f"  dt={time.dt*1000:.1f}ms\n"
    )
    footer.text = "só observando  ·  P grade  ·  R reinicia  ·  scroll zoom  ·  ESC sai"


def update_hud_present(action):
    """Painel da Apresentação: só o didático. NADA de treino/avaliação."""
    _paint_badge()
    dist, smell = _food_info()
    dist_txt = f"{dist:.1f}m" if dist is not None else "—"
    flash = state.get('present_eating_flash', 0.0)
    if flash > 0:
        status = "COMENDO!  A fome zerou e nasceu outra comida."
    elif dist is not None and dist < CONFIG.get('food_radius', 3.0) + 2.0:
        status = "Chegou pertinho — vai comer!"
    elif smell > 0.08:
        status = "Sentiu o cheiro e está se aproximando..."
    else:
        status = "Procurando a comida pelo faro..."
    hud.text = (
        f"O verme está {status}\n"
        f"● comida verde a {dist_txt}  ·  faro {smell:.2f}  ·  {ACTION_ARROW[action]} {ACTION_PT[action]}\n"
        f"comidas: {state.get('foods_eaten_total', 0)}\n"
    )
    footer.text = "azul = verme  ·  verde = comida  ·  scroll zoom  ·  botão direito gira  ·  ESC sai"


# ─── REINICIAR ────────────────────────────────────────────────────────────────
def reset():
    worm.reset()
    worm.teleport(SPAWN)
    state.update({
        'total_reward'    : 0.0,
        'log_timer'       : 0.0,
        'prev_dist_food'  : None,
        'prev_position'   : None,
        'hunger'          : 0.0,
        'encontros_food'  : 0,
        'passos_ate_comer': None,
        'episode'         : [],
        'episode_count'   : 0,
        'episode_rewards' : [],
        'encontros_hist'  : [],
        'foods_eaten_total': 0,
        'present_eating_flash': 0.0,
    })
    _last_action[0] = 2
    brain.reset()
    print("\n-- Reiniciado -----------------------------------------\n")


# ─── FIM DE EPISÓDIO ──────────────────────────────────────────────────────────
def finish_episode():
    stage = current_stage()
    lam = lambda_imitation()

    if EVAL_MODE:
        # Avaliação: só mede, SEM treino.
        stats = episode_stats(brain, state['episode'])
    else:
        scaled_rewards = [r * CONFIG['reward_scale'] for _, _, r, _ in state['episode']]
        values = []
        for sensors, *_ in state['episode']:
            values.append(critic.value(sensors))
        advantages = compute_gae(scaled_rewards, values, CONFIG['gamma'],
                                CONFIG['gae_lambda'])
        returns = [a + v for a, v in zip(advantages, values)]
        old_probs_list = []
        for sensors, action, *_ in state['episode']:
            brain.probabilities(sensors)
            old_probs_list.append(brain.last_probs[action])

        replay_buffer.add_episode(state['episode'], advantages,
                                  returns=returns, old_probs=old_probs_list)

        use_buffer = (EPISODE_LIMIT is not None) and len(replay_buffer) >= CONFIG.get('buffer_min_transitions', 200)
        if use_buffer:
            stats = brain.ppo_update_from_buffer(
                replay_buffer, critic,
                epochs=CONFIG.get('buffer_epochs', 2),
                batch_size=CONFIG.get('buffer_batch_size', 64),
                ppo_clip=CONFIG['ppo_clip'],
                value_coef=CONFIG['value_coef'],
                gae_lambda=CONFIG['gae_lambda'],
            )
        else:
            stats = brain.update_episode(state['episode'], imitation_weight=lam,
                                         critic=critic, value_coef=CONFIG['value_coef'],
                                         gae_lambda=CONFIG['gae_lambda'],
                                         ppo_clip=CONFIG['ppo_clip'])

    if not EVAL_MODE:
        brain.learning_rate *= CONFIG['lr_decay']
        brain.temperature = max(
            CONFIG['min_temperature'],
            brain.temperature * CONFIG['temperature_decay'],
        )

    state['episode_count'] += 1
    total = sum(r for _, _, r, _ in state['episode'])
    state['episode_rewards'].append(total)
    state['encontros_hist'].append(state['encontros_food'])
    principal = max(range(len(stats['action_counts'])),
                    key=lambda a: stats['action_counts'][a])
    if csv_writer is not None:
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
            state['encontros_food'],
            state['passos_ate_comer'] if state['passos_ate_comer'] is not None else '',
            ACTIONS[principal],
            CONFIG['seed'],
        ])
        log_csv.flush()

    actions = ' '.join(f"{ACTIONS[a][:6]}:{c}" for a, c in enumerate(stats['action_counts']))
    print(
        f"\n-- Episodio {state['episode_count']} [estagio {stage}: {STAGE_NAMES[stage]}] -----\n"
        f"  total={total:+.1f}  media={stats['mean_reward']:+.4f}  "
        f"entropia={stats['mean_entropy']:.3f}  lr={brain.learning_rate:.5f}\n"
        f"  encontros_food={state['encontros_food']}  "
        f"passos_ate_comer={state['passos_ate_comer']}\n"
        f"  acoes: {actions}\n"
    )

    if EPISODE_LIMIT and state['episode_count'] >= EPISODE_LIMIT:
        save_brain(CONFIG['weights_file'], brain, critic)
        print(f"\n-- Concluido ({EPISODE_LIMIT} episodios): pesos salvos em {CONFIG['weights_file']}")
        quit()

    if not EVAL_MODE:
        _ckpt = CONFIG.get('checkpoint_interval', 50)
        if _ckpt and state['episode_count'] % _ckpt == 0:
            try:
                save_brain(CONFIG['weights_file'], brain, critic)
                print(f"  [checkpoint ep{state['episode_count']}] pesos salvos em {CONFIG['weights_file']}")
            except Exception as e:
                print(f"  [checkpoint ep{state['episode_count']}] FALHA ao salvar: {e}")

    # Novo episódio: alimento pelo currículo (perto→longe do spawn), verme no spawn
    _maxd = env.food_max_dist(state['episode_count'])
    try:
        env.randomize_food_near(SPAWN.x, SPAWN.z, _maxd, n_foods=1)
    except Exception:
        env.randomize_food(n_foods=1, limit=CONFIG.get('food_limit', 12))
    worm.reset()
    worm.teleport(SPAWN)
    state['episode'] = []
    state['encontros_food'] = 0
    state['passos_ate_comer'] = None
    state['hunger'] = 0.0
    state['action_history'] = []
    state['prev_position'] = Vec3(worm.head.position)
    state['prev_dist_food'] = None


# ─── UPDATE (loop único por frame) ────────────────────────────────────────────
def update():
    cam_pivot.position = lerp(cam_pivot.position, worm.head.position, 0.1)
    worm_light.position = worm.head.position + Vec3(0, 5, 0)
    worm.animate()

    if mouse.right:
        cam_pivot.rotation_y += mouse.velocity[0] * CONFIG['cam']['rot_speed']
        cam_pivot.rotation_x -= mouse.velocity[1] * CONFIG['cam']['rot_speed']
        cam_pivot.rotation_x  = clamp(cam_pivot.rotation_x, -80, 80)

    if held_keys['w']: cam_pivot.y += CONFIG['cam']['pan_speed'] * time.dt
    if held_keys['s']: cam_pivot.y -= CONFIG['cam']['pan_speed'] * time.dt
    if held_keys['a']: cam_pivot.x -= CONFIG['cam']['pan_speed'] * time.dt
    if held_keys['d']: cam_pivot.x += CONFIG['cam']['pan_speed'] * time.dt

    sensors = get_sensor_inputs(worm, env, state)
    action = brain.sample_action(sensors)

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

    teacher_dir = get_teacher_dir()
    max_turn = CONFIG['turn_rate'] * time.dt
    teacher_turn = max(-max_turn, min(max_turn, signed_angle(worm.direction, teacher_dir)))

    policy_turn = TURN_MULTIPLIERS[action] * max_turn
    turn = lerp(teacher_turn, policy_turn, autonomy)

    new_dir = turn_vector(worm.direction, turn)
    if new_dir.length() > 0.01:
        worm.direction = new_dir.normalized()
    worm.step(time.dt, maze=MAZE, bounce=(VISUAL_MODE or PRESENT_MODE))
    _last_action[0] = action

    # ── APRESENTAÇÃO: passeio didático, come e conta, sem treino/CVS ──────
    if PRESENT_MODE:
        if state.get('present_eating_flash', 0.0) > 0:
            state['present_eating_flash'] = max(
                0.0, state['present_eating_flash'] - time.dt)
        _d_food = _nearest_dist(worm.head.position, getattr(env, 'food_sources', []))
        _fr = CONFIG.get('food_radius', 3.0)
        if _d_food is not None and _d_food < _fr:
            state['foods_eaten_total'] = state.get('foods_eaten_total', 0) + 1
            state['present_eating_flash'] = 2.5
            _maxd_eat = env.food_max_dist(state['episode_count'])
            env.eat_and_respawn(
                min(env.food_sources,
                    key=lambda s: (s.position - worm.head.position).length()),
                limit=_maxd_eat,
                near=(worm.head.position.x, worm.head.position.z))
        update_hud_present(action)
        return

    if VISUAL_MODE:
        _d_food = _nearest_dist(worm.head.position, getattr(env, 'food_sources', []))
        _fr = CONFIG.get('food_radius', 3.0)
        if _d_food is not None and _d_food < _fr:
            state['foods_eaten_total'] = state.get('foods_eaten_total', 0) + 1
            state['hunger'] = 0.0
            _maxd_eat = env.food_max_dist(state['episode_count'])
            env.eat_and_respawn(
                min(env.food_sources,
                    key=lambda s: (s.position - worm.head.position).length()),
                limit=_maxd_eat,
                near=(worm.head.position.x, worm.head.position.z))
        state['log_timer'] += time.dt
        if state['log_timer'] >= CONFIG['log_interval']:
            state['log_timer'] = 0.0
            print(f"aquario  acao={ACTIONS[action]:>14}  food_dist={_nearest_dist(worm.head.position, getattr(env, 'food_sources', [])) or 0:.1f}")
        update_hud_aquarium(action)
        return

    reward = calculate_reward(worm, env, state, action=action)

    # ── Comer e reaparecer + fome ──────────────────────────────────────────
    _d_food = _nearest_dist(worm.head.position, getattr(env, 'food_sources', []))
    _fr = CONFIG.get('food_radius', 3.0)
    if _d_food is not None and _d_food < _fr:
        state['encontros_food'] += 1
        if state['passos_ate_comer'] is None:
            state['passos_ate_comer'] = len(state['episode']) + 1
        state['hunger'] = 0.0
        _maxd_eat = env.food_max_dist(state['episode_count'])
        env.eat_and_respawn(
            min(env.food_sources,
                key=lambda s: (s.position - worm.head.position).length()),
            limit=_maxd_eat,
            near=(worm.head.position.x, worm.head.position.z))
    else:
        state['hunger'] = min(1.0, state['hunger'] + state['hunger_rate'])

    teacher = teacher_action()
    state['episode'].append((sensors, action, reward, teacher))
    state['total_reward'] += reward

    if len(state['episode']) >= CONFIG['episode_steps']:
        finish_episode()

    state['log_timer'] += time.dt
    if state['log_timer'] >= CONFIG['log_interval']:
        state['log_timer'] = 0.0
        print(
            f"ep={state['episode_count']+1}[{stage}]  acao={ACTIONS[action]:>14}  "
            f"recompensa={reward:+.3f}  "
            f"acumulada={state['total_reward']:+.1f}  "
            f"autonomia={autonomy*100:.0f}%"
        )

    update_hud()


# ─── INPUT ────────────────────────────────────────────────────────────────────
def input(key):

    if key == 'scroll up':
        camera.z = min(CONFIG['cam']['min_zoom'] * -1, camera.z + CONFIG['cam']['zoom_speed'])
    if key == 'scroll down':
        camera.z = max(CONFIG['cam']['max_zoom'] * -1, camera.z - CONFIG['cam']['zoom_speed'])

    # Apresentação: tela limpa — sem editor, professor, save/load ou grade.
    if PRESENT_MODE and key in ('1', '3', 'a', 's', 'l', 'p'):
        print("  (modo Apresentação: tecla desativada p/ manter a tela limpa)")
        return

    if key == '1':
        editor['mode'] = 'place_food' if editor['mode'] != 'place_food' else 'none'
        print(f"  Modo: {editor['mode']}")

    if key == '3':
        editor['mode'] = 'delete' if editor['mode'] != 'delete' else 'none'
        print(f"  Modo: {editor['mode']}")

    if key == 'left mouse down' and editor['mode'] != 'none':

        if editor['mode'] == 'delete':
            target = mouse.hovered_entity
            if target and target in env.food_sources:
                env.delete_source(target)

        else:
            hit = camera.raycast(distance=200, ignore=[worm.head] + worm.segments)
            if hit.hit:
                pos = hit.world_point
                if editor['mode'] == 'place_food':
                    env.place_food(pos)

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
        if editor['mode'] != 'none':
            editor['mode'] = 'none'
            print("  Modo cancelado.")
        else:
            quit()


print("\n-- Verme Neural: terreno limpo + alimento -----------------")
print(f"  Terreno limpo | estado 8-dim | objetivo: COMER")
if PRESENT_MODE:
    print("  MODO APRESENTACAO: só o verme + didático (sem treino, sem CSV)")
    print("  Teclas ativas: scroll zoom, botão direito orbita, R reinicia, ESC sai")
elif VISUAL_MODE:
    print("  MODO VISUALIZACAO: passeio livre, sem treino (professor desligado)")
    if EPISODE_LIMIT:
        print(f"  Roda {EPISODE_LIMIT} episodios e fecha")
elif EVAL_MODE:
    print("  MODO AVALIACAO: professor desligado, sem treino")
    if EPISODE_LIMIT:
        print(f"  Roda {EPISODE_LIMIT} episodios, salva pesos e fecha")
elif EPISODE_LIMIT:
    print(f"  MODO TREINO com limite: {EPISODE_LIMIT} episodios, salva pesos e fecha")
print("  Botão direito + mouse -> orbitar câmera")
print("  WASD                  -> mover foco da câmera")
print("  Scroll                -> zoom")
print("  R                     -> reiniciar verme e cérebro")
print("  A                     -> ligar/desligar PROFESSOR (autonomia total)")
print("  P                     -> grade de setas da política aprendida")
print("  S / L                 -> salvar / carregar pesos (pesos.json)")
print("  ESC                   -> cancelar modo / sair")
print("  -- Editor -----------------------------------------")
print("  [1] + clique esquerdo -> colocar ALIMENTO")
print("  [3] + clique esquerdo -> deletar alimento")
print("  (pressione a tecla de modo novamente para cancelar)")
print("---------------------------------------------------------\n")

app.run()
