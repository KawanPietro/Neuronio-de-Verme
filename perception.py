"""Percepcao canonica: alimento 8-dim em terreno limpo (sem obstáculos).

Estado (todos normalizados):
  0-1 food_dir.x/z  vetor XZ unitario ate o alimento (0,0 se sem alimento)
  2   food_dist     distancia normalizada [0,1]
  3   smell         olfato 1/(1+dist) em [0,1] — gradiente denso mesmo longe
  4-5 vel_x/z       propriocepcao [-1,1]
  6-7 borda_x/z     distancia a borda [0,1] (1=centro, 0=parede)

Recompensa por passo, clip [-1,1]:
  r = progresso + smell + eventos comer/permanecer x (1+fome).
"""
from ursina import Vec3

from config import CONFIG

FOOD_DIM = 8
STATE_DIM = 8  # alias canonico (n_inputs=8)


def _nearest(worm, sources):
    """Fonte mais próxima da cabeça e a distância 3D até ela."""
    if not sources:
        return None, None
    nearest = None
    nearest_dist = float('inf')
    for src in sources:
        d = (worm.head.position - src.position).length()
        if d < nearest_dist:
            nearest_dist = d
            nearest = src
    return nearest, nearest_dist


def _xz_unit(to_vec):
    """Vetor unitário no plano XZ (0, 0) se o vetor for (quase) nulo."""
    d2 = (to_vec.x ** 2 + to_vec.z ** 2) ** 0.5
    if d2 > 0.01:
        return to_vec.x / d2, to_vec.z / d2
    return 0.0, 0.0


def get_food_sensor_inputs(worm, env, state) -> list:
    """Estado 8-dim do alimento (canonico, terreno limpo)."""
    max_dist = CONFIG.get('sensor_max_dist', 60.0)
    limit = CONFIG.get('map_limit', 32)
    features = [0.0] * FOOD_DIM

    direction = getattr(worm, 'direction', None)
    if direction is None:
        dir_x, dir_z = 0.0, 1.0
    else:
        dir_x, dir_z = direction.x, direction.z

    # ── Alimento + olfato ──────────────────────────────────────────────────
    foods = list(getattr(env, 'food_sources', []) or [])
    food_src, dist_food = _nearest(worm, foods)
    if food_src is not None:
        to_food = food_src.position - worm.head.position
        dx, dz = _xz_unit(to_food)
        features[0] = dx
        features[1] = dz
        features[2] = min(dist_food / max_dist, 1.0)
        features[3] = 1.0 / (1.0 + dist_food)  # denso, [0,1]

    # ── Obstáculos removidos (terreno limpo): sem features 4-6 ─────────────────
    # Índices 4-7 = propriocepção + bordas (antes 7-10 no 11-dim).

    # ── Propriocepção + bordas ─────────────────────────────────────────────
    features[4] = max(-1.0, min(1.0, dir_x))
    features[5] = max(-1.0, min(1.0, dir_z))
    px = worm.head.position.x
    pz = worm.head.position.z
    features[6] = max(0.0, min(1.0, (limit - abs(px)) / limit))
    features[7] = max(0.0, min(1.0, (limit - abs(pz)) / limit))

    return features


def _state_hash(worm, env):
    """Hash da posição + direção para identificar estados únicos (novelty)."""
    px = int(round(worm.head.position.x * 2))
    pz = int(round(worm.head.position.z * 2))
    direction = getattr(worm, 'direction', None)
    if direction is None:
        dx, dz = 0, 1
    else:
        dx = int(round(direction.x * 2))
        dz = int(round(direction.z * 2))
    return (px, pz, dx, dz)


def calculate_food_reward(worm, env, state, action=None) -> float:
    """Recompensa unimodal do alimento, por passo, clip [-1,1].

    Atualiza apenas prev_dist_food/prev_position.
    Lê `hunger` como multiplicador (NÃO incrementa aqui; o loop incrementa).
    """
    radius = CONFIG.get('food_radius', 3.0)
    reward = 0.0

    foods = list(getattr(env, 'food_sources', []) or [])
    _, dist_food = _nearest(worm, foods)

    # ── Olfato denso (sinal mesmo longe) ───────────────────────────────────
    if dist_food is not None and dist_food > 0.05:
        reward += CONFIG.get('smell_bonus', 0.3) / (1.0 + dist_food)

    # ── Progresso (aproximar = bom; sem termo negativo) ────────────────────
    prev_food = state.get('prev_dist_food', None)
    if dist_food is not None and prev_food is not None:
        reward += CONFIG.get('progress_scale', 3.0) * (prev_food - dist_food)

    # ── Eventos comer / permanecer ─────────────────────────────────────────
    if dist_food is not None:
        if prev_food is not None and dist_food < radius <= prev_food:
            reward += CONFIG.get('eat_bonus', 5.0)
        if dist_food < radius:
            reward += CONFIG.get('inside_food_reward', 0.5)

    # ── Terreno limpo: sem termos de obstáculo (colisão/proximidade/afastamento) ─

    # ── Recompensa por novidade (L9) ───────────────────────────────────────
    novelty_bonus = CONFIG.get('novelty_bonus', 0.2)
    if novelty_bonus > 0:
        h = _state_hash(worm, env)
        visited = state.setdefault('novelty_visited', {})
        count = visited.get(h, 0)
        reward += novelty_bonus * (CONFIG.get('novelty_decay', 0.99) ** count)
        visited[h] = count + 1
        if len(visited) > CONFIG.get('novelty_window', 200):
            visited.pop(next(iter(visited)))

    # ── Penalidade por repetição de ação (L10) ────────────────────────────
    if action is not None:
        history = state.setdefault('action_history', [])
        history.append(action)
        window = CONFIG.get('action_repeat_window', 20)
        if len(history) > window:
            history.pop(0)
        if len(history) >= window:
            recent = history[-window:]
            if len(set(recent)) == 1:
                reward -= CONFIG.get('action_repeat_penalty', 0.1)

    # ── Fome: multiplica (lê; não incrementa aqui) ─────────────────────────
    hunger = float(state.get('hunger', 0.0) or 0.0)
    reward *= (1.0 + CONFIG.get('hunger_gain', 0.5) * max(0.0, min(1.0, hunger)))

    # ── Anti-farniente (respeita idle_cost, hoje 0) ────────────────────────
    if CONFIG.get('idle_cost', 0) > 0 and state.get('prev_position') is not None:
        movement = (worm.head.position - state['prev_position']).length()
        if movement < CONFIG.get('idle_threshold', 0.05):
            reward -= CONFIG['idle_cost']

    state['prev_dist_food'] = dist_food
    state['prev_position'] = Vec3(worm.head.position)

    return max(-1.0, min(1.0, reward))


# ─── Nomes canônicos (sem prefixo food_) ─────────────────────────────────────
# Mantidos os dois estilos: código novo usa get_sensor_inputs/calculate_reward.
def get_sensor_inputs(worm, env, state) -> list:
    return get_food_sensor_inputs(worm, env, state)


def calculate_reward(worm, env, state, action=None) -> float:
    return calculate_food_reward(worm, env, state, action=action)
