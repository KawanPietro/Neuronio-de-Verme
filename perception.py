from ursina import Vec3

from config import CONFIG


# ─── AUXILIARES ───────────────────────────────────────────────────────────────

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


# ─── ESTADO (SENSORES 15-DIM, Fase 12 enxuta) ───────────────────────────────────
#
# O verme "vê" geometria + obstáculos + propriocepção. Isso dá à rede a
# informação de EM QUE DIREÇÃO ir, O QUE desviar e PARA ONDE ESTÁ INDO —
# sem isso, navegar com barreiras é impossível (o verme não sabia "para onde
# estava indo", só "para onde está a chuva").
#
# Revisão anti-overfit (unificado 1+2): ângulos 15-16 removidos — redundantes
# com dir+vel, mascarar no_ang deu 43.3% vs 26.7% full17. Mantidos vel+borda.
#
#   índice | feature       | significado
#   -------+---------------+--------------------------------------------
#   0      | luz_dir.x     | vetor XZ unitário até a luz mais próxima
#   1      | luz_dir.z     | (0, 0) se não houver fonte de luz
#   2      | luz_dist      | distância normalizada [0,1] até essa luz
#   3      | luz_perigo    | 1 se dentro do ARRIVAL_RADIUS, senão 0
#   4      | chuva_dir.x   | vetor XZ unitário até a chuva mais próxima
#   5      | chuva_dir.z   | (0, 0) se não houver fonte de chuva
#   6      | chuva_dist    | distância normalizada [0,1] até essa chuva
#   7      | pulso_chuva   | vibração (sinal de "tato"), em [0,1]
#   8      | obs_dir.x     | vetor XZ unitário até o obstáculo mais próximo
#   9      | obs_dir.z     | (0, 0) se não houver obstáculo
#  10      | obs_dist      | distância normalizada [0,1] até esse obstáculo
#  11      | vel_x         | direção atual X (propriocepção, [-1,1])
#  12      | vel_z         | direção atual Z (propriocepção, [-1,1])
#  13      | borda_x       | distância à borda X normalizada [0,1] (1=centro, 0=borda)
#  14      | borda_z       | distância à borda Z normalizada [0,1] (1=centro, 0=borda)

def _wrap_pi(a):
    """Normaliza um ângulo para [-pi, pi]."""
    import math
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def get_sensor_inputs(worm, env, state) -> list:
    """Devolve o estado com geometria: 15 features (direções em [-1,1], resto em [0,1])."""
    max_dist = CONFIG['sensor_max_dist']
    radius   = CONFIG['arrival_radius']
    limit    = CONFIG.get('map_limit', 32)

    features = [0.0] * 15

    # ── Direção atual (propriocepção — fallback para vermes falsos de teste) ──
    direction = getattr(worm, 'direction', None)
    if direction is None:
        dir_x, dir_z = 0.0, 1.0
    else:
        dir_x, dir_z = direction.x, direction.z

    # ── Luz ───────────────────────────────────────────────────────────────────
    light_src, dist_light = _nearest(worm, env.light_sources)
    to_light = None
    if light_src is not None:
        to_light = light_src.position - worm.head.position
        dx, dz = _xz_unit(to_light)
        features[0] = dx
        features[1] = dz
        features[2] = min(dist_light / max_dist, 1.0)
        features[3] = 1.0 if dist_light < radius else 0.0

    # ── Chuva ─────────────────────────────────────────────────────────────────
    rain_src, dist_rain = _nearest(worm, env.rain_sources)
    to_rain = None
    if rain_src is not None:
        to_rain = rain_src.position - worm.head.position
        dx, dz = _xz_unit(to_rain)
        features[4] = dx
        features[5] = dz
        features[6] = min(dist_rain / max_dist, 1.0)

    # ── Pulso de chuva (vibração) ─────────────────────────────────────────────
    features[7] = state.get('rain_pulse', 0.0)

    # ── Obstáculo mais próximo (Fase 15) ─────────────────────────────────────
    obstacles = getattr(env, 'obstacles', [])
    obs_src, dist_obs = _nearest(worm, obstacles)
    if obs_src is not None:
        to_obs = obs_src.position - worm.head.position
        dx, dz = _xz_unit(to_obs)
        features[8] = dx
        features[9] = dz
        features[10] = min(dist_obs / max_dist, 1.0)

    # ── Fase 12: velocidade / propriocepção ───────────────────────────────────
    features[11] = max(-1.0, min(1.0, dir_x))
    features[12] = max(-1.0, min(1.0, dir_z))

    # ── Fase 12: distância às bordas (1=centro seguro, 0=na parede) ───────────
    px = worm.head.position.x
    pz = worm.head.position.z
    features[13] = max(0.0, min(1.0, (limit - abs(px)) / limit))
    features[14] = max(0.0, min(1.0, (limit - abs(pz)) / limit))

    return features


# ─── RECOMPENSA POR PROGRESSO ─────────────────────────────────────────────────
#
# Em vez de premiar apenas posições absolutas (o que permite "andar em círculo"),
# a recompensa agora diz se o verme MELHOROU desde o passo anterior:
#
#   r = + progresso em direção à chuva   (Δ distância curvada)
#       − progresso em direção à luz      (aproximar da luz é ruim)
#       + BÔNUS ao entrar no raio da chuva
#       − PENALIDADE ao entrar na zona de perigo da luz
#       − pequeno CUSTO por ficar parado  (anti-farniente)
#
# Esta função TEM efeito colateral: atualiza os campos prev_* do `state`,
# que são a memória de um passo para o cálculo do Δ do próximo.

def calculate_reward(worm, env, state) -> float:
    """Recompensa por progresso (reward shaping), calculada por passo (frame).

    Fase 7: adiciona potencial de proximidade (1/dist) para puxar o verme
    para perto da chuva e empurrá-lo da luz — gera gradiente mesmo quando
    o verme está longe e não há progresso mensurável.
    """
    max_dist = CONFIG['sensor_max_dist']
    radius   = CONFIG['arrival_radius']

    reward = 0.0

    _, dist_rain  = _nearest(worm, env.rain_sources)
    _, dist_light = _nearest(worm, env.light_sources)
    _, dist_obs   = _nearest(worm, getattr(env, 'obstacles', []))

    # ── Potencial de proximidade (Fase 7): sinal contínuo em 1/dist ──────────
    # Quando o verme está longe da chuva, ainda assim recebe um "puxão" suave.
    # Quando está perto, o bônus cresce — cria um campo de atração/repulsão.
    if dist_rain is not None and dist_rain > 0.1:
        reward += CONFIG['proximity_rain_bonus'] / dist_rain
    if dist_light is not None and dist_light > 0.1:
        reward -= CONFIG['proximity_light_penalty'] / dist_light

    # ── Progresso em direção à chuva (aproximar = recompensa) ─────────────────
    if dist_rain is not None and state['prev_dist_rain'] is not None:
        progress_rain = state['prev_dist_rain'] - dist_rain
        reward += CONFIG['progress_scale'] * progress_rain

    # ── Progresso em direção à luz (aproximar da luz = ruim) ─────────────────
    if dist_light is not None and state['prev_dist_light'] is not None:
        progress_light = state['prev_dist_light'] - dist_light
        reward -= CONFIG['progress_scale'] * progress_light

    # ── Eventos de chegada / perigo ──────────────────────────────────────────
    if dist_rain is not None:
        prev = state['prev_dist_rain']
        if prev is not None and dist_rain < radius <= prev:
            reward += CONFIG['arrival_bonus']          # acabou de chegar na chuva
        if dist_rain < radius:
            reward += CONFIG['inside_rain_reward']     # permanece na chuva

    if dist_light is not None:
        prev = state['prev_dist_light']
        if prev is not None and dist_light < radius <= prev:
            reward -= CONFIG['danger_penalty']         # acabou de entrar no perigo
        if dist_light < radius:
            reward -= CONFIG['inside_light_penalty']   # permanece no perigo

    # ── Obstáculos (Fase 15): colisão, proximidade e progresso ───────────────
    obs_radius = CONFIG['obstacle_radius']
    if dist_obs is not None:
        # Penalidade contínua por estar perto do obstáculo
        if dist_obs > 0.1:
            reward -= CONFIG['obstacle_proximity_penalty'] / max(dist_obs, 0.5)
        # Colisão direta
        prev_obs = state.get('prev_dist_obs', None)
        if prev_obs is not None and dist_obs < obs_radius <= prev_obs:
            reward -= CONFIG['obstacle_penalty']  # acabou de bater
        if dist_obs < obs_radius:
            reward -= CONFIG['obstacle_penalty'] * 0.3  # permanecer colado
        # Progresso: afastar-se do obstáculo é bom
        if prev_obs is not None and dist_obs is not None:
            reward += CONFIG['obstacle_avoid_reward'] * (dist_obs - prev_obs)

    # ── Anti-farniente (desativado na Fase 7) ────────────────────────────────
    if CONFIG['idle_cost'] > 0 and state['prev_position'] is not None:
        movement = (worm.head.position - state['prev_position']).length()
        if movement < CONFIG['idle_threshold']:
            reward -= CONFIG['idle_cost']

    # ── Guarda o estado para o próximo passo ─────────────────────────────────
    state['prev_dist_rain']  = dist_rain
    state['prev_dist_light'] = dist_light
    state['prev_dist_obs']   = dist_obs
    state['prev_position']   = Vec3(worm.head.position)

    return max(-1.0, min(1.0, reward))


# ─── MODO ALIMENTO 11-DIM (reformulação E1-E4, aditivo) ───────────────────────
#
# Legado 17-dim acima permanece INTACTO até T4 migrar main.py — isso evita
# quebrar treino atual, debug_sensores.py e pesos existentes.
#
#   índice | feature     | significado
#   -------+-------------+--------------------------------------------
#   0      | food_dir.x  | vetor XZ unitário até o alimento mais próximo
#   1      | food_dir.z  | (0, 0) se não houver alimento
#   2      | food_dist   | distância normalizada [0,1]
#   3      | smell       | olfato 1/(1+dist) em [0,1] (E3, denso mesmo longe)
#   4      | obs_dir.x   | vetor XZ unitário até o obstáculo mais próximo
#   5      | obs_dir.z   | (0, 0) se não houver obstáculo
#   6      | obs_dist    | distância normalizada [0,1]
#   7      | vel_x       | direção atual X (propriocepção, [-1,1])
#   8      | vel_z       | direção atual Z (propriocepção, [-1,1])
#   9      | borda_x     | distância à borda X [0,1] (1=centro, 0=parede)
#   10     | borda_z     | distância à borda Z [0,1]
#
# Armadilhas evitadas:
# - usa state.get() em tudo novo (main.py legado não tem 'hunger'/'prev_dist_food')
# - NÃO incrementa fome aqui (efeito colateral seria invisível); T4 incrementa
#   no main e zera ao comer. Aqui só LÊ hunger como multiplicador.

FOOD_DIM = 11


def _food_sources(env):
    """Fontes de alimento; fallback p/ rain_sources na transição (testes)."""
    foods = list(getattr(env, 'food_sources', []) or [])
    if not foods:
        # Compat transição: permite testar modo alimento sem load_maze
        foods = list(getattr(env, 'rain_sources', []) or [])
    return foods


def get_food_sensor_inputs(worm, env, state) -> list:
    """Estado 11-dim do modo alimento (todos normalizados)."""
    max_dist = CONFIG['sensor_max_dist']
    limit = CONFIG.get('map_limit', 32)
    features = [0.0] * FOOD_DIM

    direction = getattr(worm, 'direction', None)
    if direction is None:
        dir_x, dir_z = 0.0, 1.0
    else:
        dir_x, dir_z = direction.x, direction.z

    # ── Alimento + olfato ──────────────────────────────────────────────────
    food_src, dist_food = _nearest(worm, _food_sources(env))
    if food_src is not None:
        to_food = food_src.position - worm.head.position
        dx, dz = _xz_unit(to_food)
        features[0] = dx
        features[1] = dz
        features[2] = min(dist_food / max_dist, 1.0)
        features[3] = 1.0 / (1.0 + dist_food)  # E3: denso, [0,1]

    # ── Obstáculo (mesmo contrato Fase 15) ─────────────────────────────────
    obs_src, dist_obs = _nearest(worm, getattr(env, 'obstacles', []))
    if obs_src is not None:
        to_obs = obs_src.position - worm.head.position
        dx, dz = _xz_unit(to_obs)
        features[4] = dx
        features[5] = dz
        features[6] = min(dist_obs / max_dist, 1.0)

    # ── Propriocepção + bordas ─────────────────────────────────────────────
    features[7] = max(-1.0, min(1.0, dir_x))
    features[8] = max(-1.0, min(1.0, dir_z))
    px = worm.head.position.x
    pz = worm.head.position.z
    features[9] = max(0.0, min(1.0, (limit - abs(px)) / limit))
    features[10] = max(0.0, min(1.0, (limit - abs(pz)) / limit))

    return features


def calculate_food_reward(worm, env, state) -> float:
    """Recompensa unimodal do alimento (E1), por passo, clip [-1,1].

    r = progresso + smell + eventos comer/permanecer (+ obstáculos) × (1+fome).
    Atualiza APENAS prev_dist_food/prev_dist_obs/prev_position (legado separado).
    """
    radius = CONFIG.get('food_radius', CONFIG.get('arrival_radius', 7.0))
    reward = 0.0

    _, dist_food = _nearest(worm, _food_sources(env))
    _, dist_obs = _nearest(worm, getattr(env, 'obstacles', []))

    # ── Olfato denso (sinal mesmo longe, sem progresso mensurável) ──────────
    if dist_food is not None and dist_food > 0.05:
        reward += CONFIG.get('smell_bonus', 0.3) / (1.0 + dist_food)

    # ── Progresso (aproximar = bom; sem termo negativo de luz) ──────────────
    prev_food = state.get('prev_dist_food', None)
    if dist_food is not None and prev_food is not None:
        reward += CONFIG.get('progress_scale', 3.0) * (prev_food - dist_food)

    # ── Eventos comer / permanecer ──────────────────────────────────────────
    if dist_food is not None:
        if prev_food is not None and dist_food < radius <= prev_food:
            reward += CONFIG.get('eat_bonus', 5.0)
        if dist_food < radius:
            reward += CONFIG.get('inside_food_reward', 0.5)

    # ── Obstáculos (idêntico ao legado: colisão + proximidade + progresso) ──
    obs_radius = CONFIG.get('obstacle_radius', 2.5)
    if dist_obs is not None:
        if dist_obs > 0.1:
            reward -= CONFIG.get('obstacle_proximity_penalty', 0.2) / max(dist_obs, 0.5)
        prev_obs = state.get('prev_dist_obs', None)
        if prev_obs is not None and dist_obs < obs_radius <= prev_obs:
            reward -= CONFIG.get('obstacle_penalty', 3.0)
        if dist_obs < obs_radius:
            reward -= CONFIG.get('obstacle_penalty', 3.0) * 0.3
        if prev_obs is not None:
            reward += CONFIG.get('obstacle_avoid_reward', 0.18) * (dist_obs - prev_obs)

    # ── Fome E4: multiplica (lê; não incrementa aqui) ────────────────────────
    hunger = float(state.get('hunger', 0.0) or 0.0)
    reward *= (1.0 + CONFIG.get('hunger_gain', 0.5) * max(0.0, min(1.0, hunger)))

    # ── Anti-farniente (respeita idle_cost legado, hoje 0) ──────────────────
    if CONFIG.get('idle_cost', 0) > 0 and state.get('prev_position') is not None:
        movement = (worm.head.position - state['prev_position']).length()
        if movement < CONFIG.get('idle_threshold', 0.05):
            reward -= CONFIG['idle_cost']

    state['prev_dist_food'] = dist_food
    state['prev_dist_obs'] = dist_obs
    state['prev_position'] = Vec3(worm.head.position)

    return max(-1.0, min(1.0, reward))