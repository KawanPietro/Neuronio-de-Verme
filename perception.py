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


# ─── ESTADO (SENSORES 8-DIM) ──────────────────────────────────────────────────
#
# O verme agora "vê" geometria, não só proximidade. Isso dá à rede a informação
# de EM QUE DIREÇÃO ir — sem isso, aprender a navegar é praticamente impossível.
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

def get_sensor_inputs(worm, env, state) -> list:
    """Devolve o estado com geometria: 8 features em [0,1] (direções em [-1,1])."""
    max_dist = CONFIG['sensor_max_dist']
    radius   = CONFIG['arrival_radius']

    features = [0.0] * 8

    # ── Luz ───────────────────────────────────────────────────────────────────
    light_src, dist_light = _nearest(worm, env.light_sources)
    if light_src is not None:
        to_light = light_src.position - worm.head.position
        dx, dz = _xz_unit(to_light)
        features[0] = dx
        features[1] = dz
        features[2] = min(dist_light / max_dist, 1.0)
        features[3] = 1.0 if dist_light < radius else 0.0

    # ── Chuva ─────────────────────────────────────────────────────────────────
    rain_src, dist_rain = _nearest(worm, env.rain_sources)
    if rain_src is not None:
        to_rain = rain_src.position - worm.head.position
        dx, dz = _xz_unit(to_rain)
        features[4] = dx
        features[5] = dz
        features[6] = min(dist_rain / max_dist, 1.0)

    # ── Pulso de chuva (vibração) ─────────────────────────────────────────────
    features[7] = state['rain_pulse']

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

    # ── Anti-farniente (desativado na Fase 7) ────────────────────────────────
    if CONFIG['idle_cost'] > 0 and state['prev_position'] is not None:
        movement = (worm.head.position - state['prev_position']).length()
        if movement < CONFIG['idle_threshold']:
            reward -= CONFIG['idle_cost']

    # ── Guarda o estado para o próximo passo ─────────────────────────────────
    state['prev_dist_rain']  = dist_rain
    state['prev_dist_light'] = dist_light
    state['prev_position']   = Vec3(worm.head.position)

    return max(-1.0, min(1.0, reward))