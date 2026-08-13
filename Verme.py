from ursina import *
from Rede_Neural import WormBrain
import random
import math

app = Ursina()

# ─── ILUMINAÇÃO ───────────────────────────────────────────────────────────────
DirectionalLight(y=2, z=-1)
AmbientLight(color=Color(0.6, 0.6, 0.6, 1))  # Iluminação mais clara e vibrante

# Luz pontual que segue o verme 
worm_light = PointLight(parent=scene, color=color.azure, position=(0, 10, 0))

# ─── CHÃO COM TEXTURA APRIMORADA ─────────────────────────────────────────────────────────
Entity(
    model='plane',
    scale=40,
    texture='grass',  # Textura mais detalhada
    texture_scale=(40, 40),
    color=color.white,  # Cor mais clara para destacar o verme
)

# ─── LISTAS DE FONTES DINÂMICAS ───────────────────────────────────────────────
light_sources = []
rain_sources = []

# ─── MODO DE EDIÇÃO ───────────────────────────────────────────────────────────
editor = {'mode': 'none'}

# ─── PARTÍCULAS DE CHUVA ──────────────────────────────────────────────────────
rain_particles = []

def rebuild_rain_particles():
    for p in rain_particles:
        destroy(p)
    rain_particles.clear()

    for src in rain_sources:
        for _ in range(40):
            rain_particles.append(Entity(
                model='sphere',  # Partículas agora são esferas
                color=Color(0.5, 0.8, 1, 0.6),
                scale=0.1,
                position=(
                    src.x + random.uniform(-3, 3),
                    random.uniform(0, 8),
                    src.z + random.uniform(-3, 3),
                )
            ))

# ─── CRIAÇÃO INICIAL DAS FONTES ───────────────────────────────────────────────
def place_light(pos):
    src = Entity(
        model='sphere',
        color=color.yellow.tint(-0.2),
        scale=2,
        position=Vec3(pos.x, 1, pos.z),
        collider='box',
    )
    light_sources.append(src)
    # Efeito visual ao adicionar luz
    invoke(destroy, Entity(model='sphere', color=color.white, scale=3, position=src.position), delay=0.5)

def place_rain(pos):
    src = Entity(
        model='sphere',  # Substituímos o cubo por uma esfera
        color=color.cyan.tint(-0.2),  # Cor mais suave
        scale=2,
        position=Vec3(pos.x, 1, pos.z),
        collider='box',
    )
    rain_sources.append(src)
    rebuild_rain_particles()

# Fundo do cenário
sky = Entity(
    model='sphere',
    scale=500,
    double_sided=True,
    texture='sky_sunset',  # Textura de céu
    color=color.white.tint(-0.2),
)

def delete_source(entity):
    if entity in light_sources:
        light_sources.remove(entity)
        destroy(entity)
    elif entity in rain_sources:
        rain_sources.remove(entity)
        destroy(entity)
        rebuild_rain_particles()

place_light(Vec3(10, 1, 10))
place_rain(Vec3(-10, 1, -10))

# ─── PLANO INVISÍVEL DE POSICIONAMENTO ────────────────────────────────────────
ground_collider = Entity(
    model='plane',
    scale=40,
    collider='box',
    visible=False,
    y=0,
)

# ─── VERME COM DESIGN SOFISTICADO ─────────────────────────────────────────────
NUM_SEGMENTS = 12  # Aumentar o número de segmentos
SEGMENT_SIZE = 2.5  # Aumentar o tamanho dos segmentos
SEGMENT_GAP = 2.5  # Ajustar o espaçamento entre os segmentos
SPEED = 6.0  # Tornar o movimento mais rápido
ARRIVAL_RADIUS = 7.0  # Ajustar o raio de chegada

# Função para interpolar entre duas cores
def interpolate_color(color1, color2, t):
    t = max(0.0, min(1.0, t))  # Garantir que t esteja entre 0 e 1
    r = color1.r + (color2.r - color1.r) * t
    g = color1.g + (color2.g - color1.g) * t
    b = color1.b + (color2.b - color1.b) * t
    a = color1.a + (color2.a - color1.a) * t
    return Color(r, g, b, a)

# Gradiente de cores mais dramático (do preto ao azul brilhante)
segment_colors = [
    interpolate_color(color.black, color.cyan.tint(0.5), i / NUM_SEGMENTS) for i in range(NUM_SEGMENTS)
]

# Remova as texturas e use apenas cores sólidas
head = Entity(
    model='sphere',
    color=color.cyan.tint(-0.2),  # Cor sólida para teste
    scale=SEGMENT_SIZE * 1.1,
    position=(0, SEGMENT_SIZE / 2, 0),
    collider='sphere',
)

segments = [
    Entity(
        model='sphere',
        color=segment_colors[i],  # Gradiente de cores
        scale=SEGMENT_SIZE * (1.0 - i * 0.05),
        position=head.position - Vec3(0, 0, i * SEGMENT_GAP),
    )
    for i in range(NUM_SEGMENTS)
]
# Luz suave que segue o verme
worm_light = PointLight(
    parent=head,
    color=color.cyan,  # Luz azul brilhante
    position=(0, 5, 0),
    intensity=1.5,  # Aumentar a intensidade da luz
)

# ─── ANIMAÇÃO DO VERME ────────────────────────────────────────────────────────
# ─── ANIMAÇÃO DO VERME ────────────────────────────────────────────────────────
def animate_worm():
    for i, seg in enumerate(segments):
        # Animação de ondulação nos segmentos
        seg.scale = SEGMENT_SIZE * (1.0 - i * 0.05) * (1 + math.sin(time.time() * 5 + i) * 0.1)
        seg.color = segment_colors[i].tint(math.sin(time.time() * 3 + i) * 0.2)
        seg.rotation_y += math.sin(time.time() * 2 + i) * 5  # Leve rotação

# ─── CÉREBRO ──────────────────────────────────────────────────────────────────
brain = WormBrain()

# ─── CÂMERA ───────────────────────────────────────────────────────────────────
# Ajuste a posição inicial da câmera
cam_pivot = Entity()
cam_pivot.y = 10

camera.parent = cam_pivot
camera.position = (0, 20, -50)  # Ajuste a posição para visualizar o verme
camera.rotation = (20, 0, 0)

cam = {
    'rot_speed' : 40.0,
    'pan_speed' : 20.0,
    'zoom_speed': 5.0,
    'min_zoom'  : 5.0,
    'max_zoom'  : 80.0,
}

# ─── ESTADO DO VERME ──────────────────────────────────────────────────────────
state = {
    'direction'      : Vec3(0, 0, 1),
    'reward_timer'   : 0.0,
    'reward_interval': 1.0,
    'total_reward'   : 0.0,
    'rain_pulse'     : 0.0,
    'pulse_timer'    : 0.0,
}

# ─── FUNÇÕES DO VERME ─────────────────────────────────────────────────────────
# Histórico de posições da cabeça do verme
history = [Vec3(0, SEGMENT_SIZE / 2, 0)] * (NUM_SEGMENTS + 1) * 4
MAX_HISTORY_LENGTH = (NUM_SEGMENTS + 1) * 10  # Definir um tamanho máximo global

# ─── UPDATE ───────────────────────────────────────────────────────────────────
# Função para criar partículas ao redor do verme
def spawn_energy_particles():
    for _ in range(20):  # Criar 20 partículas
        Entity(
            model='sphere',
            color=color.cyan.tint(0.5),
            scale=0.2,
            position=head.position + Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)),
            add_to_scene_entities=False,
            lifetime=0.5,  # Partículas desaparecem após 0.5 segundos
        )

# Adicionar partículas no update
def update():
    spawn_energy_particles()
    update_camera()
    update_rain_particles()
    update_segments()
    update_rewards()

def update_camera():
    cam_pivot.position = lerp(cam_pivot.position, head.position, 0.1)
    worm_light.position = head.position + Vec3(0, 5, 0)

def update_rain_particles():
    for p in rain_particles:
        p.y -= time.dt * 3
        if p.y < 0:
            if rain_sources:
                src = min(
                    rain_sources,
                    key=lambda r: (Vec3(p.x, 0, p.z) - Vec3(r.x, 0, r.z)).length()
                )
                p.x = src.x + random.uniform(-3, 3)
                p.z = src.z + random.uniform(-3, 3)
            p.y = 8

def update_segments():
    history.insert(0, Vec3(head.position))
    while len(history) > MAX_HISTORY_LENGTH:
        history.pop()

    for i, seg in enumerate(segments):
        idx = min(int((i + 1) * SEGMENT_GAP * (SPEED / 4)), len(history) - 1)
        seg.position = history[idx]

# Função para criar partículas de feedback
def spawn_feedback_particles(reward):
    color_feedback = color.green if reward > 0 else color.red
    for _ in range(10):  # Criar 10 partículas
        Entity(
            model='sphere',
            color=color_feedback,
            scale=0.1,
            position=head.position + Vec3(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)),
            add_to_scene_entities=False,
            lifetime=0.5,  # Partículas desaparecem após 0.5 segundos
        )

def update_rewards():
    state['reward_timer'] += time.dt
    if state['reward_timer'] >= state['reward_interval']:
        state['reward_timer'] = 0.0
        reward = calculate_reward()
        brain.reinforce(reward=reward)
        state['total_reward'] += reward

        # Adicionar feedback visual
        spawn_feedback_particles(reward)
def reset():
    """Reinicia o verme e o cérebro sem fechar o programa."""
    if not history:
        print("Erro: 'history' não está inicializado.")
        return

    head.position = Vec3(0, SEGMENT_SIZE / 2, 0)
    for i, seg in enumerate(segments):
        seg.position = head.position - Vec3(0, 0, (i + 1) * SEGMENT_GAP)
    history.clear()
    for _ in range(MAX_HISTORY_LENGTH):
        history.append(Vec3(head.position))
    state.update({
        'direction': Vec3(0, 0, 1),
        'reward_timer': 0.0,
        'total_reward': 0.0,
        'rain_pulse': 0.0,
        'pulse_timer': 0.0,
    })
    brain.__init__()
    print("\n── Reiniciado ──────────────────────────────────\n")


def get_sensor_inputs():
    max_dist = 30.0

    # Fotorrecepção: fonte de luz mais próxima
    if light_sources:
        dist_light = min(
            (head.position - ls.position).length()
            for ls in light_sources
        )
        light_input = max(0.0, 1.0 - (dist_light / max_dist))
    else:
        light_input = 0.0  # Sem fontes de luz → sem estímulo luminoso

    # Tato/vibração: fonte de chuva mais próxima
    if rain_sources:
        dist_rain = min(
            (head.position - rs.position).length()
            for rs in rain_sources
        )
        rain_proximity = max(0.0, 1.0 - (dist_rain / max_dist))
        touch_input = min(1.0, rain_proximity + state['rain_pulse'] * 0.3)
    else:
        touch_input = 0.0

    return light_input, touch_input


def calculate_reward():
    max_dist = 30.0
    reward = 0.0

    # Penalidade pela luz mais próxima
    if light_sources:
        dist_light = min(
            (head.position - ls.position).length()
            for ls in light_sources
        )
        norm_light = min(dist_light / max_dist, 1.0)
        reward += norm_light  # Longe da luz = recompensa positiva
        if dist_light < ARRIVAL_RADIUS:
            reward -= 0.5  # Penalidade extra por estar no perigo

    # Recompensa pela chuva mais próxima
    if rain_sources:
        dist_rain = min(
            (head.position - rs.position).length()
            for rs in rain_sources
        )
        norm_rain = min(dist_rain / max_dist, 1.0)
        reward -= norm_rain  # Longe da chuva = recompensa negativa
        if dist_rain < ARRIVAL_RADIUS:
            reward += 0.5  # Bônus extra por estar no alvo

    # Garantir que a recompensa esteja no intervalo [-1.0, 1.0]
    return max(-1.0, min(1.0, reward))


def get_target_direction():
    """
    Calcula a direção ideal de movimento usando física simples.

    Esta função atua como o "professor" do sistema de aprendizado.
    Em vez de esperar que a rede neural descubra o comportamento correto
    por tentativa e erro puro (o que levaria muito tempo), ela demonstra
    a direção correta a cada frame. A rede aprende observando e recebendo
    reforço positivo quando segue este comportamento.

    Com múltiplas fontes:
    - A atração da chuva é o vetor médio para todas as fontes de chuva,
      ponderado pelo inverso da distância (fontes mais próximas puxam mais).
    - A repulsão da luz soma todos os vetores de fuga, ponderados pela
      proximidade (fontes mais próximas repelem mais forte).

    Este modelo é inspirado em campos de potencial (potential fields),
    uma técnica clássica de navegação em robótica.
    """
    combined = Vec3(0, 0, 0)

    # ── Atração pelas fontes de chuva ─────────────────────────────────────────
    nearest_rain     = None
    nearest_rain_dist = float('inf')

    for rs in rain_sources:
        to_rain  = rs.position - head.position
        dist     = to_rain.length()
        if dist < nearest_rain_dist:
            nearest_rain_dist = dist
            nearest_rain      = rs
        if dist > 0.01:
            # Peso inversamente proporcional à distância: fontes próximas
            # atraem mais do que fontes distantes
            weight    = max(0.0, 1.0 - (dist / 30.0))
            combined += to_rain.normalized() * weight

    # ── Órbita ao redor da chuva mais próxima ────────────────────────────────
    if nearest_rain and nearest_rain_dist < ARRIVAL_RADIUS:
        dir_to_rain = (nearest_rain.position - head.position)
        if dir_to_rain.length() > 0.01:
            dir_to_rain = dir_to_rain.normalized()
        tangent  = Vec3(-dir_to_rain.z, 0, dir_to_rain.x)
        combined = tangent * 0.7 + dir_to_rain * 0.3

    # ── Repulsão pelas fontes de luz ──────────────────────────────────────────
    for ls in light_sources:
        from_light = head.position - ls.position
        dist       = from_light.length()
        if dist > 0.01:
            weight    = max(0.0, 1.0 - (dist / 20.0))
            combined += from_light.normalized() * weight

    if Vec3(combined).length() > 0.01:
        return Vec3(combined).normalized()

    return state['direction']


# ─── UPDATE ───────────────────────────────────────────────────────────────────
def update():
    # Atualizar posição da câmera para seguir o verme
    cam_pivot.position = lerp(cam_pivot.position, head.position, 0.1)
    worm_light.position = head.position + Vec3(0, 5, 0)
    animate_worm()

    # ── Câmera: órbita ────────────────────────────────────────────────────────
    if mouse.right:
        cam_pivot.rotation_y += mouse.velocity[0] * cam['rot_speed']
        cam_pivot.rotation_x -= mouse.velocity[1] * cam['rot_speed']
        cam_pivot.rotation_x  = clamp(cam_pivot.rotation_x, -80, 80)

    # ── Câmera: pan ───────────────────────────────────────────────────────────
    if held_keys['w']: cam_pivot.y += cam['pan_speed'] * time.dt
    if held_keys['s']: cam_pivot.y -= cam['pan_speed'] * time.dt
    if held_keys['a']: cam_pivot.x -= cam['pan_speed'] * time.dt
    if held_keys['d']: cam_pivot.x += cam['pan_speed'] * time.dt

    # ── Pulso de chuva ────────────────────────────────────────────────────────
    state['pulse_timer'] += time.dt
    if state['pulse_timer'] > 0.3:
        state['pulse_timer'] = 0.0
        state['rain_pulse']  = 1.0
    else:
        state['rain_pulse'] *= (1 - time.dt * 8)

    # ── Partículas de chuva ───────────────────────────────────────────────────
    for p in rain_particles:
        p.y -= time.dt * 3
        if p.y < 0:
            # Encontra a rain_source mais próxima para reancorá-la
            if rain_sources:
                src = min(
                    rain_sources,
                    key=lambda r: (Vec3(p.x, 0, p.z) - Vec3(r.x, 0, r.z)).length()
                )
                p.x = src.x + random.uniform(-3, 3)
                p.z = src.z + random.uniform(-3, 3)
            p.y = 8

    # ── Sensores e rede neural ────────────────────────────────────────────────
    light_input, touch_input = get_sensor_inputs()
    brain.forward(light=light_input, touch=touch_input)

    # ── Direção: híbrido professor + rede neural (curriculum learning) ────────
    #
    # O parâmetro 'autonomy' (autonomia) vai de 0.0 a 1.0 e controla quanto
    # do movimento vem da rede neural vs do professor determinístico.
    #
    # Curriculum Learning: começar com exemplos fáceis (professor guiando) e
    # gradualmente aumentar a dificuldade (deixar a rede decidir sozinha) é
    # uma técnica comprovada tanto em RL quanto no aprendizado humano.
    # Um aluno que nunca viu um exemplo correto dificilmente aprende sozinho.
    #
    # A autonomia cresce com a recompensa acumulada — não com o tempo.
    # Isso é importante: usamos o desempenho real como critério de maturidade,
    # não simplesmente quantos segundos passaram.
    #
    # Fórmula: autonomy = clamp(total_reward / AUTONOMY_SCALE, 0, 1)
    # AUTONOMY_SCALE = quanto de recompensa acumulada representa "maturidade total"
    AUTONOMY_SCALE = 30.0
    autonomy = min(1.0, max(0.0, state['total_reward'] / AUTONOMY_SCALE))

    # Direção do professor (determinística, sempre correta)
    teacher_dir = get_target_direction()

    # Direção da rede neural (aprende com o tempo)
    light_input, touch_input = get_sensor_inputs()
    nx, ny, nz = brain.forward(light=light_input, touch=touch_input)
    neural_dir = Vec3(nx, 0, nz)
    if neural_dir.length() > 0.01:
        neural_dir = neural_dir.normalized()
    else:
        neural_dir = teacher_dir  # fallback se a rede der vetor zero

    # Interpolação entre professor e rede neural conforme a autonomia
    # autonomy=0.0 → 100% professor | autonomy=1.0 → 100% rede neural
    blended_dir = lerp(Vec3(teacher_dir), Vec3(neural_dir), autonomy)

    # Suavização final para evitar mudanças bruscas de direção
    smooth = lerp(Vec3(state['direction']), Vec3(blended_dir), 0.08)
    if Vec3(smooth).length() > 0.01:
        state['direction'] = Vec3(smooth).normalized()

    # ── Mover verme ───────────────────────────────────────────────────────────
    head.position += state['direction'] * SPEED * time.dt
    head.x = max(-18, min(18, head.x))
    head.z = max(-18, min(18, head.z))
    head.y = SEGMENT_SIZE / 2

    # ── Histórico e segmentos ─────────────────────────────────────────────────
    history.insert(0, Vec3(head.position))
    while len(history) > (NUM_SEGMENTS + 1) * 10:
        history.pop()

    for i, seg in enumerate(segments):
        idx = min(int((i + 1) * SEGMENT_GAP * (SPEED / 4)), len(history) - 1)
        seg.position = history[idx]

    if state['direction'].length() > 0:
        head.look_at(head.position + state['direction'])

    # ── Reforço ───────────────────────────────────────────────────────────────
    state['reward_timer'] += time.dt
    if state['reward_timer'] >= state['reward_interval']:
        state['reward_timer'] = 0.0
        reward = calculate_reward()
        brain.reinforce(reward=reward)
        state['total_reward'] += reward
        print(
            f"luz={light_input:.2f}  "
            f"toque={touch_input:.2f}  "
            f"recompensa={reward:+.2f}  "
            f"acumulada={state['total_reward']:+.1f}  "
            f"autonomia={autonomy*100:.0f}%"
        )


# ─── INPUT ────────────────────────────────────────────────────────────────────
def input(key):

    # ── Zoom ──────────────────────────────────────────────────────────────────
    if key == 'scroll up':
        camera.z = min(cam['min_zoom'] * -1, camera.z + cam['zoom_speed'])
    if key == 'scroll down':
        camera.z = max(cam['max_zoom'] * -1, camera.z - cam['zoom_speed'])

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
            # ── Modo deletar ──────────────────────────────────────────────────
            # mouse.hovered_entity retorna a entidade sob o cursor, se houver.
            # Verificamos se é uma fonte conhecida antes de destruir — nunca
            # devemos destruir o chão, o verme ou outras entidades da cena.
            target = mouse.hovered_entity
            if target and (target in light_sources or target in rain_sources):
                delete_source(target)

        else:
            # ── Modo colocar ──────────────────────────────────────────────────
            # raycast dispara um raio da câmera até o mouse e retorna o ponto
            # de interseção com o collider atingido (nosso ground_collider).
            # hit.world_point é o Vec3 da posição exata no mundo 3D.
            hit = raycast(
                camera.world_position,
                camera.forward,
                distance=200,
                ignore=[head] + segments,
            )
            if hit.hit:
                pos = hit.world_point
                if editor['mode'] == 'place_light':
                    place_light(pos)
                elif editor['mode'] == 'place_rain':
                    place_rain(pos)

    # ── Outras teclas ─────────────────────────────────────────────────────────
    if key == 'r':
        reset()

    if key == 'escape':
        # ESC cancela o modo ativo primeiro; segundo ESC fecha o programa
        if editor['mode'] != 'none':
            editor['mode'] = 'none'
            print("  Modo cancelado.")
        else:
            quit()


print("\n── Verme Neural ────────────────────────────────")
print("  Botão direito + mouse → orbitar câmera")
print("  WASD                  → mover foco da câmera")
print("  Scroll                → zoom")
print("  R                     → reiniciar verme e cérebro")
print("  ESC                   → cancelar modo / sair")
print("  ── Editor ──────────────────────────────────")
print("  [1] + clique esquerdo → colocar fonte de LUZ")
print("  [2] + clique esquerdo → colocar fonte de CHUVA")
print("  [3] + clique esquerdo → deletar bloco")
print("  (pressione a tecla de modo novamente para cancelar)")
print("────────────────────────────────────────────────\n")

app.run()