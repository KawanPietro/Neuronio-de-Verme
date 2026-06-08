from ursina import *
from Rede_Neural import WormBrain
import random
import math

app = Ursina()

# ─── ILUMINAÇÃO ───────────────────────────────────────────────────────────────
DirectionalLight(y=2, z=-1)
AmbientLight(color=Color(0.4, 0.4, 0.4, 1))

# ─── CHÃO ─────────────────────────────────────────────────────────────────────
Entity(model='plane', scale=40, color=color.dark_gray)

# ─── LISTAS DE FONTES DINÂMICAS ───────────────────────────────────────────────
# Em vez de variáveis fixas (light_source, rain_source), usamos LISTAS.
#
# Por quê listas?
# Com variáveis fixas, só poderíamos ter exatamente 1 luz e 1 chuva.
# Com listas, podemos ter 0, 1, ou muitas fontes de cada tipo — e adicionar
# ou remover em tempo de execução sem mudar mais nada no código.
#
# Este é um padrão fundamental em programação de jogos chamado
# "Entity Management": entidades são criadas e destruídas dinamicamente,
# e o sistema de comportamento itera sobre a lista atual a cada frame.
light_sources = []   # lista de cubos amarelos (luz → verme foge)
rain_sources  = []   # lista de cubos ciano   (chuva → verme persegue)

# ─── MODO DE EDIÇÃO ───────────────────────────────────────────────────────────
# O editor_mode controla o que o clique do mouse faz.
# 'none'  → nenhuma ação de edição ativa (modo observação)
# 'place_light' → próximo clique coloca uma fonte de luz
# 'place_rain'  → próximo clique coloca uma fonte de chuva
# 'delete'      → próximo clique remove o bloco clicado
#
# Separar modos em vez de usar múltiplas teclas simultâneas evita
# conflitos de input e torna o sistema extensível: adicionar um novo
# modo é só adicionar uma nova string e tratar no input().
editor = {
    'mode': 'none',
}

# ─── PARTÍCULAS DE CHUVA ──────────────────────────────────────────────────────
# As partículas são visuais — não têm lógica de jogo.
# Elas são gerenciadas separadamente das rain_sources porque seu número
# e posição dependem de quantas fontes existem na cena.
rain_particles = []

def rebuild_rain_particles():
    """
    Reconstrói todas as partículas de chuva com base nas rain_sources atuais.

    Esta função é chamada sempre que uma rain_source é criada ou destruída.
    Destruir e recriar todas as partículas é mais simples do que tentar
    rastrear quais partículas pertencem a qual fonte — e para 40 partículas,
    o custo de performance é desprezível.
    """
    # Destrói todas as partículas existentes
    for p in rain_particles:
        destroy(p)
    rain_particles.clear()

    # Recria uma nuvem de partículas ao redor de cada rain_source
    for src in rain_sources:
        for _ in range(40):
            rain_particles.append(Entity(
                model='cube',
                color=Color(0.5, 0.8, 1, 0.6),
                scale=0.1,
                position=(
                    src.x + random.uniform(-3, 3),
                    random.uniform(0, 8),
                    src.z + random.uniform(-3, 3),
                )
            ))


# ─── CRIAÇÃO INICIAL DAS FONTES ───────────────────────────────────────────────
# Criamos as fontes iniciais usando as mesmas funções que o editor usará.
# Isso garante que o estado inicial é idêntico a qualquer estado criado
# manualmente — sem código duplicado.

def place_light(pos):
    """
    Cria uma nova fonte de luz na posição dada e a adiciona à lista.

    O collider='box' é necessário para que o mouse.hovered funcione,
    permitindo que o modo 'delete' detecte cliques sobre o bloco.
    """
    src = Entity(
        model='cube',
        color=color.yellow,
        scale=2,
        position=Vec3(pos.x, 1, pos.z),
        collider='box',
    )
    light_sources.append(src)

def place_rain(pos):
    """
    Cria uma nova fonte de chuva na posição dada, adiciona à lista
    e reconstrói as partículas para incluir a nova fonte.
    """
    src = Entity(
        model='cube',
        color=color.cyan,
        scale=2,
        position=Vec3(pos.x, 1, pos.z),
        collider='box',
    )
    rain_sources.append(src)
    rebuild_rain_particles()

def delete_source(entity):
    """
    Remove uma fonte de luz ou chuva da cena e da lista correspondente.

    destroy() remove a entidade do Ursina (libera memória e para de renderizar).
    list.remove() remove a referência Python à entidade.
    Se não fizermos os dois, ou a entidade continua visível (sem destroy),
    ou o Python tenta acessar um objeto destruído (sem remove) → crash.
    """
    if entity in light_sources:
        light_sources.remove(entity)
        destroy(entity)
    elif entity in rain_sources:
        rain_sources.remove(entity)
        destroy(entity)
        rebuild_rain_particles()

# Fontes iniciais
place_light(Vec3(10, 1, 10))
place_rain(Vec3(-10, 1, -10))

# ─── PLANO INVISÍVEL DE POSICIONAMENTO ────────────────────────────────────────
# Quando o usuário clica para colocar um bloco, precisamos saber ONDE no chão
# o mouse está apontando. O Ursina faz isso com raycast — um raio invisível
# disparado da câmera na direção do mouse.
#
# Para o raycast funcionar, o chão precisa ter um collider.
# Criamos um plano invisível só para isso (o plano visual não tem collider).
ground_collider = Entity(
    model='plane',
    scale=40,
    collider='box',
    visible=False,
    y=0,
)

# ─── VERME ────────────────────────────────────────────────────────────────────
NUM_SEGMENTS   = 8
SEGMENT_SIZE   = 1.8
SEGMENT_GAP    = 2.0
SPEED          = 4.0
ARRIVAL_RADIUS = 5.0

history = [Vec3(0, SEGMENT_SIZE / 2, 0)] * (NUM_SEGMENTS + 1) * 4

head = Entity(
    model='sphere',
    color=color.lime,
    scale=SEGMENT_SIZE * 1.2,
    position=(0, SEGMENT_SIZE / 2, 0),
    collider='sphere',
)

segments = [
    Entity(
        model='sphere',
        color=color.lime.tint(i * -0.06),
        scale=SEGMENT_SIZE * (1.0 - i * 0.04),
        position=head.position - Vec3(0, 0, i * SEGMENT_GAP),
    )
    for i in range(1, NUM_SEGMENTS + 1)
]

# ─── CÉREBRO ──────────────────────────────────────────────────────────────────
brain = WormBrain()

# ─── CÂMERA ───────────────────────────────────────────────────────────────────
cam_pivot = Entity()
cam_pivot.y = 10

camera.parent   = cam_pivot
camera.position = (0, 0, -50)
camera.rotation = (10, 0, 0)

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

def reset():
    """Reinicia o verme e o cérebro sem fechar o programa."""
    head.position = Vec3(0, SEGMENT_SIZE / 2, 0)
    for i, seg in enumerate(segments):
        seg.position = head.position - Vec3(0, 0, (i + 1) * SEGMENT_GAP)
    history.clear()
    for _ in range((NUM_SEGMENTS + 1) * 4):
        history.append(Vec3(head.position))
    state['direction']    = Vec3(0, 0, 1)
    state['reward_timer'] = 0.0
    state['total_reward'] = 0.0
    state['rain_pulse']   = 0.0
    state['pulse_timer']  = 0.0
    brain.__init__()
    print("\n── Reiniciado ──────────────────────────────────\n")


def get_sensor_inputs():
    """
    Calcula os dois sinais sensoriais do verme: luz e tato.

    O verme não "vê" a cena como nós vemos — ele só recebe dois números.
    Esta função é a ponte entre o mundo 3D rico do Ursina e o mundo
    simplificado de 2 dimensões que o cérebro do verme consegue processar.

    Quando há múltiplas fontes, usamos a fonte MAIS PRÓXIMA de cada tipo.
    Biologicamente, isso simula um receptor que responde ao estímulo
    mais intenso — análogo a como fotorreceptores reais funcionam.
    """
    max_dist = 30.0

    # ── Fotorrecepcao: fonte de luz mais próxima ──────────────────────────────
    if light_sources:
        dist_light = min(
            (head.position - ls.position).length()
            for ls in light_sources
        )
        light_input = max(0.0, 1.0 - (dist_light / max_dist))
    else:
        # Sem fontes de luz → sem estimulo luminoso
        light_input = 0.0

    # ── Tato/vibracao: fonte de chuva mais próxima ────────────────────────────
    if rain_sources:
        dist_rain = min(
            (head.position - rs.position).length()
            for rs in rain_sources
        )
        rain_proximity = max(0.0, 1.0 - (dist_rain / max_dist))
        touch_input    = min(1.0, rain_proximity + state['rain_pulse'] * 0.3)
    else:
        touch_input = 0.0

    return light_input, touch_input


def calculate_reward():
    """
    Define numericamente o que é "bom" e o que é "ruim" para o verme.

    Esta é a função mais importante do aprendizado por reforço.
    A rede neural não sabe intrinsecamente o que deve fazer —
    ela só sabe que deve maximizar a recompensa ao longo do tempo.
    Logo, a qualidade do aprendizado depende diretamente da qualidade
    desta função. Uma recompensa mal definida gera comportamento errado,
    mesmo que a rede seja perfeita.

    Com múltiplas fontes, recompensamos com base na fonte mais favorável:
    - Luz mais distante (melhor caso de fuga)
    - Chuva mais próxima (melhor caso de aproximação)
    """
    max_dist = 30.0
    reward   = 0.0

    # Penalidade pela luz mais próxima
    if light_sources:
        dist_light = min(
            (head.position - ls.position).length()
            for ls in light_sources
        )
        norm_light = min(dist_light / max_dist, 1.0)
        reward    += norm_light  # longe da luz = recompensa positiva
        if dist_light < ARRIVAL_RADIUS:
            reward -= 0.5        # penalidade extra por estar no perigo

    # Recompensa pela chuva mais próxima
    if rain_sources:
        dist_rain = min(
            (head.position - rs.position).length()
            for rs in rain_sources
        )
        norm_rain  = min(dist_rain / max_dist, 1.0)
        reward    -= norm_rain   # longe da chuva = recompensa negativa
        if dist_rain < ARRIVAL_RADIUS:
            reward += 0.5        # bônus extra por estar no alvo

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