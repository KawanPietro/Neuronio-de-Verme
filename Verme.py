from ursina import *

app = Ursina()

# ─── CÂMERA ───────────────────────────────────────────────────────────────────
# Removemos o EditorCamera — ele sobrescreve o position que definimos.
# Posicionamos a câmera diretamente e apontamos para a origem da cena.
camera.position = (0, 20, -25)
camera.look_at(Vec3(0, 0, 0))

# ─── ILUMINAÇÃO ───────────────────────────────────────────────────────────────
# shadows=True pode causar tela cinza em algumas GPUs/drivers.
# Removemos o parâmetro por enquanto.
DirectionalLight(y=2, z=-1)
AmbientLight(color=Color(0.4, 0.4, 0.4, 1))

# ─── CHÃO ─────────────────────────────────────────────────────────────────────
# Um plano simples para ter referência visual de posição e movimento.
Entity(
    model='plane',
    scale=40,
    color=color.dark_gray,
    texture='white_cube',
    texture_scale=(20, 20),
)

# ─── CONFIGURAÇÃO DO VERME ────────────────────────────────────────────────────
NUM_SEGMENTS  = 15     # quantos segmentos de corpo (além da cabeça)
SEGMENT_SIZE  = 1.2   # tamanho de cada segmento em unidades do Ursina
SEGMENT_GAP   = 1.6   # distância entre o centro de cada segmento
SPEED         = 5.0   # unidades por segundo

# ─── HISTÓRICO DE POSIÇÕES ────────────────────────────────────────────────────
# Esta lista é o coração do sistema de movimento do verme.
#
# A ideia: a cada frame, salvamos a posição atual da cabeça no início da lista.
# Cada segmento do corpo lê uma posição diferente desta lista — o segmento 1
# lê a posição de index 1 (onde a cabeça estava 1 passo atrás), o segmento 2
# lê o index 2, e assim por diante.
#
# Resultado: o corpo "escorrega" pelos rastros da cabeça, exatamente como
# um verme real se move. Este padrão tem um nome em programação de jogos:
# "follow the leader" (seguir o líder).
#
# Pré-populamos a lista com Vec3(0,0,0) para que os segmentos tenham uma
# posição válida desde o primeiro frame, antes de qualquer movimento.
history = [Vec3(0, SEGMENT_SIZE / 2, 0)] * (NUM_SEGMENTS + 1) * 4

# ─── CRIAÇÃO DA CABEÇA ────────────────────────────────────────────────────────
# A cabeça é uma esfera levemente maior que o corpo.
# Ela é o único elemento que recebe input direto do teclado.
head = Entity(
    model='sphere',
    color=color.lime,
    scale=SEGMENT_SIZE * 1.2,
    position=(0, SEGMENT_SIZE / 2, 0),
    collider='sphere',
)

# ─── CRIAÇÃO DOS SEGMENTOS DE CORPO ───────────────────────────────────────────
# Guardamos todos os segmentos numa lista para acessar depois no update().
# list comprehension: forma compacta de criar N objetos iguais em Python.
segments = [
    Entity(
        model='sphere',
        # Degradê do verde (frente) para o amarelo-escuro (cauda)
        # lerp de cor: quanto mais longe da cabeça, mais amarelado
        color=color.lime.tint(i * -0.06),
        scale=SEGMENT_SIZE * (1.0 - i * 0.02),  # afina levemente em direção à cauda
        position=head.position - Vec3(0, 0, i * SEGMENT_GAP),
    )
    for i in range(1, NUM_SEGMENTS + 1)
]

# ─── ESTADO DO MOVIMENTO ──────────────────────────────────────────────────────
# Separamos o estado em um dicionário, igual fizemos no ciclo dia/noite.
# 'direction' guarda para onde o verme está apontando (vetor normalizado).
# Começa olhando para frente (eixo Z positivo).
worm = {
    'direction': Vec3(0, 0, 1),
}


# ─── FUNÇÃO AUXILIAR: ler o input e calcular a direção ───────────────────────
# Separar a leitura de input do movimento em si é boa prática:
# futuramente você substituirá esta função pela saída da rede neural,
# sem precisar tocar na lógica de movimento.
def get_direction():
    """Lê o teclado e retorna o vetor de direção desejado (ou None)."""

    # held_keys['tecla'] retorna True enquanto a tecla estiver pressionada.
    # Diferente de input(), que dispara apenas uma vez ao pressionar.

    # ── Movimento horizontal (plano XZ) ──────────────────────────────────────
    if held_keys['up arrow'] or held_keys['w']:
        return Vec3(0, 0, 1)     # frente
    if held_keys['down arrow'] or held_keys['s']:
        return Vec3(0, 0, -1)    # trás
    if held_keys['left arrow'] or held_keys['a']:
        return Vec3(-1, 0, 0)    # esquerda
    if held_keys['right arrow'] or held_keys['d']:
        return Vec3(1, 0, 0)     # direita

    # ── Movimento vertical (eixo Y) ──────────────────────────────────────────
    # Q e E permitem subir e descer — útil para testar o espaço 3D completo.
    if held_keys['q']:
        return Vec3(0, 1, 0)     # sobe
    if held_keys['e']:
        return Vec3(0, -1, 0)    # desce

    return None  # nenhuma tecla pressionada


# ─── UPDATE: loop principal ───────────────────────────────────────────────────
def update():

    # ── 1. Obter direção do teclado ───────────────────────────────────────────
    new_dir = get_direction()

    if new_dir:
        # Atualizamos a direção apenas se uma tecla foi pressionada.
        # Isso mantém o verme parado ao soltar as teclas, em vez de continuar
        # deslizando — comportamento mais fácil de controlar.
        worm['direction'] = new_dir

        # ── 2. Mover a cabeça ────────────────────────────────────────────────
        # A fórmula clássica de movimento uniforme:
        #   nova_posicao = posicao_atual + direcao * velocidade * tempo_do_frame
        # Multiplicar por time.dt garante que a velocidade seja a mesma
        # em computadores lentos e rápidos (60fps ou 30fps dão o mesmo resultado).
        head.position += worm['direction'] * SPEED * time.dt

        # ── 3. Registrar posição no histórico ────────────────────────────────
        # insert(0, ...) adiciona a posição mais nova no início da lista.
        # Pense na lista como uma fila: o mais novo entra na frente,
        # e os segmentos mais distantes leem posições mais antigas (mais ao fundo).
        history.insert(0, Vec3(head.position))

        # Mantemos o histórico com tamanho fixo para não crescer infinitamente.
        # Precisamos de (NUM_SEGMENTS * SEGMENT_GAP / SPEED) posições no máximo.
        while len(history) > (NUM_SEGMENTS + 1) * 10:
            history.pop()

        # ── 4. Mover cada segmento para sua posição no histórico ─────────────
        for i, seg in enumerate(segments):
            # Cada segmento lê um índice diferente do histórico.
            # SEGMENT_GAP controla o "esticamento" do verme:
            # quanto maior, mais separados ficam os segmentos.
            history_index = int((i + 1) * SEGMENT_GAP * (SPEED / 5))
            history_index = min(history_index, len(history) - 1)
            seg.position = history[history_index]

        # ── 5. Orientar a cabeça na direção do movimento ─────────────────────
        # look_at() rotaciona a entidade para "olhar" em direção a um ponto.
        # Somamos a direção à posição atual para obter um ponto à frente.
        if worm['direction'].length() > 0:
            head.look_at(head.position + worm['direction'])


# ─── CONTROLES ADICIONAIS ─────────────────────────────────────────────────────
def input(key):
    # ESC fecha a aplicação
    if key == 'escape':
        quit()


# ─── EXIBIR CONTROLES NO TERMINAL ────────────────────────────────────────────
print("\n── Controles do Verme ─────────────────────────")
print("  Setas / WASD    Mover no plano horizontal")
print("  Q               Subir")
print("  E               Descer")
print("  Mouse           Orbitar câmera (EditorCamera)")
print("  ESC             Sair")
print("───────────────────────────────────────────────\n")

app.run()