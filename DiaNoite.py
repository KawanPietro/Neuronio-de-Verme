from ursina import *

app = Ursina()

Entity(model='cube', color=color.orange, scale=2)

# ─── LUZES ───────────────────────────────────────────────────────────────────
sun = DirectionalLight()
sun.look_at(Vec3(1, -1, 1))

ambient = AmbientLight()
ambient.color = color.white

sky = Sky()

# ─── ESTADO DO CICLO ─────────────────────────────────────────────────────────
# Usamos um dicionário para agrupar todas as variáveis de controle em um só lugar.
# Isso evita variáveis globais soltas e facilita expandir o sistema futuramente.
cycle = {
    'auto'       : True,   # True = roda sozinho | False = controle manual
    'speed'      : 20,     # graus por segundo no modo automático
    'angle'      : 0.0,    # ângulo atual do sol (0° a 360°)
    'step_manual': 2.0,    # quantos graus avançar/recuar por tecla no modo manual
}


# ─── FUNÇÃO AUXILIAR: aplicar o ângulo atual na cena ─────────────────────────
# Separar a lógica de "calcular cores" da lógica de "avançar o tempo" é uma
# boa prática chamada Separação de Responsabilidades. Assim, tanto o modo
# automático quanto o manual chamam a mesma função para atualizar a cena,
# sem duplicar código.
def apply_cycle(angle):
    """Recebe um ângulo (0–360) e atualiza céu, sol e luz ambiente."""

    t = (angle % 360) / 360  # normaliza para 0.0–1.0

    # ── Fase 1: Amanhecer (t: 0.00 → 0.25) ──────────────────────────────────
    if t < 0.25:
        f = t * 4                                          # fração local 0→1
        sky_color    = lerp(color.orange, color.cyan, f)
        sun_intensity = lerp(0.2, 1.0, f)

    # ── Fase 2: Tarde (t: 0.25 → 0.50) ──────────────────────────────────────
    elif t < 0.5:
        f = (t - 0.25) * 4
        sky_color    = lerp(color.cyan, color.orange, f)
        sun_intensity = lerp(1.0, 0.2, f)

    # ── Fase 3: Entardecer → Noite (t: 0.50 → 0.75) ─────────────────────────
    elif t < 0.75:
        f = (t - 0.5) * 4
        sky_color    = lerp(color.orange, color.black, f)
        sun_intensity = lerp(0.2, 0.0, f)

    # ── Fase 4: Noite funda → Amanhecer (t: 0.75 → 1.00) ────────────────────
    else:
        f = (t - 0.75) * 4
        sky_color    = lerp(color.black, color.orange, f)
        sun_intensity = lerp(0.0, 0.2, f)

    # Aplica as cores calculadas na cena
    sky.color    = sky_color
    sun.color    = Color(sun_intensity, sun_intensity, sun_intensity * 0.9, 1)

    ambient_strength = max(0.05, sun_intensity * 0.4)
    ambient.color    = Color(ambient_strength, ambient_strength, ambient_strength, 1)

    # Reposiciona a luz direcional conforme o ângulo do ciclo.
    # rotate() acumula rotação a cada frame; aqui preferimos definir a rotação
    # de forma absoluta para que o manual e o automático se comportem igual.
    sun.rotation_x = angle


# ─── FUNÇÃO AUXILIAR: exibir instruções no terminal ──────────────────────────
def print_controls():
    print("\n── Controles do Ciclo Dia/Noite ──────────────────")
    print("  [SPACE]      Alternar automático / manual")
    print("  [→] ou [D]   Avançar o sol  (modo manual)")
    print("  [←] ou [A]   Recuar  o sol  (modo manual)")
    print("  [+]          Aumentar velocidade (modo automático)")
    print("  [-]          Diminuir velocidade (modo automático)")
    print("──────────────────────────────────────────────────")
    mode = "AUTOMÁTICO" if cycle['auto'] else "MANUAL"
    print(f"  Modo atual: {mode}  |  Velocidade: {cycle['speed']}°/s\n")


# ─── INPUT: alternar modo automático / manual ─────────────────────────────────
# input() é chamado pelo Ursina sempre que uma tecla é pressionada.
# 'key' é uma string com o nome da tecla (ex: 'space', 'right arrow', 'a').
def input(key):

    if key == 'space':
        # O operador 'not' inverte um booleano: True → False, False → True.
        cycle['auto'] = not cycle['auto']
        print_controls()

    # ── Controles manuais de ângulo ───────────────────────────────────────────
    # Só respondem quando o modo automático está desligado.
    elif key in ('right arrow', 'd') and not cycle['auto']:
        cycle['angle'] = (cycle['angle'] + cycle['step_manual']) % 360
        apply_cycle(cycle['angle'])

    elif key in ('left arrow', 'a') and not cycle['auto']:
        cycle['angle'] = (cycle['angle'] - cycle['step_manual']) % 360
        apply_cycle(cycle['angle'])

    # ── Controles de velocidade ───────────────────────────────────────────────
    # max(..., 1) impede que a velocidade chegue a zero ou fique negativa.
    elif key == 'o':
        cycle['speed'] = min(cycle['speed'] + 5, 360)
        print(f"  Velocidade: {cycle['speed']}°/s")

    elif key == 'p':
        cycle['speed'] = max(cycle['speed'] - 5, 1)
        print(f"  Velocidade: {cycle['speed']}°/s")


# ─── UPDATE: loop principal (chamado todo frame) ──────────────────────────────
# No modo automático, avançamos o ângulo com base no tempo decorrido (time.dt).
# No modo manual, o ângulo só muda quando o jogador pressiona uma tecla,
# por isso não fazemos nada aqui nesse caso.
def update():
    if cycle['auto']:
        # time.dt garante movimento uniforme independente dos FPS da máquina.
        cycle['angle'] = (cycle['angle'] + cycle['speed'] * time.dt) % 360
        apply_cycle(cycle['angle'])


# ─── INICIALIZAÇÃO ────────────────────────────────────────────────────────────
# Aplica o estado inicial antes do primeiro frame e exibe os controles.
apply_cycle(cycle['angle'])
print_controls()

app.run()