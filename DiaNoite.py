#Realizado tudo 100% Pela IA

from ursina import *

app = Ursina()

# O objeto que o sol vai "iluminar" na cena
Entity(model='cube', color=color.orange, scale=2)

# ─── LUZ DIRECIONAL (o "sol") ───────────────────────────────────────────────
# DirectionalLight simula uma fonte de luz infinitamente distante (como o sol real).
# Todos os raios chegam paralelos, criando sombras nítidas e consistentes.
sun = DirectionalLight()
sun.look_at(Vec3(1, -1, 1))  # direção inicial: vindo de cima e da lateral

# ─── LUZ AMBIENTE ────────────────────────────────────────────────────────────
# A luz ambiente preenche as sombras. Sem ela, a noite não funciona visualmente:
# o cubo ficaria totalmente iluminado mesmo sem o sol.
# Usamos a classe AmbientLight do próprio Ursina para isso.
ambient = AmbientLight()
ambient.color = color.white  # começa claro (dia)

# ─── CÉU (Skybox) ────────────────────────────────────────────────────────────
# Sky() cria uma esfera gigante ao redor da cena.
# Mudamos sky.color para trocar a cor do céu.
sky = Sky()

# ─── CONFIGURAÇÃO DO CICLO ───────────────────────────────────────────────────
cycle_speed = 20  # graus por segundo. Aumente para um dia mais rápido.


def update():
    # ─── PASSO 1: ROTACIONAR O SOL ──────────────────────────────────────────
    # time.dt = "delta time": tempo (em segundos) desde o último frame.
    # Multiplicar por time.dt garante que a velocidade seja igual em qualquer
    # computador, independente dos FPS (frames por segundo).
    sun.rotate(Vec3(cycle_speed * time.dt, 0, 0))

    # ─── PASSO 2: CALCULAR "t" (posição no ciclo) ───────────────────────────
    # t vai de 0.0 (começo do dia) até 1.0 (fim do ciclo completo).
    # O % 360 garante que o ângulo sempre fique entre 0 e 360, evitando
    # números gigantes que poderiam causar imprecisão ao longo do tempo.
    t = (sun.world_rotation_x % 360) / 360

    # ─── PASSO 3: CALCULAR AS CORES ─────────────────────────────────────────
    # lerp(a, b, t) = "linear interpolation" (interpolação linear).
    # Retorna um valor entre 'a' e 'b' conforme 't' vai de 0 a 1.
    # Exemplo: lerp(azul, preto, 0.5) = um azul bem escuro (metade do caminho).

    if t < 0.25:
        # Amanhecer: laranja → azul céu
        sky_color = lerp(color.orange, color.cyan, t * 4)
        sun_intensity = lerp(0.2, 1.0, t * 4)  # sol vai ganhando força

    elif t < 0.5:
        # Tarde: azul céu → laranja entardecer
        sky_color = lerp(color.cyan, color.orange, (t - 0.25) * 4)
        sun_intensity = lerp(1.0, 0.2, (t - 0.25) * 4)

    elif t < 0.75:
        # Entardecer → noite
        sky_color = lerp(color.orange, color.black, (t - 0.5) * 4)
        sun_intensity = lerp(0.2, 0.0, (t - 0.5) * 4)

    else:
        # Noite funda → próximo amanhecer
        sky_color = lerp(color.black, color.orange, (t - 0.75) * 4)
        sun_intensity = lerp(0.0, 0.2, (t - 0.75) * 4)

    # ─── PASSO 4: APLICAR AS CORES NA CENA ──────────────────────────────────

    # Cor do céu: usamos sky.color (e não window.color, que não funciona assim)
    sky.color = sky_color

    # Intensidade da luz do sol: clamp01 garante que o valor fique entre 0 e 1
    sun.color = Color(sun_intensity, sun_intensity, sun_intensity * 0.9, 1)

    # Luz ambiente: à noite fica quase zero; de dia fica clara
    # Sem isso, os objetos ficam igualmente iluminados em qualquer hora do dia!
    ambient_strength = max(0.05, sun_intensity * 0.4)
    ambient.color = Color(ambient_strength, ambient_strength, ambient_strength, 1)


app.run()