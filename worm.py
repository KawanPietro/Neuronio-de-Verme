import math

from ursina import *

from config import CONFIG


def interpolate_color(color1, color2, t):
    """Interpola duas cores com fator t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    r = color1.r + (color2.r - color1.r) * t
    g = color1.g + (color2.g - color1.g) * t
    b = color1.b + (color2.b - color1.b) * t
    a = color1.a + (color2.a - color1.a) * t
    return Color(r, g, b, a)


class Worm:
    """O corpo do verme: cabeça + segmentos, com histórico e animação."""

    def __init__(self):
        num  = CONFIG['num_segments']
        size = CONFIG['segment_size']
        gap  = CONFIG['segment_gap']

        # Gradiente de cores do preto ao azul brilhante
        self.colors = [
            interpolate_color(color.black, color.cyan.tint(0.5), i / num)
            for i in range(num)
        ]

        self.head = Entity(
            model='sphere',
            color=color.cyan.tint(-0.2),
            scale=size * 1.1,
            position=(0, size / 2, 0),
            collider='sphere',
        )

        self.segments = [
            Entity(
                model='sphere',
                color=self.colors[i],
                scale=size * (1.0 - i * 0.05),
                position=self.head.position - Vec3(0, 0, i * gap),
            )
            for i in range(num)
        ]

        # Histórico de posições da cabeça usado para o efeito "cobra"
        self.history = [Vec3(self.head.position)] * (num + 1) * 4
        self.max_history_length = (num + 1) * 10
        self.direction = Vec3(0, 0, 1)

    def animate(self):
        """Animação de ondulação dos segmentos."""
        for i, seg in enumerate(self.segments):
            seg.scale = CONFIG['segment_size'] * (1.0 - i * 0.05) * (1 + math.sin(time.time() * 5 + i) * 0.1)
            seg.color = self.colors[i].tint(math.sin(time.time() * 3 + i) * 0.2)
            seg.rotation_y += math.sin(time.time() * 2 + i) * 5

    def update_segments(self):
        """Segue o histórico de posições da cabeça (efeito cobra)."""
        self.history.insert(0, Vec3(self.head.position))
        while len(self.history) > self.max_history_length:
            self.history.pop()

        for i, seg in enumerate(self.segments):
            idx = min(int((i + 1) * CONFIG['segment_gap'] * (CONFIG['speed'] / 4)), len(self.history) - 1)
            seg.position = self.history[idx]

    def step(self, dt, maze=None, bounce=False):
        """Move a cabeça conforme self.direction, com colisão deslizante.

        maze: objeto com .free(x,z,clearance) ou None (terreno limpo).
        Desliza em X depois Z: nunca atravessa parede, nunca prende.
        bounce (só passeio visual/apresentação): ao encostar na borda do mapa,
        reflete a direção em vez de empurrar a grade para sempre (no treino é
        False: a física do treino/eval/headless fica intacta).
        """
        import math as _math
        nx = self.head.position.x + self.direction.x * CONFIG['speed'] * dt
        nz = self.head.position.z + self.direction.z * CONFIG['speed'] * dt
        lim = CONFIG['map_limit']
        hit_x = abs(nx) >= lim
        hit_z = abs(nz) >= lim
        nx = max(-lim, min(lim, nx))
        nz = max(-lim, min(lim, nz))
        radius = CONFIG.get('worm_radius', 1.5)
        if maze is not None:
            try:
                free = maze.free
            except AttributeError:
                free = lambda x, z, clearance=radius: True
            cx, cz = self.head.position.x, self.head.position.z
            # slide: tenta X, depois Z (qualquer direção, inclusive descer)
            if free(nx, cz, clearance=radius):
                cx = nx
            if free(cx, nz, clearance=radius):
                cz = nz
            nx, nz = cx, cz
        # Atribuição do Vec3 INTEIRO (via setter): mutar .position.x/.z num
        # objeto temporário não move a entidade na cena (regressão corrigida).
        self.head.position = Vec3(nx, CONFIG['segment_size'] / 2, nz)

        # Passeio: quicou na borda -> volta p/ dentro (nunca moe a grade).
        if bounce and (hit_x or hit_z):
            if hit_x:
                self.direction.x *= -1
            if hit_z:
                self.direction.z *= -1
            if self.direction.length() > 0.01:
                self.direction = self.direction.normalized()

        self.update_segments()

        if self.direction.length() > 0:
            self.head.look_at(self.head.position + self.direction)

    def teleport(self, pos):
        """Coloca cabeca + corpo + historico em `pos` (sem rastro esticado).

        Sem isso, resetar a cabeca p/ o spawn com o historico ainda na origem
        empilhava os 12 segmentos na origem (verme "bugado"/esticado por ~2s).
        """
        self.head.position = Vec3(pos)
        for i, seg in enumerate(self.segments):
            seg.position = self.head.position - Vec3(0, 0, (i + 1) * CONFIG['segment_gap'])
        self.history.clear()
        for _ in range(self.max_history_length):
            self.history.append(Vec3(self.head.position))

    def reset(self, heading=None):
        """Reposiciona o corpo no centro e zera o histórico.

        heading: angulo rad (0=L, pi/2=N...); None = CONFIG['spawn_heading']
        ('random' = qualquer direcao, 'fixed' = norte legado (0,0,1)).
        Diagonais saem naturalmente (angulo continuo).
        """
        import math as _math
        import random as _random
        self.teleport(Vec3(0, CONFIG['segment_size'] / 2, 0))
        if heading is None:
            mode = CONFIG.get('spawn_heading', 'random')
            if mode == 'random':
                heading = _random.uniform(0, 2 * _math.pi)
            else:
                heading = _math.pi / 2  # norte legado
        self.direction = Vec3(_math.cos(heading), 0, _math.sin(heading))
        if self.direction.length() > 0.01:
            self.direction = self.direction.normalized()