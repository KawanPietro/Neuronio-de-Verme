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

    def step(self, dt):
        """Move a cabeça conforme self.direction, limitado ao mapa."""
        self.head.position += self.direction * CONFIG['speed'] * dt
        self.head.x = max(-CONFIG['map_limit'], min(CONFIG['map_limit'], self.head.x))
        self.head.z = max(-CONFIG['map_limit'], min(CONFIG['map_limit'], self.head.z))
        self.head.y = CONFIG['segment_size'] / 2

        self.update_segments()

        if self.direction.length() > 0:
            self.head.look_at(self.head.position + self.direction)

    def reset(self):
        """Reposiciona o corpo no centro e zera o histórico."""
        self.head.position = Vec3(0, CONFIG['segment_size'] / 2, 0)
        for i, seg in enumerate(self.segments):
            seg.position = self.head.position - Vec3(0, 0, (i + 1) * CONFIG['segment_gap'])
        self.history.clear()
        for _ in range(self.max_history_length):
            self.history.append(Vec3(self.head.position))
        self.direction = Vec3(0, 0, 1)