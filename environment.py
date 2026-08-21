import random

from ursina import *


class Environment:
    """Fontes editáveis (luz/chuva) e o sistema de partículas de chuva."""

    def __init__(self):
        self.light_sources = []
        self.rain_sources = []
        self.rain_particles = []

    def place_light(self, pos):
        """Cria uma fonte de luz (perigo)."""
        src = Entity(
            model='sphere',
            color=color.yellow.tint(-0.2),
            scale=2,
            position=Vec3(pos.x, 1, pos.z),
            collider='box',
        )
        self.light_sources.append(src)
        # Efeito visual ao adicionar luz
        invoke(destroy, Entity(model='sphere', color=color.white, scale=3, position=src.position), delay=0.5)

    def place_rain(self, pos):
        """Cria uma fonte de chuva (alvo)."""
        src = Entity(
            model='sphere',
            color=color.cyan.tint(-0.2),
            scale=2,
            position=Vec3(pos.x, 1, pos.z),
            collider='box',
        )
        self.rain_sources.append(src)
        self.rebuild_rain_particles()

    def delete_source(self, entity):
        """Remove uma fonte existente da cena."""
        if entity in self.light_sources:
            self.light_sources.remove(entity)
            destroy(entity)
        elif entity in self.rain_sources:
            self.rain_sources.remove(entity)
            destroy(entity)
            self.rebuild_rain_particles()

    def randomize_sources(self, n_lights=1, n_rains=1, margin=4, limit=15):
        """
        Reposiciona as fontes em lugares aleatórios (novo episódio / nova cena).

        É o que faz o verme generalizar: cada episódio começa com uma cena nova,
        em vez de decorar a posição fixa das fontes.
        """
        for src in list(self.light_sources) + list(self.rain_sources):
            self.delete_source(src)
        for _ in range(n_lights):
            self.place_light(Vec3(random.uniform(-limit, limit), 1, random.uniform(-limit, limit)))
        for _ in range(n_rains):
            self.place_rain(Vec3(random.uniform(-limit, limit), 1, random.uniform(-limit, limit)))

    def rebuild_rain_particles(self):
        """Recria todas as partículas de chuva ao redor das fontes atuais."""
        for p in self.rain_particles:
            destroy(p)
        self.rain_particles.clear()

        for src in self.rain_sources:
            for _ in range(40):
                self.rain_particles.append(Entity(
                    model='sphere',
                    color=Color(0.5, 0.8, 1, 0.6),
                    scale=0.1,
                    position=(
                        src.x + random.uniform(-3, 3),
                        random.uniform(0, 8),
                        src.z + random.uniform(-3, 3),
                    )
                ))

    def update_rain_particles(self):
        """Faz as partículas caírem e se reposicionarem na fonte mais próxima."""
        for p in self.rain_particles:
            p.y -= time.dt * 3
            if p.y < 0:
                if self.rain_sources:
                    src = min(
                        self.rain_sources,
                        key=lambda r: (Vec3(p.x, 0, p.z) - Vec3(r.x, 0, r.z)).length()
                    )
                    p.x = src.x + random.uniform(-3, 3)
                    p.z = src.z + random.uniform(-3, 3)
                p.y = 8