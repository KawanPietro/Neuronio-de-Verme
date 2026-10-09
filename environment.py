import random

from ursina import *


class Environment:
    """Terreno limpo por default; com maze opt-in (colisao + spawn seguro)."""

    def __init__(self, maze=None):
        self.food_sources = []
        # maze: objeto Maze (ou None = limpo). Futuro gigante usa maze_enabled=1.
        self.maze = maze
        # Sem self.obstacles — terreno limpo por decisão de reconstrução.
        # O futuro labirinto gigante usará labirintos.json (atualmente vazio).

    def place_food(self, pos):
        """Cria uma fonte de alimento (alvo unico)."""
        src = Entity(
            model='sphere',
            color=color.lime.tint(-0.2),
            scale=1.6,
            position=Vec3(pos.x, 1, pos.z),
            collider='box',
        )
        self.food_sources.append(src)
        return src

    def clear_food(self):
        """Remove todos os alimentos."""
        for src in list(self.food_sources):
            destroy(src)
        self.food_sources.clear()

    def delete_source(self, entity):
        """Remove uma fonte de alimento existente da cena."""
        if entity in self.food_sources:
            self.food_sources.remove(entity)
            destroy(entity)
        else:
            # Tolerante a entidades desconhecidas (compat com scripts antigos)
            try:
                destroy(entity)
            except Exception:
                pass

    def randomize_food(self, n_foods=1, limit=24):
        """Reposiciona alimentos em lugares aleatórios (novo episódio)."""
        for src in list(self.food_sources):
            self.delete_source(src)
        for _ in range(n_foods):
            x = random.uniform(-limit, limit)
            z = random.uniform(-limit, limit)
            if self.maze is not None:
                from config import CONFIG as _C
                clr = _C.get('maze_clearance', 2.5)
                ok = False
                for _ in range(50):
                    if self.maze.free(x, z, clearance=clr):
                        ok = True
                        break
                    x = random.uniform(-limit, limit)
                    z = random.uniform(-limit, limit)
                if not ok:
                    continue  # sem ponto livre: mantém menos alimentos a atravessar
            self.place_food(Vec3(x, 1, z))

    def safe_spawn_and_food(self, seed, clearance=None, min_wall_dist=None,
                            min_food_spawn_dist=None):
        """Sorteia (spawn, food) por seed, longe de paredes e entre si.

        Retorna ((sx,sz),(fx,fz)). Sem maze: uniforme com anti-colado.
        Com maze: Maze.random_food_and_spawn (rejeicao + BFS fora daqui).
        """
        from config import CONFIG as _C
        clr = clearance or _C.get('maze_clearance', 2.5)
        mwd = min_wall_dist or _C.get('min_wall_dist', 2.5)
        mfd = min_food_spawn_dist or _C.get('min_food_spawn_dist', 8.0)
        if self.maze is not None:
            return self.maze.random_food_and_spawn(
                seed, clearance=clr, min_wall_dist=mwd,
                min_food_spawn_dist=mfd)
        rng = random.Random(seed)
        limit = _C.get('food_limit', 24)
        sx = rng.uniform(-limit, limit)
        sz = rng.uniform(-limit, limit)
        for _ in range(200):
            fx = rng.uniform(-limit, limit)
            fz = rng.uniform(-limit, limit)
            if ((fx - sx) ** 2 + (fz - sz) ** 2) ** 0.5 >= mfd:
                return (sx, sz), (fx, fz)
        return (sx, sz), (rng.uniform(-limit, limit),
                          rng.uniform(-limit, limit))

    def food_max_dist(self, episode_count=None):
        """Schedule do currículo (0/desligado = food_limit uniforme)."""
        from config import CONFIG
        if not CONFIG.get('food_curriculum', 0):
            return CONFIG.get('food_limit', 24)
        start = CONFIG.get('food_start_dist', 10.0)
        growth = CONFIG.get('food_growth', 0.3)
        ep = 0 if episode_count is None else episode_count
        return min(CONFIG.get('food_limit', 24), start + growth * ep)

    def randomize_food_near(self, cx, cz, max_dist, n_foods=1):
        """Alimento a até max_dist de (cx,cz). Terreno limpo: sem checagem de muros."""
        for src in list(self.food_sources):
            self.delete_source(src)
        import math as _math
        from config import CONFIG as _CFG
        _min_d = _CFG.get('food_radius', 3.0) + 1.0  # anti-sorte: nunca nasce já-comido
        for _ in range(n_foods):
            ang = random.uniform(0, 2 * _math.pi)
            dist = random.uniform(_min_d, max(_min_d + 0.5, max_dist))
            x, z = cx + _math.cos(ang) * dist, cz + _math.sin(ang) * dist
            self.place_food(Vec3(x, 1, z))

    def eat_and_respawn(self, eaten, limit=24, near=None):
        """Come o alimento e reaparece outro (sinal denso: 3-5 encontros/ep).

        Se near=(x,z), respawna a até `limit` do ponto (currículo);
        senão, uniforme.
        """
        if eaten in self.food_sources:
            self.delete_source(eaten)
        if near is not None:
            self.randomize_food_near(near[0], near[1], limit, n_foods=1)
        else:
            self.randomize_food(n_foods=1, limit=limit)
        return self.food_sources[0] if self.food_sources else None
