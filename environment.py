import random

from ursina import *


class Environment:
    """Fontes editáveis (luz/chuva/alimento), obstáculos e partículas de chuva."""

    def __init__(self):
        self.light_sources = []
        self.rain_sources = []
        self.food_sources = []  # Reformulacao alimento (E1/E2): reforçador único
        self.obstacles = []       # Fase 15: blocos físicos que exigem desvio
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

    def place_food(self, pos):
        """Cria uma fonte de alimento (alvo único da reformulação)."""
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

    def place_obstacle(self, pos, scale=2.0):
        """Cria um obstáculo sólido (pedra) — Fase 15."""
        from config import CONFIG
        # Variação de tamanho/cor para não ficar monótono
        jitter = random.uniform(0.85, 1.15)
        s = scale * jitter
        obs = Entity(
            model='cube',
            color=color.gray.tint(random.uniform(-0.2, -0.05)),
            scale=(s, s * 0.9, s),
            position=Vec3(pos.x, s * 0.45, pos.z),
            collider='box',
        )
        # Sombra/borda como filho — é destruída junto com o pai
        Entity(parent=obs, model='cube', color=color.black33,
               scale=(1.02, 0.12 / (s * 0.9), 1.02),
               position=(0, -0.44, 0))
        self.obstacles.append(obs)
        return obs

    def place_wall(self, pos, length=9.0, thickness=1.4, angle=0):
        """Muro comprido que força contorno — níveis 2-3."""
        obs = Entity(
            model='cube',
            color=color.dark_gray.tint(-0.1),
            scale=(length, 2.0, thickness),
            position=Vec3(pos.x, 1.0, pos.z),
            rotation_y=angle,
            collider='box',
        )
        # Sombra do muro
        Entity(parent=obs, model='cube', color=color.black33,
               scale=(1.02, 0.06, 1.02), position=(0, -0.47, 0))
        self.obstacles.append(obs)
        return obs

    def delete_source(self, entity):
        """Remove uma fonte existente da cena."""
        if entity in self.light_sources:
            self.light_sources.remove(entity)
            destroy(entity)
        elif entity in self.rain_sources:
            self.rain_sources.remove(entity)
            destroy(entity)
            self.rebuild_rain_particles()
        elif entity in self.food_sources:
            self.food_sources.remove(entity)
            destroy(entity)
        elif entity in self.obstacles:
            self.obstacles.remove(entity)
            destroy(entity)

    def clear_obstacles(self):
        """Remove todos os obstáculos (troca de nível)."""
        for obs in list(self.obstacles):
            destroy(obs)
        self.obstacles.clear()

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

    def randomize_food(self, n_foods=1, limit=12):
        """Reposiciona alimentos em lugares aleatórios livres (novo episódio)."""
        for src in list(self.food_sources):
            self.delete_source(src)
        for _ in range(n_foods):
            for _ in range(20):
                x = random.uniform(-limit, limit)
                z = random.uniform(-limit, limit)
                ok = True
                for obs in self.obstacles:
                    if abs(obs.x - x) < 3.0 and abs(obs.z - z) < 3.0:
                        ok = False
                        break
                if not ok:
                    continue
                self.place_food(Vec3(x, 1, z))
                break

    def food_max_dist(self, episode_count=None):
        """T8: schedule do currículo (0/desligado = food_limit legado)."""
        from config import CONFIG
        if not CONFIG.get('food_curriculum', 0):
            return CONFIG.get('food_limit', 12)
        start = CONFIG.get('food_start_dist', 5.0)
        growth = CONFIG.get('food_growth', 0.15)
        ep = 0 if episode_count is None else episode_count
        return min(CONFIG.get('food_limit', 12), start + growth * ep)

    def randomize_food_near(self, cx, cz, max_dist, n_foods=1):
        """T8: alimento a até max_dist de (cx,cz), livre de obstáculos.

        Fallback: se 20 tentativas falharem (corredor estreito), usa
        randomize_food legado em vez de travar o episódio (anti-defeito).
        """
        for src in list(self.food_sources):
            self.delete_source(src)
        import math as _math
        from config import CONFIG as _CFG
        _min_d = _CFG.get('food_radius', 3.0) + 1.0  # anti-sorte: nunca nasce já-comido
        for _ in range(n_foods):
            placed = False
            for _ in range(20):
                ang = random.uniform(0, 2 * _math.pi)
                dist = random.uniform(_min_d, max(_min_d + 0.5, max_dist))
                x, z = cx + _math.cos(ang) * dist, cz + _math.sin(ang) * dist
                ok = True
                for obs in self.obstacles:
                    if abs(obs.x - x) < 3.0 and abs(obs.z - z) < 3.0:
                        ok = False
                        break
                if not ok:
                    continue
                self.place_food(Vec3(x, 1, z))
                placed = True
                break
            if not placed:
                self.randomize_food(n_foods=1)
                break

    def eat_and_respawn(self, eaten, limit=12, near=None):
        """E2: come o alimento e reaparece outro em ponto livre (sinal denso).

        T8: se near=(x,z), respawna a até `limit` do ponto (currículo);
        senão, uniforme legado.
        """
        if eaten in self.food_sources:
            self.delete_source(eaten)
        if near is not None:
            self.randomize_food_near(near[0], near[1], limit, n_foods=1)
        else:
            self.randomize_food(n_foods=1, limit=limit)
        return self.food_sources[0] if self.food_sources else None

    def load_maze(self, name, maze_file='labirintos.json'):
        """E5: carrega labirinto A_treino ou B_teste (treino != teste).

        Aceita 'A'/'a_treino' e 'B'/'b_teste'. Limpa obstáculos + alimentos,
        constrói muros/pedras e posiciona alimentos das food_zones.
        Retorna {'spawn': Vec3, 'food_zones': [...]}.
        """
        import json
        import os
        aliases = {'a': 'a_treino', 'a_treino': 'a_treino',
                   'b': 'b_teste', 'b_teste': 'b_teste'}
        key = aliases.get(str(name).lower(), str(name).lower())
        base = os.path.dirname(os.path.abspath(__file__))
        path = maze_file if os.path.isabs(maze_file) else os.path.join(base, maze_file)
        with open(path) as f:
            data = json.load(f)
        if key not in data:
            raise ValueError(f"Labirinto '{name}' desconhecido (chaves: {list(data.keys())})")
        maze = data[key]
        self.clear_obstacles()
        self.clear_food()
        for w in maze.get('muros', []):
            self.place_wall(Vec3(w['x'], 0, w['z']),
                            length=w.get('length', 9.0),
                            thickness=w.get('thickness', 1.4),
                            angle=w.get('angle', 0))
        for p in maze.get('pedras', []):
            self.place_obstacle(Vec3(p['x'], 0, p['z']),
                                scale=p.get('scale', 1.8))
        for fz in maze.get('food_zones', []):
            self.place_food(Vec3(fz[0], 1, fz[2]))
        spawn = maze.get('spawn', [0, 1.25, -12])
        return {'spawn': Vec3(spawn[0], spawn[1], spawn[2]),
                'food_zones': list(maze.get('food_zones', []))}

    def randomize_obstacles(self, difficulty=1):
        """Gera obstáculos conforme o nível de dificuldade — Fase 15+."""
        from config import CONFIG
        self.clear_obstacles()
        lvl = CONFIG['difficulty_levels'].get(difficulty, CONFIG['difficulty_levels'][1])
        n = lvl['n_obstacles']
        n_walls = lvl.get('n_walls', 0)
        scale = lvl['obstacle_scale']
        lim = CONFIG['map_limit'] - 2
        # ── Pedras espalhadas ──────────────────────────────────────────────
        for _ in range(n):
            for _ in range(14):
                x = random.uniform(-lim, lim)
                z = random.uniform(-lim, lim)
                if abs(x) < 3 and abs(z) < 3:
                    continue
                ok = True
                for src in self.light_sources + self.rain_sources + self.food_sources:
                    if abs(src.x - x) < 3.5 and abs(src.z - z) < 3.5:
                        ok = False
                        break
                # Evita pedras muito coladas entre si
                for obs in self.obstacles:
                    if abs(obs.x - x) < 3.0 and abs(obs.z - z) < 3.0:
                        ok = False
                        break
                if not ok:
                    continue
                self.place_obstacle(Vec3(x, 0, z), scale=scale)
                break
        # ── Muros (níveis 2-3) — criam gargalos que exigem planejar rota ──
        wall_len = CONFIG.get('wall_length', 9.0)
        wall_thick = CONFIG.get('wall_thickness', 1.4)
        for i in range(n_walls):
            for _ in range(14):
                x = random.uniform(-lim+4, lim-4)
                z = random.uniform(-lim+4, lim-4)
                if abs(x) < 4 and abs(z) < 4:
                    continue
                angle = random.choice([0, 90, 0, 90, 45, -45])
                # Não bloqueia completamente o centro: deixa corredor
                if abs(x) < 7 and abs(z) < 7 and angle in (0, 90):
                    continue
                ok = True
                for src in self.light_sources + self.rain_sources + self.food_sources:
                    if abs(src.x - x) < 4.5 and abs(src.z - z) < 4.5:
                        ok = False
                        break
                if not ok:
                    continue
                self.place_wall(Vec3(x, 0, z), length=wall_len, thickness=wall_thick, angle=angle)
                break

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