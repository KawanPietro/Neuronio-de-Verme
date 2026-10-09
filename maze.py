"""Labirinto gigante: geracao, colisao e spawn seguro (sem Ursina).

Formato labirintos.json v2 (compativel com legado):
  {
    "version": 2,
    "gigante": {
      "nome": "Gigante",
      "map_limit": 32,
      "spawn": [x, y, z],
      "food_zones": [[x, y, z], ...],
      "muros": [{"x":..,"z":..,"sx":..,"sz":..}, ...],   # caixas AABB (novo)
      "pedras": [{"x":..,"z":..,"scale":..}, ...],        # opcional
      "seed": 42
    }
  }
Legado (A/B era 11-dim): muros {x,z,angle,length,thickness} ainda sao
aceitos e convertidos para AABB.

Regras de seguranca (nunca nascer dentro da parede):
  free(x,z,clearance) = dentro do limite E fora de toda parede inflada.
  clearance padrao = worm_radius + wall_margin.
"""
import json
import math
import random
from collections import deque


def legacy_muro_to_aabb(m):
    """Converte muro legado {angle,length,thickness} em AABB {x,z,sx,sz}."""
    if 'sx' in m and 'sz' in m:
        return {'x': float(m['x']), 'z': float(m['z']),
                'sx': float(m['sx']), 'sz': float(m['sz'])}
    ang = float(m.get('angle', 0)) % 180
    length = float(m.get('length', 4))
    thick = float(m.get('thickness', 1.4))
    if abs(ang - 90) < 45:
        return {'x': float(m['x']), 'z': float(m['z']),
                'sx': thick, 'sz': length}
    return {'x': float(m['x']), 'z': float(m['z']),
            'sx': length, 'sz': thick}


def point_wall_dist(px, pz, wall):
    """Distancia euclidiana do ponto a caixa AABB (0 = dentro)."""
    hx, hz = wall['sx'] / 2.0, wall['sz'] / 2.0
    dx = max(abs(px - wall['x']) - hx, 0.0)
    dz = max(abs(pz - wall['z']) - hz, 0.0)
    return math.hypot(dx, dz)


def collides(px, pz, walls, radius=1.5):
    """True se o disco (px,pz,r) toca alguma parede."""
    for w in walls:
        if point_wall_dist(px, pz, w) < radius:
            return True
    return False


class Maze:
    def __init__(self, walls=None, limit=32, spawn=None, food_zones=None,
                 seed=42):
        self.walls = [legacy_muro_to_aabb(w) for w in (walls or [])]
        self.limit = float(limit)
        self.spawn = list(spawn) if spawn else [0, 1.25, -12]
        self.food_zones = [list(f) for f in (food_zones or [])]
        self.seed = int(seed)

    @classmethod
    def from_dict(cls, d):
        return cls(
            walls=d.get('muros', []),
            limit=d.get('map_limit', 32),
            spawn=d.get('spawn', [0, 1.25, -12]),
            food_zones=d.get('food_zones', []),
            seed=d.get('seed', 42),
        )

    @classmethod
    def load(cls, path, key='gigante'):
        with open(path) as f:
            data = json.load(f)
        if key not in data:
            raise KeyError(f"labirinto '{key}' ausente em {path}")
        return cls.from_dict(data[key])

    # ── Consultas ──────────────────────────────────────────────
    def free(self, x, z, clearance=2.5):
        if abs(x) > self.limit - clearance or abs(z) > self.limit - clearance:
            return False
        return not collides(x, z, self.walls, radius=clearance)

    def min_wall_dist(self, x, z):
        if not self.walls:
            return float('inf')
        return min(point_wall_dist(x, z, w) for w in self.walls)

    def random_free_point(self, rng, clearance=2.5, min_wall_dist=2.5,
                          tries=200):
        """Amostra ponto livre com folga minima das paredes (rejeicao)."""
        need = max(clearance, min_wall_dist)
        lo, hi = -self.limit + need, self.limit - need
        for _ in range(tries):
            x = rng.uniform(lo, hi)
            z = rng.uniform(lo, hi)
            if self.free(x, z, clearance=need):
                return (x, z)
        # fallback deterministico: varre grade grossa
        step = max(2.0, need)
        gx = lo
        while gx <= hi:
            gz = lo
            while gz <= hi:
                if self.free(gx, gz, clearance=need):
                    return (gx, gz)
                gz += step
            gx += step
        raise RuntimeError('sem ponto livre: labirinto denso demais ou '
                           f'clearance={need} grande p/ limit={self.limit}')

    def random_food_and_spawn(self, seed, clearance=2.5, min_wall_dist=2.5,
                              min_food_spawn_dist=8.0):
        """Sorteia (spawn, food) por seed, longe de paredes e entre si.

        Garante: ambos livres, dist minima das paredes, e distancia minima
        spawn<->food (anti-decoreba: cada seed = episodio diferente).
        """
        rng = random.Random(seed)
        sx, sz = self.random_free_point(rng, clearance, min_wall_dist)
        for _ in range(200):
            fx, fz = self.random_free_point(rng, clearance, min_wall_dist)
            if math.hypot(fx - sx, fz - sz) >= min_food_spawn_dist:
                return (sx, sz), (fx, fz)
        # labirinto pequeno demais p/ separar: aceita o mais distante amostrado
        fx, fz = self.random_free_point(rng, clearance, min_wall_dist)
        return (sx, sz), (fx, fz)

    # ── Validacao com BFS (conectividade spawn -> foods) ───────
    def bfs_path_len(self, sx, sz, fx, fz, cell=1.0, clearance=1.5):
        """Comprimento do caminho em grade (None = inalancavel)."""
        n = int(2 * self.limit / cell)
        if n <= 0 or n > 256:
            return None
        def to_grid(x, z):
            return (int((x + self.limit) / cell), int((z + self.limit) / cell))
        s = to_grid(sx, sz)
        t = to_grid(fx, fz)
        if not (0 <= s[0] < n and 0 <= s[1] < n):
            return None
        if not (0 <= t[0] < n and 0 <= t[1] < n):
            return None
        # bloqueios
        blocked = [[False] * n for _ in range(n)]
        for ix in range(n):
            for iz in range(n):
                x = -self.limit + (ix + 0.5) * cell
                z = -self.limit + (iz + 0.5) * cell
                if collides(x, z, self.walls, radius=clearance):
                    blocked[ix][iz] = True
        if blocked[s[0]][s[1]] or blocked[t[0]][t[1]]:
            return None
        dist = {s: 0}
        q = deque([s])
        while q:
            cx, cz = q.popleft()
            if (cx, cz) == t:
                return dist[(cx, cz)] * cell
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, nz = cx + dx, cz + dz
                if 0 <= nx < n and 0 <= nz < n and not blocked[nx][nz]:
                    if (nx, nz) not in dist:
                        dist[(nx, nz)] = dist[(cx, cz)] + 1
                        q.append((nx, nz))
        return None

    def validate(self, clearance=2.5, min_wall_dist=2.5):
        """Checa spawn/food livres + conectividade. Retorna lista de erros."""
        errors = []
        sx, _, sz = self.spawn[0], self.spawn[1], self.spawn[2]
        if not self.free(sx, sz, clearance):
            errors.append(f'spawn ({sx:.1f},{sz:.1f}) dentro de parede/borda')
        if self.min_wall_dist(sx, sz) < min_wall_dist:
            errors.append(f'spawn a {self.min_wall_dist(sx, sz):.2f}u da parede '
                          f'(min {min_wall_dist})')
        for i, f in enumerate(self.food_zones):
            fx, fz = f[0], f[2]
            if not self.free(fx, fz, clearance):
                errors.append(f'food[{i}] ({fx:.1f},{fz:.1f}) dentro de parede')
            if self.min_wall_dist(fx, fz) < min_wall_dist:
                errors.append(f'food[{i}] a {self.min_wall_dist(fx, fz):.2f}u '
                              f'da parede (min {min_wall_dist})')
            path = self.bfs_path_len(sx, sz, fx, fz)
            if path is None:
                errors.append(f'food[{i}] inalancavel do spawn (BFS sem caminho)')
        return errors


# ─── Gerador gigante (recursive-backtracker + braid p/ ramificacoes) ──
def generate_giant(seed=42, limit=32, cells=8, braid=0.25, wall_thick=1.2):
    """Gera labirinto com varias ramificacoes (DFS + atalhos braid).

    Grade cells x cells cobrindo [-limit+m, limit-m]. Cada celula comeca
    fechada; DFS abre arvore (caminho garantido); braid abre p% das paredes
    restantes (multiplos caminhos, sem beco unico). Muros fundidos por linha.
    """
    rng = random.Random(seed)
    margin = 3.0
    span = 2 * (limit - margin)
    cs = span / cells  # tamanho da celula
    # paredes internas: vertical (x): (ix,iz) parede leste da celula; horizontal: sul
    v_wall = [[True] * cells for _ in range(cells)]  # leste (ultima col ignorada)
    h_wall = [[True] * cells for _ in range(cells)]  # sul (ultima lin ignorada)
    seen = [[False] * cells for _ in range(cells)]
    stack = [(0, 0)]
    seen[0][0] = True
    while stack:
        cx, cz = stack[-1]
        opts = []
        if cx + 1 < cells and not seen[cx + 1][cz]:
            opts.append('E')
        if cx - 1 >= 0 and not seen[cx - 1][cz]:
            opts.append('W')
        if cz + 1 < cells and not seen[cx][cz + 1]:
            opts.append('S')
        if cz - 1 >= 0 and not seen[cx][cz - 1]:
            opts.append('N')
        if not opts:
            stack.pop()
            continue
        d = rng.choice(opts)
        if d == 'E':
            v_wall[cx][cz] = False
            seen[cx + 1][cz] = True
            stack.append((cx + 1, cz))
        elif d == 'W':
            v_wall[cx - 1][cz] = False
            seen[cx - 1][cz] = True
            stack.append((cx - 1, cz))
        elif d == 'S':
            h_wall[cx][cz] = False
            seen[cx][cz + 1] = True
            stack.append((cx, cz + 1))
        else:
            h_wall[cx][cz - 1] = False
            seen[cx][cz - 1] = True
            stack.append((cx, cz - 1))
    # braid: reabre p das paredes restantes (multiplos caminhos)
    for ix in range(cells):
        for iz in range(cells):
            if ix + 1 < cells and v_wall[ix][iz] and rng.random() < braid:
                v_wall[ix][iz] = False
            if iz + 1 < cells and h_wall[ix][iz] and rng.random() < braid:
                h_wall[ix][iz] = False
    # converte para caixas, fundindo segmentos colineares contiguos
    x0 = -(limit - margin)
    z0 = -(limit - margin)
    muros = []
    # verticais: para cada coluna de grade ix+1, funde linhas contiguas
    for ix in range(cells - 1):
        run = None
        for iz in range(cells):
            if v_wall[ix][iz]:
                zc0 = z0 + iz * cs
                if run is None:
                    run = [zc0, zc0 + cs]
                else:
                    run[1] = zc0 + cs
            else:
                if run is not None:
                    wx = x0 + (ix + 1) * cs
                    muros.append({'x': wx, 'z': (run[0] + run[1]) / 2,
                                  'sx': wall_thick, 'sz': run[1] - run[0]})
                    run = None
        if run is not None:
            wx = x0 + (ix + 1) * cs
            muros.append({'x': wx, 'z': (run[0] + run[1]) / 2,
                          'sx': wall_thick, 'sz': run[1] - run[0]})
    # horizontais
    for iz in range(cells - 1):
        run = None
        for ix in range(cells):
            if h_wall[ix][iz]:
                xc0 = x0 + ix * cs
                if run is None:
                    run = [xc0, xc0 + cs]
                else:
                    run[1] = xc0 + cs
            else:
                if run is not None:
                    wz = z0 + (iz + 1) * cs
                    muros.append({'x': (run[0] + run[1]) / 2, 'z': wz,
                                  'sx': run[1] - run[0], 'sz': wall_thick})
                    run = None
        if run is not None:
            wz = z0 + (iz + 1) * cs
            muros.append({'x': (run[0] + run[1]) / 2, 'z': wz,
                          'sx': run[1] - run[0], 'sz': wall_thick})
    # spawn: celula (0,0) centro; foods: 2 cantos distantes (BFS valido depois)
    spawn = [x0 + cs / 2, 1.25, z0 + cs / 2]
    foods = [[x0 + span - cs / 2, 1, z0 + span - cs / 2],
             [x0 + span - cs / 2, 1, z0 + cs / 2]]
    return {'nome': 'Gigante', 'descricao': f'DFS {cells}x{cells} seed {seed}',
            'map_limit': limit, 'spawn': spawn, 'food_zones': foods,
            'muros': muros, 'pedras': [], 'seed': seed}


def ascii_map(maze, cell=2.0):
    """Mapa ASCII p/ revisao humana antes da GUI (# = parede, S = spawn)."""
    n = int(2 * maze.limit / cell)
    n = max(1, min(n, 64))
    lines = []
    sx, sz = maze.spawn[0], maze.spawn[2]
    for iz in range(n - 1, -1, -1):
        row = ''
        for ix in range(n):
            x = -maze.limit + (ix + 0.5) * cell
            z = -maze.limit + (iz + 0.5) * cell
            if math.hypot(x - sx, z - sz) < cell:
                row += 'S'
            elif any(math.hypot(x - f[0], z - f[2]) < cell for f in maze.food_zones):
                row += 'F'
            elif collides(x, z, maze.walls, radius=cell / 2):
                row += '#'
            else:
                row += '.'
        lines.append(row)
    return '\n'.join(lines)
