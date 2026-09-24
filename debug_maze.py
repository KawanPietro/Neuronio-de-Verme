"""
Validador headless dos labirintos (seção F8 do README.md).

Verifica:
- labirintos.json valido com A_treino != B_teste
- spawn e food_zones fora de muros/pedras (com margem)
- existe caminho BFS spawn -> cada food_zone (sem bloqueio total)

Rode com: python debug_maze.py
"""
import json
import math
from collections import deque

MAZE_FILE = 'labirintos.json'
CELL = 1.0
MARGIN = 1.0  # margem extra ao redor dos muros


def load():
    with open(MAZE_FILE) as f:
        return json.load(f)


def rect_of_wall(w):
    """Retorna (cx, cz, hx, hz) do muro no plano XZ (angulo 0/90 exato, outros via bbox)."""
    x, z = w['x'], w['z']
    length = w.get('length', 9.0)
    thick = w.get('thickness', 1.4)
    angle = w.get('angle', 0)
    if angle in (0, 180, -180):
        return x, z, length / 2.0 + MARGIN, thick / 2.0 + MARGIN
    if angle in (90, -90, 270, -270):
        return x, z, thick / 2.0 + MARGIN, length / 2.0 + MARGIN
    # diagonal (45/-45): bbox conservadora
    r = (abs(math.cos(math.radians(angle))) * length
         + abs(math.sin(math.radians(angle))) * thick) / 2.0 + MARGIN
    c = (abs(math.sin(math.radians(angle))) * length
         + abs(math.cos(math.radians(angle))) * thick) / 2.0 + MARGIN
    return x, z, r, c


def build_blocked(maze, limit):
    """Conjunto de celulas (gx, gz) bloqueadas."""
    blocked = set()
    n = int(limit * 2 / CELL)
    def to_grid(x, z):
        return (int(round((x + limit) / CELL)), int(round((z + limit) / CELL)))
    # muros
    for w in maze.get('muros', []):
        cx, cz, hx, hz = rect_of_wall(w)
        gx0, gz0 = to_grid(cx - hx, cz - hz)
        gx1, gz1 = to_grid(cx + hx, cz + hz)
        for gx in range(min(gx0, gx1), max(gx0, gx1) + 1):
            for gz in range(min(gz0, gz1), max(gz0, gz1) + 1):
                blocked.add((gx, gz))
    # pedras (circulo aproximado)
    for p in maze.get('pedras', []):
        rad = p.get('scale', 1.8) / 2.0 + MARGIN
        gx0, gz0 = to_grid(p['x'] - rad, p['z'] - rad)
        gx1, gz1 = to_grid(p['x'] + rad, p['z'] + rad)
        for gx in range(min(gx0, gx1), max(gx0, gx1) + 1):
            for gz in range(min(gz0, gz1), max(gz0, gz1) + 1):
                wx = gx * CELL - limit
                wz = gz * CELL - limit
                if (wx - p['x']) ** 2 + (wz - p['z']) ** 2 <= rad ** 2:
                    blocked.add((gx, gz))
    return blocked, n


def bfs_reachable(blocked, n, start, goal):
    if start in blocked or goal in blocked:
        return False
    q = deque([start])
    seen = {start}
    while q:
        gx, gz = q.popleft()
        if (gx, gz) == goal:
            return True
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, nz = gx + dx, gz + dz
            if 0 <= nx <= n and 0 <= nz <= n and (nx, nz) not in seen and (nx, nz) not in blocked:
                seen.add((nx, nz))
                q.append((nx, nz))
    return False


def check_maze(name, maze):
    limit = maze.get('map_limit', 16)
    spawn = maze['spawn']
    foods = maze['food_zones']
    blocked, n = build_blocked(maze, limit)

    def to_grid(x, z):
        return (int(round((x + limit) / CELL)), int(round((z + limit) / CELL)))

    ok = True
    sg = to_grid(spawn[0], spawn[2])
    spawn_free = sg not in blocked
    print(f"[{name}] spawn={spawn} livre={spawn_free}")
    ok &= spawn_free
    for f in foods:
        fg = to_grid(f[0], f[2])
        free = fg not in blocked
        path = bfs_reachable(blocked, n, sg, fg) if (spawn_free and free) else False
        print(f"[{name}] food={f} livre={free} caminho={path}")
        ok &= free and path
    return ok


def main():
    data = load()
    assert 'a_treino' in data and 'b_teste' in data, "JSON precisa de a_treino e b_teste"
    a, b = data['a_treino'], data['b_teste']
    same = json.dumps(a['muros'], sort_keys=True) == json.dumps(b['muros'], sort_keys=True)
    print(f"labirintos: A muros={len(a['muros'])} B muros={len(b['muros'])} A!=B: {not same}")
    ok_a = check_maze('A_treino', a)
    ok_b = check_maze('B_teste', b)
    all_ok = ok_a and ok_b and (not same)
    print("A OK, B OK, A!=B" if all_ok else "FALHOU: revise labirintos.json")
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
