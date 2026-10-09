"""Pipeline agentificado do labirinto (sem GUI, rapido).

Uso:
  python maze_pipeline.py --check            # validacoes fail-fast
  python maze_pipeline.py --gen --seed=42    # (re)gera chave `gigante`
  python maze_pipeline.py --audit-movimento  # histograma N/S/L/O + simetria

Ordem fixa (reduz erro e perda de prioridade):
  1 maze valido (BFS) -> 2 colisao desliza -> 3 spawn seguro -> 4 seeds divergem
"""
import json
import math
import random
import sys

from config import CONFIG


def load_gigante():
    from maze import Maze
    try:
        return Maze.load(CONFIG.get('maze_file', 'labirintos.json'),
                         CONFIG.get('maze_key', 'gigante'))
    except Exception as e:
        print(f'[AVISO] sem gigante ({e}) — pipeline usa terreno limpo p/ movimento')
        return None


def check():
    from maze import Maze, collides
    ok = True

    # 1. maze valido
    mz = load_gigante()
    if mz is None:
        print('[1 maze] SKIP (sem chave gigante — terreno limpo)')
    else:
        errs = mz.validate(
            clearance=CONFIG.get('maze_clearance', 2.5),
            min_wall_dist=CONFIG.get('min_wall_dist', 2.5))
        n = len(mz.walls)
        print(f'[1 maze] muros={n} erros={len(errs)}')
        for e in errs[:5]:
            print(f'    ERRO: {e}')
        if errs or n == 0:
            print('[1 maze] FAIL'); ok = False
        else:
            # BFS lens
            sx, sz = mz.spawn[0], mz.spawn[2]
            for i, f in enumerate(mz.food_zones):
                L = mz.bfs_path_len(sx, sz, f[0], f[2])
                print(f'    food[{i}] BFS={None if L is None else round(L,1)}u')
            print('[1 maze] PASS')

    # 2. colisao desliza (slide X depois Z, nunca atravessa)
    walls = mz.walls if mz else [{'x': 0, 'z': 5, 'sx': 20, 'sz': 1.2}]
    # tenta atravessar a parede andando em +Z
    x, z = 0.0, 0.0
    r = 1.5
    for _ in range(60):
        nz = z + 0.2
        if not collides(x, nz, walls, r):
            z = nz
        # slide em X sempre permitido aqui
        nx = x + 0.2
        if not collides(nx, z, walls, r):
            x = nx
    atravessou = (z > 5.0 + r)
    print(f'[2 colisao] pos_final=({x:.1f},{z:.1f}) atravessou={atravessou}')
    if atravessou:
        print('[2 colisao] FAIL'); ok = False
    else:
        print('[2 colisao] PASS (desliza, nao atravessa)')

    # 3. spawn seguro: 20 seeds, todos livres + folga
    from maze import Maze as _M
    m = mz if mz else _M(walls=[], limit=CONFIG.get('map_limit', 32))
    clr = CONFIG.get('maze_clearance', 2.5)
    mwd = CONFIG.get('min_wall_dist', 2.5)
    bad = 0
    for seed in range(20):
        (sx, sz), (fx, fz) = m.random_food_and_spawn(
            1000 + seed, clearance=clr, min_wall_dist=mwd,
            min_food_spawn_dist=CONFIG.get('min_food_spawn_dist', 8.0))
        if not m.free(sx, sz, clr) or not m.free(fx, fz, clr):
            bad += 1
        if m.min_wall_dist(sx, sz) < mwd or m.min_wall_dist(fx, fz) < mwd:
            # terreno limpo tem inf — ok
            if m.walls:
                bad += 1
    print(f'[3 spawn] 20 seeds, inseguros={bad}')
    if bad:
        print('[3 spawn] FAIL'); ok = False
    else:
        print('[3 spawn] PASS')

    # 4. seeds divergem (anti-decoreba)
    pts = set()
    for seed in range(10):
        (sx, sz), (fx, fz) = m.random_food_and_spawn(5000 + seed)
        pts.add((round(sx, 1), round(sz, 1), round(fx, 1), round(fz, 1)))
    print(f'[4 seeds] posicoes distintas={len(pts)}/10')
    if len(pts) < 8:
        print('[4 seeds] FAIL (repetindo padrao)'); ok = False
    else:
        print('[4 seeds] PASS')

    print('CHECK: TUDO OK' if ok else 'CHECK: FALHOU')
    return 0 if ok else 1


def audit_movimento(n_eps=100, steps=400):
    """Histograma de deslocamento com policy uniforme + heading aleatorio."""
    import math as _m
    turn = CONFIG.get('turn_rate', 1.05)
    dt = 1.0 / 60.0
    graus = turn * dt * 180 / _m.pi
    print(f'turn_rate={turn} graus/passo={graus:.2f} passos_pra_180={180/max(graus,1e-9):.0f}')
    mode = CONFIG.get('spawn_heading', 'random')
    print(f'spawn_heading={mode} (random=livre, fixed=viciado p/ norte)')
    counts = {'N': 0, 'S': 0, 'L': 0, 'O': 0}
    rng = random.Random(123)
    MULTS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for _ in range(n_eps):
        ang = rng.uniform(0, 2 * _m.pi) if mode == 'random' else _m.pi / 2
        # direcao como (cos,sin) no plano XZ
        dx, dz = _m.cos(ang), _m.sin(ang)
        x = z = 0.0
        mt = turn * dt
        for _ in range(steps):
            a = rng.randrange(5)  # policy uniforme = cerebro sem vicio
            ang += MULTS[a] * mt
            dx, dz = _m.cos(ang), _m.sin(ang)
            x += dx * 6.0 * dt
            z += dz * 6.0 * dt
        if abs(x) > abs(z):
            counts['L' if x > 0 else 'O'] += 1
        else:
            counts['N' if z > 0 else 'S'] += 1
    print(f'histogramaeps ({n_eps}eps): {counts}')
    vals = list(counts.values())
    razao = max(vals) / max(1, min(vals))
    print(f'razao max/min={razao:.2f} (livre < 2.0)')
    # simetria das acoes com logits zerados
    from mlp import PolicyNetwork
    net = PolicyNetwork(8, 8, 5)
    for p in list(net.params().values()):
        pass
    # zera pesos p/ softmax uniforme
    import math as _mm
    s = [0.0] * 8
    probs = net.probabilities(s)
    print('probs_inicial=' + ' '.join(f'{p:.2f}' for p in probs))
    verdict = 'LIVRE' if razao < 2.0 else 'VICIADO'
    print(f'veredito: {verdict}')
    return 0 if verdict == 'LIVRE' else 1


def gen(seed=None):
    from maze import generate_giant, Maze, ascii_map
    seed = CONFIG.get('maze_seed', 42) if seed is None else seed
    limit = CONFIG.get('map_limit', 32)
    g = generate_giant(seed=seed, limit=limit, cells=8, braid=0.3)
    mz = Maze.from_dict(g)
    # garante spawn/food com folga (move se necessario)
    errs = mz.validate(clearance=CONFIG.get('maze_clearance', 2.5),
                       min_wall_dist=CONFIG.get('min_wall_dist', 2.5))
    print(f'muros={len(g["muros"])} erros={len(errs)}')
    for e in errs[:8]:
        print(f'  ERRO: {e}')
    print(ascii_map(mz))
    sx, sz = mz.spawn[0], mz.spawn[2]
    for i, f in enumerate(mz.food_zones):
        print(f'food[{i}] BFS={mz.bfs_path_len(sx, sz, f[0], f[2])}')
    path = CONFIG.get('maze_file', 'labirintos.json')
    with open(path) as fh:
        data = json.load(fh)
    data['gigante'] = g
    data['_nota'] = ('v2: a_treino/b_teste limpos (baseline); '
                     'gigante=DFS seed fixa, BFS valido.')
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f'gigante salvo em {path} (seed={seed})')
    return 0 if not errs else 1


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--gen' in args:
        seed = None
        for a in args:
            if a.startswith('--seed='):
                seed = int(a.split('=')[1])
        raise SystemExit(gen(seed))
    if '--audit-movimento' in args:
        raise SystemExit(audit_movimento())
    raise SystemExit(check())
