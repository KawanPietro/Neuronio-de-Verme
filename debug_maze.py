"""
Valida labirintos: terreno limpo (a_treino/b_teste vazios) + gigante (se existir).

  python debug_maze.py

Verde = baseline intacto E (sem gigante OU gigante valido com BFS).
"""
import json

from config import CONFIG

MAZE_FILE = CONFIG.get('maze_file', 'labirintos.json')


def main():
    with open(MAZE_FILE) as f:
        data = json.load(f)
    ok = True
    for key in ('a_treino', 'b_teste'):
        maze = data.get(key, {})
        n_muros = len(maze.get('muros', []))
        n_pedras = len(maze.get('pedras', []))
        print(f"[{key}] muros={n_muros} pedras={n_pedras} (esperado 0 em terreno limpo)")
        if n_muros != 0 or n_pedras != 0:
            print("FALHOU: terreno deveria estar limpo")
            ok = False
    if 'gigante' in data:
        from maze import Maze
        try:
            mz = Maze.from_dict(data['gigante'])
        except Exception as e:
            print(f"[gigante] FALHOU ao carregar: {e}")
            return 1
        errs = mz.validate(
            clearance=CONFIG.get('maze_clearance', 2.5),
            min_wall_dist=CONFIG.get('min_wall_dist', 2.5))
        print(f"[gigante] muros={len(mz.walls)} erros={len(errs)}")
        for e in errs[:8]:
            print(f"  ERRO: {e}")
        sx, sz = mz.spawn[0], mz.spawn[2]
        for i, fz in enumerate(mz.food_zones):
            L = mz.bfs_path_len(sx, sz, fz[0], fz[2])
            print(f"  food[{i}] BFS={None if L is None else round(L,1)}u")
        if errs:
            print("FALHOU: gigante invalido (spawn/food em parede ou sem BFS)")
            return 1
        print("debug_maze: LIMPO + GIGANTE OK")
    else:
        print("debug_maze: TERRENO LIMPO OK (gigante ainda nao gerado)")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
