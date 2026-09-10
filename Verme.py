#!/usr/bin/env python3
# Verme.py — HUB central do projeto Neurônio de Verme
#
# Permite transitar entre todos os modos sem decorar flags:
#   [1] Treino      — aprende do zero (16 fases, currículo A/B/C)
#   [2] Avaliação   — testa pesos salvos (professor desligado)
#   [3] Visualização — passeio livre, sem treino (só observar)
#   [4] Dificuldade — troca o nível de obstáculos (0 LIVRE → 3 DIFICIL)
# Teclas em jogo: O (cicla dificuldade), A (professor), P (grade), R/S/L, ESC
#
# Rode com:  python Verme.py   ou   python main.py [flags]

import subprocess
import sys
import os

from config import CONFIG

LVL = CONFIG['difficulty_levels']

def banner():
    print("\n===============================================")
    print("  NEURONIO DE VERME — HUB (Verme.py)")
    print("===============================================")
    print("  Projeto 16 fases: do sensor ao PPO+buffer")
    print("  Estados: 11 dims | Ações: 5 | Cérebro: 8→16→5 + Critic")
    print("  Dificuldade atual (CONFIG):", CONFIG['difficulty'], "-", LVL[CONFIG['difficulty']]['label'])

def menu():
    print("\n--- MODOS ---")
    print("  [1] TREINO        — aprende (curriculo A10/B20/C..), salva pesos.json")
    print("  [2] AVALIACAO     — testa pesos salvos, sem treino")
    print("  [3] VISUALIZACAO  — passeio livre, sem treino, sem professor")
    print("  [4] DIFICULDADE   — escolher nivel de obstaculos")
    print("  [5] INFO          — ver fases e controles")
    print("  [0] SAIR")
    return input("  Escolha [0-5]: ").strip()

def ask_difficulty():
    print("\n--- DIFICULDADE ---")
    for k in sorted(LVL):
        print(f"  [{k}] {LVL[k]['label']:7s} — {LVL[k]['n_obstacles']} pedras x{LVL[k]['obstacle_scale']}")
    raw = input("  Escolha dificuldade [0-3] (ENTER=cancelar): ").strip()
    if raw == "":
        return None
    try:
        d = int(raw)
        if d in LVL:
            return d
    except:
        pass
    print("  Invalido.")
    return None

def ask_int(msg, default=None):
    s = input(msg).strip()
    if s == "" and default is not None:
        return default
    try:
        return int(s)
    except:
        print("  Invalido, usando default", default)
        return default

def run_main(args):
    cmd = [sys.executable, "main.py"] + args
    print(f"\n> {' '.join(cmd)}")
    # Usa o mesmo Python/venv que rodou Verme.py
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n  Interrompido.")
    print("\n  [Voltando ao HUB Verme.py — pressione ENTER]")
    try:
        input()
    except:
        pass

def info():
    print("""
  FASES:
   0 Higiene | 1 Sensores 11-dim | 2 MLP+Policy | 3 REINFORCE | 4 Curriculo A/B/C
   5 HUD/CSV | 6 Robustez | 7 Recompensa | 8 A2C | 9 PPO | 13 Replay Buffer | 15 Obstaculos

  CONTROLES EM JOGO:
   Botao direito + mouse  orbitar | WASD mover foco | Scroll zoom
   R reiniciar | O cicla dificuldade (0→3) | A professor on/off | P grade | S/L salvar/carregar | ESC sair

  LABORATORIO: 72x72, mapa 32, 4 niveis (tecla O ou --difficulty=N)
  EQUIVALENTES diretos (sem HUB):
   python main.py --episodes=200 --difficulty=1
   python main.py --eval --episodes=50 --difficulty=2
   python main.py --visual --difficulty=0
    """)

def main_loop():
    difficulty = CONFIG['difficulty']
    while True:
        banner()
        print(f"  Ultima dificuldade escolhida: {difficulty} ({LVL[difficulty]['label']})")
        ch = menu()
        if ch == "1":
            d = ask_difficulty()
            if d is not None:
                difficulty = d
            eps = ask_int(f"  Episodios de treino [default 100]: ", default=100)
            run_main([f"--episodes={eps}", f"--difficulty={difficulty}"])
        elif ch == "2":
            d = ask_difficulty()
            if d is not None:
                difficulty = d
            eps = ask_int(f"  Episodios de avaliacao [default 50]: ", default=50)
            run_main(["--eval", f"--episodes={eps}", f"--difficulty={difficulty}"])
        elif ch == "3":
            d = ask_difficulty()
            if d is not None:
                difficulty = d
            print("  Visualizacao: sem --episodes = roda infinito ate ESC")
            raw = input("  Episodios (ENTER=infinito): ").strip()
            if raw == "":
                run_main(["--visual", f"--difficulty={difficulty}"])
            else:
                try:
                    eps = int(raw)
                    run_main(["--visual", f"--episodes={eps}", f"--difficulty={difficulty}"])
                except:
                    run_main(["--visual", f"--difficulty={difficulty}"])
        elif ch == "4":
            d = ask_difficulty()
            if d is not None:
                difficulty = d
                print(f"  Dificuldade agora {difficulty} ({LVL[difficulty]['label']}) — sera usada no proximo run")
        elif ch == "5":
            info()
            input("  ENTER para voltar: ")
        elif ch == "0":
            print("  Ate logo!")
            break
        else:
            print("  Opcao invalida.")

if __name__ == "__main__":
    # Se chamado com args, repassa direto para main.py (compat: python Verme.py --eval ...)
    if len(sys.argv) > 1:
        # Ex: python Verme.py --eval --episodes=20  -> python main.py --eval --episodes=20
        print(f"Verme.py repassando args para main.py: {sys.argv[1:]}")
        run_main(sys.argv[1:])
        sys.exit(0)
    main_loop()
