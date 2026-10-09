#!/usr/bin/env python3
# Verme.py — HUB central do projeto Neurônio de Verme (terreno limpo + alimento)
#
# Permite transitar entre todos os modos sem decorar flags:
#   [1] Treino      — aprende do zero em terreno limpo (currículo A/B/C)
#   [2] Avaliação   — testa pesos salvos, sem treino
#   [3] Visualização — passeio livre, sem treino (só observar)
#   [4] Apresentação — como o visual, mas HUD só didático (p/ apresentar)
# Teclas em jogo: A (professor), P (grade), R/S/L, 1/3 editor, ESC
# (na Apresentação, A/P/S/L/1/3 ficam desativadas p/ manter a tela limpa)
#
# Rode com:  python Verme.py   ou   python main.py [flags]

import subprocess
import sys
import os

def banner():
    print("\n===============================================")
    print("  NEURONIO DE VERME — HUB (Verme.py)")
    print("===============================================")
    print("  Terreno limpo + alimento 8-dim | Acoes: 5 | Cerebro: 8->32->16->5 + Critic")
    print("  (labirinto gigante futuro: 3-4x maior, a planejar)")

def menu():
    print("\n--- MODOS ---")
    print("  [1] TREINO        — aprende em terreno limpo, salva pesos.json")
    print("  [2] AVALIACAO     — testa pesos salvos, sem treino")
    print("  [3] VISUALIZACAO  — passeio livre, sem treino, sem professor")
    print("  [4] APRESENTACAO  — só o verme + didático (p/ apresentar o trabalho)")
    print("  [5] INFO          — ver fases e controles")
    print("  [0] SAIR")
    try:
        raw = input("  Escolha [0-5]: ")
    except EOFError:
        print("\n  [sem entrada — iniciando VISUALIZACAO direto]")
        return "3"
    raw = raw.strip()
    if ".py" in raw or ":/" in raw or ":\\" in raw:
        print(f"  (ignorado input invalido: {raw!r})")
        return ""
    return raw

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
    base = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(base, "main.py")
    if not os.path.exists(main_py):
        print(f"  ERRO: nao achei {main_py}")
        return
    cmd = [sys.executable, main_py] + args
    print(f"\n> {' '.join(cmd)}  (cwd={base})")
    try:
        result = subprocess.run(cmd, cwd=base)
        if result.returncode != 0:
            print(f"  [!] main.py saiu com codigo {result.returncode}")
            print("  Dica: rode direto no terminal para ver o erro:")
            print(f"    python main.py {' '.join(args)}")
    except KeyboardInterrupt:
        print("\n  Interrompido.")
    except Exception as e:
        print(f"  ERRO ao iniciar main.py: {e}")
        print("  Tente rodar direto: python main.py " + " ".join(args))
    print("\n  [Voltando ao HUB Verme.py — pressione ENTER]")
    try:
        input()
    except:
        pass

def info():
    print("""
  OBJETIVO: encontrar o alimento em terreno limpo no menor tempo.

  FASES:
   0 Higiene | 1 Sensores 8-dim | 2 MLP+Policy | 3 REINFORCE | 4 Curriculo A/B/C
   5 HUD/CSV | 6 Robustez | 7 Recompensa | 8 A2C | 9 PPO | 13 Replay Buffer
   F8 Reformulacao alimento + olfato/fome + curriculo de distancia (sem obstáculos)

  CONTROLES EM JOGO:
   Botao direito + mouse  orbitar | WASD mover foco | Scroll zoom
   R reiniciar | A professor on/off | P grade | S/L salvar/carregar | ESC sair
   [1] colocar ALIMENTO | [3] deletar ALIMENTO

   TERRENO: mapa 32 limpo (só verme + alimento). Labirinto gigante 3-4x a planejar.
   EQUIVALENTES diretos (sem HUB):
    python main.py --episodes=100 --set=food_curriculum=1
    python main.py --eval --episodes=100
    python main.py --visual
    python main.py --present
      """)

def main_loop():
    fails = 0
    while True:
        banner()
        print("  Terreno: LIMPO (só verme + alimento)")
        print("  Dica: digite 1-4,5 e ENTER. Ou rode direto: python Verme.py --present")
        ch = menu()
        if ch == "":
            fails += 1
            if fails >= 3:
                print("  Muitas entradas invalidas — iniciando VISUALIZACAO")
                run_main(["--visual"])
            continue
        fails = 0
        if ch == "1":
            eps = ask_int(f"  Episodios de treino [default 100]: ", default=100)
            run_main([f"--episodes={eps}"])
        elif ch == "2":
            eps = ask_int(f"  Episodios de avaliacao [default 50]: ", default=50)
            run_main(["--eval", f"--episodes={eps}"])
        elif ch == "3":
            print("  Visualizacao: sem --episodes = roda infinito ate ESC")
            raw = input("  Episodios (ENTER=infinito): ").strip()
            if raw == "":
                run_main(["--visual"])
            else:
                try:
                    eps = int(raw)
                    run_main(["--visual", f"--episodes={eps}"])
                except:
                    run_main(["--visual"])
        elif ch == "4":
            print("  Apresentacao: HUD só didático, sem números de treino.")
            run_main(["--present"])
        elif ch == "5":
            info()
            input("  ENTER para voltar: ")
        elif ch == "0":
            print("  Ate logo!")
            break
        else:
            print("  Opcao invalida.")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        print(f"Verme.py repassando args para main.py: {sys.argv[1:]}")
        run_main(sys.argv[1:])
        sys.exit(0)
    try:
        main_loop()
    except Exception as e:
        print(f"\n[ERRO no HUB] {e}")
        import traceback; traceback.print_exc()
        input("Pressione ENTER para sair...")
