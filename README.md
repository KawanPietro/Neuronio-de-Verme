# Neuronio-de-Verme

Uma simulação 3D educativa em Python que usa **redes neurais** e **aprendizado por reforço** para ensinar um verme virtual a **encontrar alimento no menor tempo** em um terreno limpo — um cenário inspirado em quimiotaxia (seguir gradiente de cheiro).

> O projeto funciona como um "laboratório vivo": você assiste ao verme aprendendo em tempo real, pode posicionar alimentos pela cena e acompanhar a evolução da sua autonomia.

> 🟢 Estado canônico: **terreno limpo 8-dim** (só verme + alimento, sem obstáculos/labirinto), `8→32→16→5`, CSV 13 cols. Baseline: **#35 — PPO 50eps, 56% encontros** (ver `docs/EXPERIMENTOS.md`). História das eras anteriores em `docs/historico/README.md`.

---

## Sumário

- [Conceito](#-conceito)
- [Tecnologias](#-tecnologias)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Como executar](#-como-executar)
- [Controles](#-controles)
- [Como o projeto funciona](#-como-o-projeto-funciona)
- [Aprendizado por currículo](#-aprendizado-por-currículo-curriculum-learning)
- [Métricas e avaliação](#-métricas-e-avaliação)
- [Baseline e diagnóstico honesto](#-baseline-e-diagnóstico-honesto)
- [Roteiro do protótipo](#-roteiro-do-protótipo)
- [A demonstração `DiaNoite.py`](#-a-demonstração-dianoitepy)
- [Histórico das eras](#-histórico-das-eras)
- [Licença / Notas](#-licença--notas)

---

## 🪱 Conceito

O projeto une duas áreas de estudo de forma visual e divertida:

| Área | Aplicação no projeto |
|------|----------------------|
| **Redes neurais** | Um MLP implementado do zero (sem bibliotecas de ML), política `8→32→16→5`. |
| **Aprendizado por reforço (RL)** | O verme recebe recompensas ao se aproximar/comer e ajusta os pesos por PPO/A2C/REINFORCE, DQN ou clonagem (BC). |
| **Aprendizado por currículo** | Um "professor" mostra a direção certa; a autonomia sobe por estágios até o verme decidir sozinho. |
| **Campos de potencial** | A direção "professor" é atração ao alimento + fuga das bordas. |
| **Game engine** | Ambiente 3D interativo em `Ursina`. |

**O objetivo do verme (único):**
- 🟢 **Achar e comer o alimento** → aproximar rende `+`, comer dá bônus, ficar sobre ele rende por permanência; a **fome** multiplica tudo (faminto sente mais).

O verme tem sensores de **olfato** (`smell`), **propriocepção** (direção) e **borda**, um **cérebro** de 8 entradas → 32 → 16 → 5 ações discretas (softmax), e aprende por **backpropagation + PPO/A2C**.

---

## 🧰 Tecnologias

- **Python 3.8+** (testado no 3.12)
- **[Ursina](https://www.ursinaengine.org/)** — engine 3D construída sobre Panda3D
- **Apenas a biblioteca padrão** para a rede neural (`math`, `random`)

---

## 📁 Estrutura do projeto

```
Neuronio-de-Verme/
├── main.py              # GUI: câmera, editor, currículo, update/input
├── config.py            # Todas as constantes (CONFIG canônico 8-dim)
├── perception.py        # Contrato 8-dim + recompensa unimodal (FOOD_DIM=8)
├── mlp.py               # MLP + Policy/Critic/QNetwork + PPO/A2C/DQN/BC + Buffer
├── test_mlp.py          # 16 testes (gradientes, PPO, buffer, save/load, DQN)
├── worm.py              # Corpo do verme (cabeça + segmentos, step sem colisão)
├── environment.py       # Só alimento: place/clear/randomize/eat_and_respawn
├── debug_food.py        # Valida contrato 8-dim (C1–C6, aproximar > afastar)
├── debug_maze.py        # Stub: confirma terreno limpo (0 muros/pedras)
├── train_food_headless.py # Treino/eval sem GUI (--eval, --csv, --bc/--dqn)
├── Verme.py             # HUB: menu treino/avaliação/visualização
├── DiaNoite.py          # Demo separada: ciclo dia/noite (não integrada)
├── labirintos.json      # Esvaziado, RESERVADO ao labirinto gigante
├── episodios.csv        # Log vivo do jogo (13 cols; recriado a cada run)
├── baseline_limpo.csv   # Baseline #35 (PPO 50eps, 13 cols)
├── Rede_Neural.py       # Cérebro LEGADO didático (Fase 1, referência)
├── requirements.txt     # Dependência: ursina
├── docs/
│   ├── EXPERIMENTOS.md         # Tabela de ciência (#35+ vigente; #1–34 histórico)
│   ├── PLANO_DE_MELHORIAS.md   # Rota por fases (Fase 16 atual)
│   ├── PLANO_LABIRINTO_COMIDA.md # Canônico §7 + pendências §8 + gigante §9
│   └── historico/              # CSVs 15-cols + 58 pesos das eras antigas
└── README.md            # Este documento
```

---

## 🚀 Como executar

> Recomenda-se ambiente virtual (venv).

```bash
pip install -r requirements.txt

# GUI — treino 50 episódios (A10/B20/C20), salva pesos.json e fecha
python main.py --episodes=50

# GUI — avaliação (carrega pesos.json, sem treino) / passeio livre
python main.py --eval --episodes=50
python main.py --visual
python Verme.py            # HUB com menu (mesmas opções)

# Headless rápido, sem GUI (ideal p/ experimentos)
python train_food_headless.py --episodes=50 --seed=42 --csv=meu_run.csv
python train_food_headless.py --episodes=50 --seed=42 --dqn --dqn-epsilon=0.2
python train_food_headless.py --episodes=50 --seed=42 --bc

# Validadores (sem GUI, rápidos — devem estar sempre verdes)
python test_mlp.py && python debug_food.py && python debug_maze.py
```

Notas: flags `--maze`/`--seeds`/`--random-maze` foram **removidas** (há um só
terreno; o labirinto gigante as reintroduz); `--set=chave=valor` sobrescreve o CONFIG (chaves removidas como
`difficulty`/`obstacle_*` não fazem nada); `--eval` com `--episodes` **sobrescreve**
`pesos.json` ao fechar — faça backup antes.

---

## 🎮 Controles

### main.py (simulação principal)

| Tecla | Ação |
|-------|------|
| **Botão direito + mouse** | Orbitar a câmera |
| **W / A / S / D** | Mover o foco da câmera |
| **Scroll** | Zoom |
| **R** | Reiniciar o verme e o cérebro |
| **A** | Ligar/desligar o **professor** (autonomia forçada) |
| **P** | Mostrar/ocultar a **grade de setas** da política |
| **S / L** | **Salvar / carregar** os pesos (`pesos.json`) |
| **1** | Alternar modo *colocar ALIMENTO* |
| **3** | Alternar modo *deletar ALIMENTO* |
| **Clique esquerdo** | Executar a ação do modo atual |
| **ESC** | Cancelar o modo ativo; de novo fecha o programa |

### DiaNoite.py

| Tecla | Ação |
|-------|------|
| **Espaço** | Alternar automático / manual |
| **→ ou D** | Avançar o sol (manual) |
| **← ou A** | Recuar o sol (manual) |
| **O / P** | Aumentar / diminuir velocidade (automático) |

---

## 🧠 Como o projeto funciona

### 1. O cérebro: `mlp.py`

Política estocástica com backpropagation do zero + cabeçote softmax:

```
        ENTRADAS (8)               OCULTAS              AÇÕES (softmax)
   ┌───────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │ food_dir (x,z)    │    │                  │    │  esquerda        │  girar −θ
   │ food_dist / smell │───▶│  h1[0]...h1[31]  │───▶│  frente_esquerda │  girar −θ/2
   │ vel (x,z)         │    │        ↓         │    │  frente          │  seguir
   │ borda (x,z)       │    │  h2[0]...h2[15]  │───▶│  frente_direita  │  girar +θ/2
   └───────────────────┘    └──────────────────┘    │  direita         │  girar +θ
                                                     └──────────────────┘
```

**Arquitetura:** actor `8→32→16→5` (~0.9k params) + critic `8→32→16→1`. Alternativas
no headless: `QNetwork` (DQN, `--dqn`) e clonagem supervisionada (`--bc`).

| Classe / método | Função |
|-----------------|--------|
| `MLP` | MLP genérico 3 camadas, `forward`/`backward` camada a camada. |
| `PolicyNetwork` | MLP + softmax → `π(a|s)`; `sample_action()` amostra; `imitate()` (CE); `update_episode()` (REINFORCE/A2C/PPO + `λ·CE` no híbrido). |
| `CriticNetwork` | Estima `V(s)` p/ GAE + loss MSE. |
| `QNetwork` | DQN: TD `r+γ·maxQ'`, rede-alvo, `act()` ε-greedy. |
| `ReplayBuffer` | Guarda até 50 episódios; PPO em mini-batches (código pronto, **OFF por padrão**). |
| `save_brain()` / `load_brain()` | Persiste actor+critic; **recusa dims incompatíveis com aviso** (pesos 11/17-dim não carregam — ver `docs/historico/`). |
| `compute_returns` / `compute_gae` | Retornos `G_t` e advantages GAE. |

### 2. O ambiente: `main.py` + `perception.py` + `environment.py`

**Cenário:** terreno limpo 64×64 (`map_limit=32`), chão de grama, céu, luz que segue o verme. Só existem **verme + 1 alimento por vez** (come → nasce outro perto, 3–5 encontros/ep).

**Sensores** (`get_sensor_inputs()`, estado 8-dim, tudo normalizado):

| # | Feature | Significado |
|---|---------|-------------|
| 0–1 | `food_dir (x,z)` | Vetor unitário até o alimento (0,0 se sem) |
| 2 | `food_dist` | Distância `[0,1]` (`sensor_max_dist=60`) |
| 3 | `smell` | Olfato `1/(1+dist)` — gradiente denso mesmo longe |
| 4–5 | `vel_x/z` | Direção atual `[-1,1]` (propriocepção) |
| 6–7 | `borda_x/z` | Distância à borda `[0,1]` (1=centro, 0=parede) |

**Recompensa** (`calculate_reward()`, por passo, clip `[-1,+1]`):
`progresso (Δdist × 3.0)` + `smell` + `bônus ao comer (5.0)` / `permanecer (0.5)` +
novidade (L9) − repetição de ação, tudo × `(1 + 0.5·fome)`.

**Movimento:** estado → `sample_action` (1 de 5 giros) → mistura com o giro do
professor conforme a autonomia (estágios A/B/C) → move a cabeça a velocidade
constante, limitado ao mapa. Sem colisão (sem obstáculos).

### 3. O ciclo de aprendizagem

```
 Sensores 8-dim ──▶ π(a|s) ──sample──▶ giro + velocidade constante
                         │
                         ▼
 Professor (atração alimento + fuga de borda) ──▶ giro_professor
                         │
                         ▼
 Mistura: giro = lerp(professor, política, autonomia A/B/C)
                         │
                         ▼
 Move ──▶ recompensa ──▶ guarda (s, a, r, professor) ──▶ a cada H=400 passos:
                                              PPO/A2C (GAE, clipping, critic)
```

---

## 📈 Aprendizado por currículo (Curriculum Learning)

Um **professor** (campo de potencial) mostra a direção certa; a autonomia sobe por estágios:

| Estágio | Episódios | Movimento | Treino |
|---------|-----------|-----------|--------|
| **A — Imitação** | 1º–10º | 100% professor | Cross-entropy supervisionada |
| **B — Híbrido** | 11º–30º | autonomia 0→1 | PPO/A2C + `λ·CE`, λ decai |
| **C — Autônomo** | depois | 100% rede | PPO/A2C puro |

- `teacher_action()` converte a direção ideal na ação discreta alvo da imitação.
- Tecla **A** força autonomia total a qualquer momento (teste do verme sozinho).
- Currículo de distância (opt-in `--set=food_curriculum=1`): nasce perto, afasta `+0.3/ep` até o teto.

---

## 📊 Métricas e avaliação

- **HUD**: episódio, estágio, encontros, `passos_ate_comer`, recompensa média rolling, entropia, lr, modo.
- **Grade (P)**: seta por célula = ação mais provável (verde se aponta p/ comida).
- **CSV 13 cols** (`episodios.csv`, `baseline_limpo.csv`): `episodio,estagio,recompensa_total,recompensa_media,retorno_medio,entropia_media,learning_rate,lambda_imitacao,autonomia_forcada,encontros_food,passos_ate_comer,acao_principal,semente`.
- **Métrica primária**: `% com encontro` + `passos_ate_comer` (mediana). **DoD relativo**: ≥80% do professor **em terreno limpo**.
- **Persistência**: `S/L` e save automático no fim de `--episodes`.

---

## 🔬 Baseline e diagnóstico honesto

**#35 (era atual, PPO 50eps seed42 H=400):** 56% (28/50), média 0.78/ep; estágio C
8/20 = 40% (entropia 0.000 — colapso p/ 1 ação); passos-até-1º mediana 163 / min 16.

Leitura honesta: o colapso persiste **sem obstáculos** (ent→0 já no ep5) — o labirinto
não era a causa raiz. Hipótese vigente: limite estrutural do policy-gradient softmax
on-policy com poucos dados (20k passos). Próximo teste barato: `--dqn` e `--bc` no
8-dim. Detalhes e eras antigas em `docs/EXPERIMENTOS.md`.

---

## 🗺️ Roteiro do protótipo

Protótipo = demo GUI jogável + treino headless reproduzível + docs consistentes
(ver `PLANO_DE_MELHORIAS.md` Fase 16). Depois: labirinto gigante 4x área
(`map_limit` 64, especificado em `PLANO_LABIRINTO_COMIDA.md §9`).

- [x] Terreno limpo 8-dim + baseline #35
- [ ] `--dqn` / `--bc` 50eps no 8-dim (decide a trilha)
- [x] Arquivo histórico (`docs/historico/`)
- [x] README reescrito p/ era atual
- [ ] Labirinto gigante (só após P0 zerado)

---

## 🌅 A demonstração `DiaNoite.py`

Mini-projeto **separado** (não integrado): ciclo do sol 0°–360° em 4 fases
(amanhecer → tarde → noite → amanhecer), modos automático/manual. Exemplo didático
de separação de responsabilidades (`apply_cycle(angle)`).

---

## 🕰️ Histórico das eras

- **Luz/chuva** (sensores de luz, chuva, perigo) e **labirinto 11/17-dim A/B**
  (muros, pedras, dificuldades, tecla `O`): código **removido**; provas em
  `docs/historico/` (7 CSVs + 58 pesos) e tabelas #1–34 em `EXPERIMENTOS.md`.
- Números de eras diferentes **não são comparáveis** (outro estado, mapa e CSV).
- `Rede_Neural.py` (Hebbian Fase 1) mantida como referência didática.

---

## 📄 Licença / Notas

Projeto educacional para estudo de redes neurais e aprendizado por reforço. Sinta-se livre para experimentar e modificar — a melhor forma de aprender RL é mexendo no cérebro de um verme. 🪱
