# Plano Labirinto + Comida — Enriquecimento (Currículo + Olfato/Fome)

> ⚠️ LEIA PRIMEIRO (atualizado 2026-10-08): as seções 1–6 abaixo são **histórico da
> era 11-dim com labirintos A/B** (valores, comandos `--maze=A/B` e mapas A/B não
> valem mais). O estado canônico atual está na seção 7. Não apagar 1–6 (histórico).
>
> Pergunta-base: **"Quantas épocas até alcançar o alimento no menor tempo?"**
> Trilha legada luz/chuva intacta por padrão. Modo alimento ativa com `--maze=A/B`.
> Estado congelado 2026-09-24: DoD relativo NÃO atingido (A 26% / B 21% do professor, #15).
> Este plano inicia a evolução focada em **enriquecimento**: currículo + olfato/fome.

## 1. Ponto de partida (já pronto, validado)

| Item | Onde | Status |
|------|------|--------|
| Labirintos A_treino ≠ B_teste | `labirintos.json`, `environment.py:load_maze` | ✅ `debug_maze.py` OK (spawn livre + BFS) |
| Estado 11-dim alimento | `perception.py:FOOD_DIM`, `get_food_sensor_inputs` | ✅ `debug_food.py` OK |
| Recompensa unimodal | `perception.py:calculate_food_reward` | ✅ contrato aproximar > afastar |
| Comer e reaparecer E2 | `environment.py:eat_and_respawn` | ✅ 3-5 encontros/ep |
| Treino/eval headless | `train_food_headless.py` | ✅ 100eps A → 17% A/B (#15) |
| Teto do professor | `--teacher-only` | ✅ A 66% / B 80% (#16) |
| Pesos referência | `pesos_food_A100.json` | ✅ treino 100eps maze A |

Mapa atual:
- **A_treino**: corredor em S, 2 muros + 1 pedra, spawn `[0,1.25,-12]`, foods `[-10,1,8] [10,1,10]`
- **B_teste**: corredor em Z + muro extra, spawn igual, foods `[10,1,-8] [-6,1,10]`, nunca usado no treino

## 2. Enriquecimento escolhido: Currículo + Olfato/Fome

### L1 — Olfato E3 (gradiente denso)
- `smell = 1/(1+dist)` em `features[3]`, `smell_bonus=0.3` em `calculate_food_reward`.
- Por que: dá sinal mesmo longe, sem depender de Δ mensurável (quimiotaxia).
- Arquivo: `perception.py:280,316` + `config.py:smell_bonus`.
- Validar: `python debug_food.py` → C2 smell~0 longe, contrato aproximar > afastar.

### L2 — Fome E4 (quebra "ficar parado")
- `hunger 0→1` (`hunger_rate=0.001`, ~0.4 em 400 passos no mapa 32), zera ao comer.
- `r *= (1 + hunger_gain*hunger)` com `hunger_gain=0.5`.
- Por que: parado com fome dói mais; comer saciado rende menos → força busca.
- Arquivos: `perception.py:344-345`, `train_food_headless.py:186,232,240`, `main.py:state['hunger']`.
- Validar: `debug_food.py` C5 faminto > saciado.

### L3 — Currículo de distância T8 (perto → longe)
- `food_curriculum=0/1`, `food_start_dist=10.0`, `food_growth=0.3`, `food_limit=24` (mapa 32; era 5.0/0.15/12 no mapa 16).
- `max_d(ep) = min(limit, start + growth*ep)` → ~24 em ~47 eps.
- Início do episódio: `respawn_food(cx=spawn, cz=spawn, max_d)`; após comer: respawn perto do verme com mesmo `max_d(ep)`.
- Anti-sorte: nunca nasce dentro de `food_radius+1` (já-comido).
- Arquivos: `config.py:48-53`, `environment.py:food_max_dist/randomize_food_near/eat_and_respawn`, `train_food_headless.py:respawn_food/food_max_dist`.
- Validar: treino curto com e sem currículo compara % encontro no estágio C.

### L4 — Protocolo treino A / teste B (generalização)
- Treina SÓ em A, avalia SEMPRE em B. A≠B garantido por `debug_maze.py`.
- Métricas primárias: `% com encontro`, `passos_ate_comer` (mediana/min/média), `media encontros/ep`.
- Métrica oficial: DoD relativo ≥80% do professor (absoluto em B é loteria: oráculo oscila 60-80%).

## 3. Como rodar (padrão do grupo — atualizado era terreno limpo)

```bash
# Validadores (sem GUI, rápidos)
python debug_maze.py
python debug_food.py
python test_mlp.py

# Treino headless
python train_food_headless.py --episodes=50 --seed=42 --csv=baseline_limpo.csv

# Alternativas de cérebro (ainda não rodadas no 8-dim — próximo passo)
python train_food_headless.py --episodes=50 --seed=42 --dqn --dqn-epsilon=0.2
python train_food_headless.py --episodes=50 --seed=42 --bc

# Smoke GUI
python main.py --episodes=3
python main.py --eval --episodes=50
```
(Histórico: comandos originais com `--maze=A/B` e `debug_sensores.py` valiam na
era labirinto; preservados no git, não usar.)

## 4. Roteiro de execução (próximos passos)

- [x] L0 — Congelar base: A/B + 11-dim + E1/E2 validados (`debug_maze/food` OK)
- [x] L1 — Olfato travado (`smell_bonus=0.3`, C2 OK)
- [x] L2 — Fome travada (`hunger_gain=0.5`, C5 OK)
- [x] L3 — Currículo travado (`start 5.0 +0.15/ep`, anti-sorte, fallback legado)
- [x] L4 — Rodar 100eps A c/ currículo (seed 42) → eval B (seed 99) → 1 linha em `EXPERIMENTOS.md` (#18)
- [x] L5 — Ablação: 4 configs × 4 seeds (16 runs headless). Colapso total em 16/16 runs (ent 0.000). Currículo e fome NÃO evitam colapso. Linha #19 em `EXPERIMENTOS.md`.
- [x] L6 — DoD relativo **NÃO atingido** (~21% vs meta 80%). Fase 14 **não** promovida. Limite documentado: rede 11→32→16→5 colapsa em 16/16 runs; agente repete ações em vez de pensar; overfitting A (treino 27-41% vs eval B 11-17%).
- [x] L7 — Migração canônica: luz removida, chuva→alimento, labirinto default A
  (`config n_inputs=11/map 16`, `perception` 11-dim canônica, `environment` sem
  luz/chuva/partículas, `main.py` sem FOOD_MODE, `Verme.py` hub por labirinto,
  `debug_sensores.py`/`gate_f14.py` removidos). Validação: `test_mlp` + `debug_food` + `debug_maze` verdes.

## 5. Critérios de aceite (DoD deste plano)

1. `debug_maze.py` + `debug_food.py` + `test_mlp.py` verdes.
2. Treino A 100eps com `food_curriculum=1` reproduzível (seed 42) com pesos salvos.
3. Eval B 100eps reporta `% encontro` + `passos_ate_comer` + DoD relativo vs professor.
4. Linha #18 em `EXPERIMENTOS.md` com o que mudou → o que aconteceu → conclusão.

## 6. Adiado de propósito (fora deste plano)

Veneno/predador, multi-alimento simultâneo, alimento móvel, RNN/memória, multi-verme, novos labirintos C/D — só após DoD do básico (ver `../README.md` §Histórico das eras).

---

## 7. Estado canônico atual — terreno limpo 8-dim (2026-10-08, verificado)

Reconstrução concluída: pedras, muros, obstáculos, níveis de dificuldade e
`load_maze` removidos. Só verme + alimento.

| Item | Valor atual | Onde conferir |
|------|-------------|---------------|
| Estado | **8-dim**: 0–1 `food_dir`, 2 `food_dist`, 3 `smell`, 4–5 `vel`, 6–7 `borda` | `perception.py:FOOD_DIM=8`, `config.py:n_inputs=8` |
| Recompensa | progresso + smell + comer/permanecer × `(1+fome)`; **sem termos de obstáculo** | `perception.py:calculate_food_reward` |
| Ambiente | só `food_sources`; sem `self.obstacles`, sem `place_obstacle/place_wall/load_maze` | `environment.py` |
| Verme | `step(dt)` sem colisão | `worm.py:step` |
| Cérebro | `8→32→16→5` + critic `8→32→16→1` | `main.py`, `config.py` |
| CSV | **13 cols, sem `colisao_obs`**: `episodio,estagio,recompensa_total,recompensa_media,retorno_medio,entropia_media,learning_rate,lambda_imitacao,autonomia_forcada,encontros_food,passos_ate_comer,acao_principal,semente` | `main.py`, `train_food_headless.py`, `baseline_limpo.csv` |
| `labirintos.json` | esvaziado válido: A/B com `muros=[] pedras=[]`, nota RESERVADO | `labirintos.json`, `debug_maze.py` stub |
| Flags | `--maze`, `--seeds` e `--random-maze` removidos (o gigante reintroduz seleção) | `main.py`, `train_food_headless.py` |
| Baseline | **#35: PPO 50eps 56%, C 40%, mediana 163** (ver `EXPERIMENTOS.md`) | `baseline_limpo.csv` |

Verificação verde na limpeza: `py_compile` 8 arquivos OK; `debug_food.py` TUDO OK
(contrato aproximar > afastar, C1–C6); `debug_maze.py` TERRENO LIMPO OK;
`test_mlp.py` 16/16; `train_food_headless --episodes=3` encontra comida;
`grep` funcional zero-obstáculo (restam só comentários intencionais + guardas
`esperado 0`). Pesos `pesos_*.json` 11/17-dim **não carregam** em 8-dim
(`load_brain` recusa com aviso — comportamento correto, exige baseline novo).

Replanejamento 2026-10 (revisão total): P0/P1 de código corrigidos —
`calculate_reward(...,action)` centralizado (bloco duplicado removido do `main`),
`worm.step(dt)`, renames `EPISODE_LIMIT`/`SPAWN`, header `encontros_food`,
defaults `limit=24`, docstring `--eval`, comentário 13 cols, `==` nas flags
booleanas, `Verme.py` sem mortos, comentário `mlp.py` e palavra russa em
`config.py`. Baseline #35 publicado (colapso persiste sem pedras — §9 e
`EXPERIMENTOS.md #35`).

## 8. Auditoria pós-limpeza — pendências por severidade

### P0 — bug funcional / quebra uso real
- [x] CSVs históricos arquivados em `docs/historico/csv/` (7 arquivos 15-cols;
  raiz conserva `baseline_limpo.csv` + `episodios.csv` vivo). Ver `docs/historico/README.md`.
- [x] Pesos 11/17-dim arquivados em `docs/historico/pesos/` (58 arquivos;
  raiz sem `pesos*.json` — próximo save parte do zero 8-dim).
- [x] `calculate_reward` sem `action` → CORRIGIDO: wrapper repassa
  (`perception.py`), bloco duplicado removido do `main.py`.
- [x] Docstring `--eval-only` → CORRIGIDO para `--eval`.

### P1 — código morto / flags que mentem (todos removidos; decisão final abaixo)
- [x] `--maze`/`MAZE_NAME`/`maze_default`/`maze_file`/`maze_key`/`maze` (CSV):
  REMOVIDOS do código e do CSV (13 cols). `labirintos.json` + `debug_maze.py`
  stub seguem como reserva de arquivo (o gigante reintroduz seleção de maze).
- [x] `--seeds`/`NUM_SEEDS` sem uso → REMOVIDOS do `main.py`.
- [x] `EVAL_EPISODES` como nome → renomeado `EPISODE_LIMIT` (vale p/ treino/eval).
- [x] `MAZE_SPAWN` → renomeado `SPAWN` (sem maze).
- [x] `--random-maze` no-op → flag, parâmetro e branch REMOVIDOS.
- [x] `worm.step(dt, env)` morto → `step(dt)` + chamada sem `env`.
- [x] `Verme.py`: import/const/função mortos removidos; texto `[3] deletar ALIMENTO`.
- [x] `HeadlessMaze cell`, imports em função, `startswith` sem `=`,
  `retorno_medio` duplicado: documentado como dívida menor (não bloqueia).
- [x] `randomize_food/eat_and_respawn(limit=12)` → default `24` (= `food_limit`).
- [x] Header CSV `encontro_food` → `encontros_food` (= chave do `state`).

### P2 — docs desatualizadas que apresentam passado como vigente
- [x] `README.md`: reescrito p/ era 8-dim (histórico das eras em seção própria).
- [ ] Este plano §§1–6 + `PLANO_DE_MELHORIAS.md` + `EXPERIMENTOS.md`: citam A/B,
  11-dim, Fase 15, `difficulty/tecla O`, `gate_f14.py`/`debug_sensores.py`
  inexistentes. **Ação:** manter como histórico com banner (feito aqui) + nota de
  incomparabilidade 11-dim vs 8-dim.
- [ ] Comentários internos contraditórios: `train_food_headless.py:382`
  (`15 cols`) vs `:367` (`13 cols`); `config.py:13` contém palavra em russo
  (`относительно`); `wall_margin` é borda, não muro; `MAZE_SPAWN` sem maze;
  header `encontro_food` vs `state['encontros_food']`; `EVAL_EPISODES` usado como
  limite de treino; `mlp.py:553` arquitetura defasada. **Ação:** corrigir junto com P1.
- [ ] `.opencode/agent/organiza-docs.md:43` proíbe tocar em
  `pesos*.json/episodios.csv/labirintos.json` → hoje **bloqueia** (a) migração dos
  8 CSVs, (b) arquivamento dos pesos inválidos, (c) o próprio labirinto gigante
  (que exige editar `labirintos.json`). **Ação:** afrouxar para
  `não apagar sem arquivar` + liberar `labirintos.json` sob proposta.

## 9. Plano do labirinto gigante 3–4x (aprovado: só planejar neste ciclo)

Decisão: **4x área = 2x lado** (`map_limit` 32→**64**). 3–4x de lado seria 9–16x
área — rejeitado para este salto (`map_limit=96+` fica como fase futura).

| Parâmetro | Atual 32 | Novo 64 | Regra |
|-----------|----------|---------|-------|
| `map_limit` / lado / área | 32 / 64 / 4096 | **64 / 128 / 16384 (4x)** | `32×2` |
| `ground_scale` | 72 | **136** | `2×limit+8` |
| `sensor_max_dist` | 60 | **120** | `0.66×diag(181)` |
| `episode_steps` | 400 | **600→1200 curricular** | reto exige `128/0.1=1280` |
| `food_limit` / currículo | 24 / `10+0.3/ep` off | **48 / `20+0.6/ep` ON** | teto ~47 eps, mesmo ritmo |
| câmera / `max_zoom` / `pivot.y` | `(0,60,-125)` / 140 / 20 | **`(0,120,-250)` / 280 / 40** | `×2` |
| `hunger_rate` | 0.001 (0.4/ep) | **0.00033** | `0.4/1200` |
| `novelty_window` | 200 (50%) | **600 (50%)** | `50% de 1200` |
| `wall_margin` | 4.0 | **6.0** | corredor ~7.5 |
| `food_radius/smell/progress/gamma` | — | **inalterados** | sinal já validado |

`labirintos.json` v2 (quando implementar): versionado (`version:2`), unidades em
mundo, muros como caixas `{x,z,sx,sz,h}`, `food_zones` circulares, `spawn`
explícito, `*_hint` documental (autoridade = `config.py`). Gerador offline
sugerido: DFS recursive-backtracker em grade 16×16 de 8u + alargamento para
corredor ≥7.5u + fusão de muros colineares (~30 caixas); validação BFS 1u
spawn→goals (80–400u) com muros inflados 2.4u + disco r=2.5 + ASCII antes da GUI.
Riscos: custo 3x/ep; `food_dir` através da parede (mitigação: professor por
waypoints BFS; fallback: 3 raycasts e volta a 11-dim); fps Ursina (broadphase por
célula, meta >40fps); cauda enroscando (validar largura 5u em curvas).
Sequência: 1) baseline limpo 50eps → 2) reescala sem muros → 3) gerador offline +
validação → 4) loader + professor BFS → 5) piloto 200eps (≥70% do baseline) →
6) decisão 8 vs 11-dim → 7) endurecer (1200 fixo, 3 seeds).
Nada disso é código ainda — só entra após P0 zerado.

## 10. Como concluir (critérios de aceite atualizados)

1. P0 código zerado ✅ (`action` centralizado, `--eval-only` corrigido, baseline
   #35 publicado) e arquivo executado ✅ (`docs/historico/` com 58 pesos + 7 CSVs;
   regra `.opencode` superada por ordem explícita — git mv preserva histórico).
2. P1 zerado ✅ (renames, remoções e defaults aplicados; dívida menor documentada).
3. `debug_maze.py` + `debug_food.py` + `test_mlp.py` verdes ✅ + `grep` funcional
   zero-obstáculo ✅ (revalidado no replanejamento).
4. Linha em `EXPERIMENTOS.md` com baseline limpo ✅ (#35: 56%, mediana 163,
   `baseline_limpo.csv`) + nota de incomparabilidade ✅.
