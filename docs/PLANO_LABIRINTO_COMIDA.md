# Plano Labirinto + Comida — Enriquecimento (Currículo + Olfato/Fome)

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
- `hunger 0→1` (`hunger_rate=0.002`, ~0.4 em 200 passos), zera ao comer.
- `r *= (1 + hunger_gain*hunger)` com `hunger_gain=0.5`.
- Por que: parado com fome dói mais; comer saciado rende menos → força busca.
- Arquivos: `perception.py:344-345`, `train_food_headless.py:186,232,240`, `main.py:state['hunger']`.
- Validar: `debug_food.py` C5 faminto > saciado.

### L3 — Currículo de distância T8 (perto → longe)
- `food_curriculum=0/1`, `food_start_dist=5.0`, `food_growth=0.15`, `food_limit=12`.
- `max_d(ep) = min(limit, start + growth*ep)` → ~12 em ~47 eps.
- Início do episódio: `respawn_food(cx=spawn, cz=spawn, max_d)`; após comer: respawn perto do verme com mesmo `max_d(ep)`.
- Anti-sorte: nunca nasce dentro de `food_radius+1` (já-comido).
- Arquivos: `config.py:48-53`, `environment.py:food_max_dist/randomize_food_near/eat_and_respawn`, `train_food_headless.py:respawn_food/food_max_dist`.
- Validar: treino curto com e sem currículo compara % encontro no estágio C.

### L4 — Protocolo treino A / teste B (generalização)
- Treina SÓ em A, avalia SEMPRE em B. A≠B garantido por `debug_maze.py`.
- Métricas primárias: `% com encontro`, `passos_ate_comer` (mediana/min/média), `media encontros/ep`.
- Métrica oficial: DoD relativo ≥80% do professor (absoluto em B é loteria: oráculo oscila 60-80%).

## 3. Como rodar (padrão do grupo)

```bash
# Validadores (sem GUI, rápidos)
python debug_maze.py
python debug_food.py
python test_mlp.py

# Treino headless com enriquecimento completo (L1+L2+L3)
python train_food_headless.py --episodes=100 --maze=A --seed=42 --set=food_curriculum=1 --save=pesos_food.json

# Eval oficial em B (nunca visto) + teto do oráculo
python train_food_headless.py --episodes=100 --maze=B --seed=99 --eval --weights=pesos_food.json
python train_food_headless.py --episodes=100 --maze=B --seed=99 --teacher-only

# Com exploração extra (fixes travados Fase 14)
python train_food_headless.py --episodes=100 --maze=A --seed=42 --set=food_curriculum=1 --set=epsilon_greedy=0.1 --set=lambda_c=0.05 --save=pesos_food.json

# Smoke GUI
python main.py --maze=A --episodes=3 --set=food_curriculum=1
python main.py --eval --maze=B --episodes=100
```

## 4. Roteiro de execução (próximos passos)

- [x] L0 — Congelar base: A/B + 11-dim + E1/E2 validados (`debug_maze/food` OK)
- [x] L1 — Olfato travado (`smell_bonus=0.3`, C2 OK)
- [x] L2 — Fome travada (`hunger_gain=0.5`, C5 OK)
- [x] L3 — Currículo travado (`start 5.0 +0.15/ep`, anti-sorte, fallback legado)
- [x] L4 — Rodar 100eps A c/ currículo (seed 42) → eval B (seed 99) → 1 linha em `EXPERIMENTOS.md` (#18)
- [ ] L5 — Ablação: sem currículo vs com currículo vs sem fome (3 runs headless, isola ganho de cada enriquecimento)
- [ ] L6 — Se DoD relativo ≥80% em B: promover a `PLANO_DE_MELHORIAS.md` Fase 14 como atingida; senão, documentar limite

## 5. Critérios de aceite (DoD deste plano)

1. `debug_maze.py` + `debug_food.py` + `test_mlp.py` verdes.
2. Treino A 100eps com `food_curriculum=1` reproduzível (seed 42) com pesos salvos.
3. Eval B 100eps reporta `% encontro` + `passos_ate_comer` + DoD relativo vs professor.
4. Linha #18 em `EXPERIMENTOS.md` com o que mudou → o que aconteceu → conclusão.

## 6. Adiado de propósito (fora deste plano)

Veneno/predador, multi-alimento simultâneo, alimento móvel, RNN/memória, multi-verme, novos labirintos C/D — só após DoD do básico (ver `../README.md` F8).
