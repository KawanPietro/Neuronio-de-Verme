# Plano de Melhorias — Verme Autônomo

> 🟢 **ESTADO ATUAL (2026-10-08, era terreno limpo 8-dim):** só verme + alimento,
> sem obstáculos/labirinto/dificuldade, CSV 13 cols, `8→32→16→5`. Baseline novo:
> **#35 (PPO 50eps: 56% encontros, C 40%, colapso persiste sem pedras)**.
> ❄️ Abaixo, tudo marcado HISTÓRICO refere-se às eras luz/chuva, labirinto 11-dim
> e mapa 32 com muros — números incomparáveis, preservados como prova do caminho.
> Veredito antigo "treinos longos congelados" **não vale** p/ era atual.
> Canônico vigente: `PLANO_LABIRINTO_COMIDA.md §7`.

> Roteiro de evolução do projeto para chegar a um **verme 100% autônomo** (decide sozinho, sem professor), **didático e funcional**.
>
> Decisões de escopo alinhadas:
> - **Cérebro**: Puro Python do zero (sem NumPy/PyTorch) — cada equação fica visível.
> - **Autonomia final**: rede neural decide 100% das ações; o professor só existe nas fases de treino.
> - **Comportamento (vigente)**: **alimento** (objetivo único) em terreno limpo,
>   estado 8-dim (`food_dir/food_dist/smell` + `vel/borda`), sem labirinto.
>   (Trilhas luz/chuva e labirinto A/B: removidas — ver Apêndice A e histórico.)
> - **Entregável**: este documento de rota por etapas + protótipo definido na Fase 16.

---

## Índice de fases

| Fase | Nome | Objetivo principal | Esforço | Status |
|------|------|--------------------|---------|--------|
| 0 | Higiene do código | Base limpa e configurável | M | ✅ concluída |
| 1 | Contrato de comportamento | Sensores 8-dim + recompensa por progresso | M | ✅ concluída |
| 2 | Cérebro de verdade | MLP com backprop + política estocástica | M | ✅ concluída |
| 3 | REINFORCE com baseline | Policy gradient real → convergência | S | ✅ concluída |
| 4 | Currículo de autonomia | Professor → piloto automático | M | ✅ concluída |
| 5 | Métricas e memória | Medir, salvar, visualizar aprendizado | S | ✅ concluída |
| 6 | Robustez e calibração | Tuning, exploração, edge cases | S | ✅ concluída |
| 7 | Engenharia de recompensa | Sinais fortes, anti-colapso, multi-seed | M | ✅ concluída |
| 8 | Actor-Critic (A2C) | Critic V(s) + GAE advantage | M | ✅ concluída |
| 9 | PPO | Clipped surrogate objective | M | ✅ concluída |
| 10 | Treino longo | 500+ episódios com PPO + semente fixa | S | ✅ parcial (17-dim 419/500 → 12.6% C, colapso) |
| 11 | Expansão de rede | Arquitetura maior (11→32→16→5, 1.7k) | M | ✅ parcial (100 eps → 0%, precisa mais dados) |
| 12 | Espaço de estado expandido | ⏪ REVERTIDA: 17-dim abandonada; canônico 8-dim (ver §Fase 12) | M | HISTÓRICO |
| 13 | Replay Buffer (off-policy) | Reutilizar dados de episódios passados (código OFF por padrão) | L | ✅ concluída (código) — ver nota |
| 14 | Avaliação e_DoD | Validar ≥80% em 100+ eps com multi-seed | M | ❄️ congelada (legado 10% + alimento 21–26% do prof; DoD não atingido, #13/#15–16) |
| 15 | Obstáculos e níveis | ⏪ REMOVIDA na reconstrução terreno-limpo | M | Apêndice A (histórico) |
| 16 | Terreno limpo + protótipo | Baseline 8-dim, estabilidade docs, labirinto gigante | M | 🟢 ATUAL (ver §Fase 16) |

> Estimativa de esforço: S = 1 sessão, M = 2-4, L = 4+.

**Critério de conclusão (Definition of Done):**
1. `teacher_influence = 0` de forma permanente. ✅ (vale em qualquer era)
2. Era terreno limpo 8-dim, DoD relativo ≥80% do professor em terreno limpo
   (treino/eval no mesmo mundo limpo; sem A/B até o labirinto gigante). Baseline: **#35 = 40% no C (longe da meta)**.
   Histórico: legado ≥80% chegada à chuva ❌ **melhor 40% (PPO)**; alimento
   11-dim relativo ≥80% ❌ **A 26% / B 21% (#15)** — números da era com muros, incomparáveis.
3. Jogo roda em tempo real com métricas. ✅

---

## Diagnóstico: por que estamos em 40%?

### Fatores identificados

#### 1. Treino curto demais (CAUSA PRINCIPAL)

O algoritmo atual usa **on-policy** (REINFORCE/A2C/PPO). Cada episódio gera dados que são descartados após o update. Para convergir, on-policy precisa de **milhões de passos de interação**.

Nosso atual:
- 100 episódios × 200 passos = **20.000 passos** totais
- Estágio C (autônomo): 70 × 200 = **14.000 passos** efetivos
- Algoritmos modernos (PPO/A2C) precisam de **1M-10M passos** para domínios simples

Estamos **100-500x** abaixo do necessário.

#### 2. Dados descartados a cada episódio

On-policy: cada batch de dados é usado **uma única vez** e jogado fora. O agente não "relembra" situações passadas. Com cenas aleatórias a cada episódio, a variância é enorme e a convergência lenta.

#### 3. Rede neural subdimensionada (HISTÓRICO luz/chuva; atualizado p/ 8-dim)

- Era luz/chuva: `8 → 16 → 5` = **229 pesos** (8×16+16+16×5+5).
- Era atual: `8 → 32 → 16 → 5` ≈ **0.9k pesos no actor** (8×32+32+32×16+16+16×5+5 = 901)
  + critic `8→32→16→1` ≈ 0.8k. Suficiente p/ terreno limpo (sem desvio a representar);
  se o labirinto gigante exigir, reavaliar (ver Fase 16).

#### 4. Espaço de estado (vigente: 8 features de alimento, sem luz/chuva/obs)

| Feature | O que é |
|---------|---------|
| `food_dir.x/z` | direção XZ unitária até o alimento |
| `food_dist` | distância normalizada [0,1] |
| `smell` | olfato `1/(1+dist)` — gradiente denso |
| `vel_x/z` | direção atual (propriocepção, [-1,1]) |
| `borda_x/z` | distância à borda [0,1] (1=centro, 0=parede) |

**Não temos (nem precisamos em terreno limpo)**: obstáculos, ângulos luz/chuva,
luz/perigo. Se o gigante exigir desvio, sensores de parede entram como hipótese nova.

#### 5. Recompensa ruim para credit assignment

O reward é por passo (200 passos/episódio), mas o agente precisa lembrar: "há 150 passos, eu estava virando à esquerda e isso deu certo". Com advantage puro (A2C/PPO), a propagção de crédito ao longo de 200 passos é fraca — o sinal se dilui.

#### 6. Termoção agressiva

- `lr_decay = 0.995`: após 100 eps, lr cai para 0.05 × 0.995^100 ≈ **0.03** (40% do original)
- `temperature_decay = 0.995`: temperatura vai de 1.0 → ~0.3 rapidamente
- O agente "congela" antes de convergir

#### 7. Avaliação confunde treino com teste

- `--eval` usa `min_temperature = 0.3` (política semi-greedy)
- Mas o treino termina com temperatura ~0.3 também
- Não há distinção clara entre "explorando" e "explotando"

### Resumo das causas

| Causa | Impacto | Solução |
|-------|---------|---------|
| Treino curto (20k passos) | **ALTO** | Fase 10: treinar 500+ eps |
| Dados descartados | **ALTO** | Fase 13: replay buffer |
| Rede pequena (229 params) | **MÉDIO** | Fase 11: rede maior |
| Estado pobre (8 feats) | **MÉDIO** | Fase 12: features extras |
| Reward diluído | **MÉDIO** | Fase 12: reward shaping melhorado |
| Termoção agressiva | **BAIXO** | Fase 10: lr/temperature menos agressivos |

---

## Convenções do projeto

### Convencão de gradiente
- `θ ← θ + lr·grad` (ascenso) → descida na CE = `+∇logπ`
- REINFORCE: `advantage · ∇logπ(a)` + `entropy_coef · ∇H`
- No teste `L = −logπ`: usa `[-g for g in grad_log_prob_z]`

### Nomes de parâmetros do MLP
- Pesos: `w_input_hidden`, `bias_hidden`, `w_hidden_output`, `bias_output`
- Gradientes: `grad_w_input_hidden`, `grad_bias_hidden`, `grad_w_hidden_output`, `grad_bias_output`
- Métodos: `params()`, `set_params()`, `grads()`, `zero_grad()`, `apply_gradients()`, `accumulate_grads()`, `apply_accumulated()`

### Tuplas de episódio
- 4 elementos: `(sensors, action, reward, acao_professor)`
- `update_episode` desempacota com `(item + (None, None))[:4]`

### Prints ASCII-safe
- Usar `−` → não, usar `-` (ASCII)
- Usar `→` → não, usar `->` (ASCII)

### Arquivos importantes (era terreno limpo 8-dim)
- `config.py` — todas as constantes (`food_*`, `n_inputs=8`, `food_curriculum`,
  `wall_margin` = BORDA, `buffer_*` OFF por padrão, `maze_*` reservados)
- `mlp.py` — PolicyNetwork + CriticNetwork + QNetwork + REINFORCE/A2C/PPO/DQN/BC
  (+ whitening, `--bc/--dqn` só no headless)
- `main.py` — loop do jogo, currículo A/B/C, HUD, editor alimento
  (professor atração+fuga de borda, CSV 13 cols `encontros_food/passos_ate_comer`)
- `perception.py` — contrato canônico 8-dim (`FOOD_DIM=8`: food/smell/vel/borda;
  olfato, fome, novidade; sem obstáculos)
- `worm.py` — corpo do verme (`step(dt)` sem colisão)
- `environment.py` — só `food_sources` + randomização/currículo proximal
  (`eat_and_respawn`, `food_max_dist`; sem `load_maze`)
- `test_mlp.py` — 16 testes (gradientes, A2C, PPO, buffer, save/load, DQN)
- `debug_food.py` / `debug_maze.py` / `train_food_headless.py` — contrato alimento
  8-dim, stub terreno limpo, treino/eval headless (`--eval`, `--weights/--save`,
  `--set`, `--csv`, `--bc/--dqn/--dqn-epsilon`)
- `labirintos.json` — esvaziado válido, RESERVADO ao labirinto gigante
- `EXPERIMENTOS.md` — tabela de ciência (eras #1–34 histórico; #35+ vigente)
- `PLANO_LABIRINTO_COMIDA.md` — canônico vigente §7 + pendências §8 + gigante §9
- `episodios.csv` / `baseline_limpo.csv` — métricas por episódio, 13 cols
  (CSV antigos 15 cols com `colisao_obs`: históricos, não comparar)
- `pesos.json` — cérebro salvo (actor + critic 8-dim; pesos antigos 11/17-dim
  recusados com aviso — treino novo obrigatório)

### Como rodar (era terreno limpo 8-dim)
```bash
# Treino com limite
python main.py --episodes=50

# Headless rápido, mesma pergunta-base, sem GUI
python train_food_headless.py --episodes=50 --seed=42 --csv=baseline_limpo.csv

# Alternativas de cérebro no headless (DQN/BC ainda válidos no 8-dim)
python train_food_headless.py --episodes=50 --seed=42 --dqn --dqn-epsilon=0.2
python train_food_headless.py --episodes=50 --seed=42 --bc

# Avaliacao (carrega pesos.json, sem treino)
python main.py --eval --episodes=50

# Override de CONFIG
python main.py --episodes=50 --set=learning_rate=0.05 --set=gamma=0.95

# Testes
python test_mlp.py
python debug_food.py
python debug_maze.py

# Compilacao
python -m py_compile config.py mlp.py main.py perception.py worm.py environment.py
```
Notas: flags `--maze`/`--seeds`/`--random-maze` removidas; `--set=difficulty=N`
é chave morta; `debug_sensores.py` foi removido (usar `debug_food.py`); `--eval`
sobrescreve `pesos.json` ao fechar com `--episodes`.

---

## Fase 10 — Treino longo (HISTÓRICO 17-dim; protocolo a repetir no 8-dim)

**Objetivo:** aumentar drasticamente a quantidade de dados de treino para que o PPO converge.

**Hipótese:** com 500+ episódios, o advantage de8-12 passos do PPO tem tempo suficiente para propagar o sinal de recompensa e o agente aprende a navegar.

**Tarefas:**
- [x] Rodar PPO com `--episodes=500` (seed fixa) — parcial 419/500 em 60min, timeout no ep417 (ver `EXPERIMENTOS.md #14`)
- [ ] Rodar PPO com `--episodes=1000` (seed fixa)
- [x] Reduzir `lr_decay` (0.998 em vez de 0.995) — já em `config.py`
- [x] Reduzir `temperature_decay` (0.998 em vez de 0.995) — já em `config.py`
- [x] Aumentar `min_temperature` para 0.2 — já em `config.py`
- [ ] Avaliar os pesos finais com `--eval --episodes=100` — pendente (pesos do run parcial não salvos)
- [ ] Comparar com baseline de40% (Fase 9)

**Resultado parcial 2026-09 (17-dim, dif FACIL):** 389eps estágio C → 49/389 seguro (12.6%) vs 10% com 35eps. Ganho marginal; colapso persiste.

**Métrica de sucesso:** ≥50% (melhoria de +10pp sobre Fase 9) — **não atingida**.

**Parâmetros sugeridos:**
```python
'lr_decay'       : 0.998,   # lr mais estável
'temperature_decay': 0.998,  # explora mais tempo
'min_temperature' : 0.2,    # sempre um pouco de exploração
```

---

## Fase 11 — Expansão de rede

**Objetivo:** aumentar a capacidade da rede neural para representar políticas mais complexas.

**Hipótese:** com 3 camadas (8→32→16→5) e ~0.9k parâmetros no actor, a rede pode capturar relações não-lineares mais sutis entre sensores e ações.

**Tarefas:**
- [x] Adicionar camada intermediária ao MLP (8→32→16→5) — `mlp.py` 3 camadas, numeric grad 1e-7
- [x] Adaptar `forward/backward` para 3 camadas + `n_hidden2` em Policy/Critic
- [x] Atualizar `PolicyNetwork` e `CriticNetwork` (8→32→16→1)
- [x] Testar com `test_mlp.py` (16/16 OK) + deep forward/backward
- [x] Treinar 100 eps com rede nova (dif 1, 11→32→16→5) — 0% (colapso esquerda 200)
- [ ] Treinar 500 eps com rede nova + multi-seed — rede maior precisa 2-3× mais dados
- [ ] Comparar com rede de 2 camadas (Fase 10: 15% shallow vs 0% deep em 100 eps)

**Resultado:** rede profunda (+13% params → 1.7k) não evita colapso com poucos dados; precisa Fase 12 (estado 16) + treino 500+.

---

## Fase 12 — Espaço de estado expandido ⏪ REVERTIDA (17-dim abandonada; canônico 8-dim)

> Esta fase foi concluída em código na era labirinto e **revertida** na
> reconstrução terreno-limpo (2026-10-08): `obs_*` e `angulo_*` removidos,
> `debug_sensores.py` deletado. Tabela abaixo preservada como histórico.
> Canônico vigente: 8-dim (`food_dir/food_dist/smell` + `vel/borda`).

**Objetivo (histórico):** dar ao agente informação mais rica sobre si mesmo e sobre o mundo.

**Hipótese:** com6 features extras (17 total = 11 da Fase 15 + 6), o agente pode aprender a "navegar" (não só reagir) — considerando velocidade, bordas, e ângulo relativo.

**Novas features (17-dim = 11 + 6):**

| Feature | Descrição |
|---------|-----------|
| `luz_dir.x/z` | direção XZ unitária para luz |
| `luz_dist` | distância normalizada [0,1] |
| `luz_perigo` | 1 se dentro do arrival_radius |
| `chuva_dir.x/z` | direção XZ unitária para chuva |
| `chuva_dist` | distância normalizada [0,1] |
| `pulso_chuva` | vibração (tato) |
| `obs_dir.x/z` | direção XZ unitária para obstáculo (Fase 15) |
| `obs_dist` | distância normalizada [0,1] até obstáculo (Fase 15) |
| **NOVO:** `vel_x/z` | direção atual X/Z (propriocepção, [-1,1]) |
| **NOVO:** `borda_x/z` | distância à borda X/Z normalizada [0,1] (1=centro, 0=parede) |
| **NOVO:** `angulo_luz` | ângulo relativo direção->luz / pi, em [-1,1] (0=alinhado) |
| **NOVO:** `angulo_chuva` | ângulo relativo direção->chuva / pi, em [-1,1] (0=alinhado) |

**Implementado:**
- [x] Expandir `get_sensor_inputs()` para17 features (`perception.py`, com `_wrap_pi`)
- [x] Atualizar `n_inputs=17` no CONFIG (actor 1189 + critic 1121 ≈ 2.3k params)
- [x] Redes genéricas (MLP/Policy/Critic já usam `n_inputs`; `load_brain` descarta pesos 11-dim com aviso)
- [x] `debug_sensores.py`: FakeWorm com `direction`, FakeEnv com `obstacles`, testes C1 (17-dim) + C4 (faixas)
- [x] `main.py`: comentários 17-dim (cérebro/crítico/estado)
- [ ] ~~Treinar 500+ eps com 17 features + rede maior (Fase 11)~~ CANCELADO (17-dim revertida; treinos agora no 8-dim, ver Fase 16)
- [ ] ~~Comparar com 11 features (Fase 10/11)~~ CANCELADO (mesmo motivo)

**Validação 2026-09 (headless, sem GUI):** 35 eps × 200 passos, PPO puro 17-dim, sem crash (ver `EXPERIMENTOS.md #11`).

**Treino GUI 2026-09 (dif FACIL, ver `EXPERIMENTOS.md #12`):** 35eps (A10/B20/C5) + eval 20eps → 15% seguro (3/20). Estágio C 1/5 seguro. Colapso persiste (entropia 0). Conclusão: 17-dim exige 500+ eps; 35eps insuficiente.

**Métrica de sucesso:** ≥60% (+5pp sobre Fase 11).

---

## Fase 13 — Replay Buffer (off-policy) ✅ concluída (código, OFF por padrão)

**Objetivo:** reutilizar dados de episódios passados em vez de descartá-los.

**Hipótese:** com um buffer de10.000 transições, o PPO pode treinar em mini-batches de dados antigos, aumentando a eficiência de dados em 2-4x.

**Implementado:**
- [x] Classe `ReplayBuffer` em `mlp.py:19` — armazena até50 episódios (~10k transições), com `add_episode` / `sample_batch` / `clear`
- [x] `PolicyNetwork.ppo_update_from_buffer()` em `mlp.py:541` — K epochs × mini-batches, PPO clipped + critic MSE, com clipping [-10,10] para estabilidade pure-python
- [x] `config.py` — `buffer_max_episodes=50`, `buffer_min_transitions=200`, `buffer_epochs=2`, `buffer_batch_size=64`, `ppo_clip=0.2`
- [x] `main.py:finish_episode` — computa `advantages/returns/old_probs` e alimenta buffer; treina do buffer quando ≥200 transições, senão fallback para `update_episode`
- [x] `test_mlp.py` — 15/15 OK (2 novos: buffer add/sample/clear, PPO from buffer)
- [x] `overflow fix` — clipa `advantage/returns/delta` em [-10,10] para evitar `OverflowError` do critic (reward_scale=5 × 200 passos)

**Avaliação (pré-fix vs pós-fix):**
- Pré-fix (ratio=1.0, critic target=advantage): 25-35% (50 eps eval, 200 eps treino) — PPO clipping inoperante
- Pós-fix: treino puro-python com buffer custa ~4.3s/ep (vs 1.2s/ep sem buffer) → 200 eps ≈14 min/seeds (75 min total). Bug de overflow travava no ep.34; corrigido mas tempo 7× maior.
- Conclusão: buffer **arquiteturalmente correto mas caro em pure-python**. Requer otimização (reduzir epochs/batch ou treino noturno) ou implementação vetorizada.

**Próximo passo para validar ganho real:** rodar 200 eps × 3 seeds com buffer pós-fix em ambiente com mais tempo (ou nightly run), comparar com Fase 9 (40%). Meta mantida: ≥50% para justificar custo.
> Nota 8-dim: buffer segue desativado (`buffer_min_transitions=999999`); reativar
> só via `--set=buffer_min_transitions=200` após baseline #35 estabilizar.

---

## Apêndice A — Fase 15: era dos obstáculos e níveis (REMOVIDA na reconstrução terreno-limpo)

> Conteúdo abaixo preservado como histórico. Em 2026-10-08 pedras, muros,
> `difficulty 0–3`, tecla `O`, `randomize_obstacles`, sensores `obs_*` e coluna
> `colisao_obs` foram removidos do código. Não usar como referência de API.

**Objetivo:** complicar o ambiente para forçar aprendizado de desvio e navegação real.

**Implementado:**
- `config.py`: `difficulty` 0–3 (`LIVRE/FACIL/MEDIO/DIFICIL`), `obstacle_radius=2.5`, `obstacle_penalty=3.0`, `n_inputs` 8→11
- `environment.py`: `obstacles[]`, `place_obstacle()`, `clear_obstacles()`, `randomize_obstacles(difficulty)` — até 7 pedras com colisão box, reserva centro e evita sobrepor luz/chuva
- `perception.py`: sensores 11-dim (`obs_dir.x/z`, `obs_dist`), recompensa com colisão + proximidade + progresso de afastamento, `prev_dist_obs`
- `worm.py`: `step(dt, env)` com teste de colisão simples — se `next_pos` dentro do raio, desliza lateralmente
- `main.py`: `difficulty` no `state`, HUD mostra `nivel/label/obs:N`, CSV `colisao_obs`, professor contorna pedras (repulsão + tangente), `randomize_obstacles` por episódio, tecla **O** cicla 0→3, `--set=difficulty=N` para experimentos
- `mlp.py`: `load_brain` com checagem de `n_inputs` — pesos antigos 8→11 são ignorados com aviso
- Economia: 11×16+16×5=256 pesos (+13% vs 229) — treino novo necessário; pesos antigos incompatíveis são descartados automaticamente
- **Níveis:** 0=0 pedras, 1=3×1.8, 2=5×2.2, 3=7×2.6 — cada nível aumenta densidade e exige desvio; use `python main.py --set=difficulty=0` para comparar baseline

## Fase 14 — Avaliação e DoD (REPLANEJADA 2026-09-30)

**Objetivo:** validar se o verme atinge o critério de 80% de chegada sem perigo.

**Status anterior (congelado 2026-09-24):** DoD relativo não atingido (21-26% do professor).

### Resultados L5 (2026-09-30) — Ablação 4 configs × 4 seeds (16 runs)

| Config | Treino A % | Eval B % | Entropia final | Ação dominante C |
|--------|------------|----------|----------------|------------------|
| Sem currículo | 27-38% | 11-17% | 0.000 | 4 (direita) 200/200 |
| Com currículo | 33-41% | 11-17% | 0.000 | 0 (frente) 200/200 |
| Sem fome | 27-34% | 11-17% | 0.000 | 4 (direita) 200/200 |
| Sem ambos | 27-34% | 11-17% | 0.000 | 4 (direita) 200/200 |

**Veredito L5:** Colapso total em 16/16 runs. Currículo e fome NÃO evitam colapso. Agente repete ações (sempre direita ou frente) em vez de pensar. Overfitting A (treino 27-41% vs eval B 11-17%). DoD relativo ~21% vs meta 80%.

### Diagnóstico atualizado

1. **Rede 11→32→16→5 (~1.7k params) insuficiente** para representar desvios no labirinto
2. **Colapso de política** em 16/16 runs (entropia 0.000) — rede converge para 1 ação
3. **Overfitting ao labirinto A** — agente aprende a resolver A mas não generaliza para B
4. **Recompensa esparsa** — agente só recebe recompensa significativa perto do alimento
5. **Falta de exploração** — agente não explora o suficiente para descobrir novas estratégias

### Novo plano de ação (L6+)

**L7 — Anti-colapso estrutural:**

**Objetivo:** evitar o colapso de política (entropia → 0) observado em 16/16 runs da L5.

**Resultado L7 (2026-09-30):** 3/3 configs colapsaram (ent 0.000, ação 0 200/200). Anti-colapso estrutural FALHOU.

| Config | Treino A % | Eval B % | Entropia | Ação dominante C |
|--------|------------|----------|----------|------------------|
| 1: entropy=0.15, min_T=0.4, lr_decay=0.999 | 31% | 11% | 0.000 | 0 (frente) 200/200 |
| 2: entropy=0.2, min_T=0.5, lr_decay=0.999 | 31% | 11% | 0.000 | 0 (frente) 200/200 |
| 3: entropy=0.15, min_T=0.4, lr_decay=0.999, repeat_penalty=0.2 | 31% | 11% | 0.000 | 0 (frente) 200/200 |

**Diagnóstico:** O problema é estrutural — a rede 11→32→16→5 com PPO está fundamentalmente limitada para este domínio. Os parâmetros de exploração (entropy_coef, min_temperature) não são suficientes para evitar o colapso. A rede converge para uma única ação (frente) independentemente dos parâmetros de exploração.

**Próximos passos (L8+):**
- **L8 — Recompensa densa:** Aumentar smell_bonus, adicionar recompensa por progresso contínuo, reduzir progress_scale
- **L9 — Generalização:** Treinar com labirintos aleatórios, adicionar ruído aos sensores, data augmentation
- **L10 — Avaliação final:** Rodar 4 configs × 4 seeds com anti-colapso + recompensa densa + generalização

**L8 — Recompensa densa:**

**Resultado L8 (2026-09-30):** 3/3 configs colapsaram (ent 0.000, ação dominante 200/200). Recompensa densa FALHOU.

| Config | Treino A % | Eval B % | Entropia | Ação dominante C |
|--------|------------|----------|----------|------------------|
| 1: smell=0.5, progress=1.5 | 28% | 17% | 0.000 | 4 (direita) 200/200 |
| 2: smell=1.0, progress=1.0 | 28% | 17% | 0.000 | 4 (direita) 200/200 |
| 3: smell=0.5, progress=2.0 | 31% | 11% | 0.000 | 0 (frente) 200/200 |

**Diagnóstico:** O problema é estrutural — a rede 11→32→16→5 com PPO está fundamentalmente limitada para este domínio. Aumentar a recompensa (smell_bonus, progress_scale) não evita o colapso. A rede converge para uma única ação independentemente dos parâmetros de recompensa.

**Conclusão L7+L8:** Parâmetros de exploração (entropy_coef, min_temperature) e recompensa (smell_bonus, progress_scale) NÃO evitam o colapso. O problema é a capacidade da rede de representar políticas complexas para este domínio.

**Próximos passos (L9+):**
- **L9 — Generalização:** Treinar com labirintos aleatórios, adicionar ruído aos sensores, data augmentation
- **L10 — Avaliação final:** Rodar 4 configs × 4 seeds com anti-colapso + recompensa densa + generalização
- **Alternativas radicais:** DQN com ε-greedy, behavioral cloning, ou aceitar limitação do RL puro sem frameworks

---

## Reformulação Estrutural (2026-09-30)

**Problema:** A rede 11→32→16→5 com PPO colapsa para uma única ação (sempre direita ou frente) em 16/16 runs (L5) + 3/3 configs (L7) + 3/3 configs (L8). O agente não tem "curiosidade" — não explora, apenas repete padrões.

**Causa raiz:** A rede não tem incentivo para explorar. A recompensa é esparsa (só perto do alimento) e a política converge para uma única ação que maximiza a recompensa imediata.

**Solução proposta:** Recompensa por novidade + diversidade de ações + arquitetura mais robusta.

### L9 — Recompensa por novidade (novelty bonus)

**Objetivo:** Incentivar o agente a visitar estados novos, forçando exploração.

**Hipótese:** Se o agente recebe um bônus por visitar estados novos, ele vai explorar mais e evitar colapso.

**Implementação:**
- [ ] Adicionar `novelty_bonus` em `config.py` (ex: 0.1-0.5)
- [ ] Rastrear estados visitados (hash da posição + direção)
- [ ] Dar bônus quando o agente visita um estado novo
- [ ] Decair o bônus com o tempo (estados já visitados dão menos bônus)

**Métricas de sucesso:**
- Entropia final > 0.1 (vs 0.000 na L5)
- Ação dominante < 150/200 (vs 200/200 na L5)
- DoD relativo ≥50% (vs 21% na L5)

### L10 — Recompensa por diversidade de ações

**Objetivo:** Penalizar a repetição de ações, forçando o agente a tentar ações diferentes.

**Hipótese:** Se o agente sofre penalidade por repetir ações, ele vai tentar ações diferentes e evitar colapso.

**Implementação:**
- [ ] Adicionar `action_repeat_penalty` em `config.py` (ex: 0.1-0.3)
- [ ] Rastrear as últimas N ações (ex: 20)
- [ ] Se o agente repete a mesma ação mais de X vezes, aplicar penalidade crescente
- [ ] Aumentar `action_repeat_window` de 20 para 50-100

**Métricas de sucesso:**
- Entropia final > 0.1 (vs 0.000 na L5)
- Ação dominante < 150/200 (vs 200/200 na L5)
- DoD relativo ≥50% (vs 21% na L5)

### L11 — Arquitetura mais robusta

**Objetivo:** Aumentar a capacidade da rede para representar políticas mais complexas.

**Hipótese:** Uma rede maior (11→64→32→5, ~3.5k params) pode capturar relações não-lineares mais sutis entre sensores e ações.

**Implementação:**
- [ ] Aumentar `n_hidden` de 32 para 64
- [ ] Aumentar `n_hidden2` de 16 para 32
- [ ] Testar com `test_mlp.py` (15/15 OK)
- [ ] Treinar 100 eps com rede nova + novelty bonus + action diversity

**Métricas de sucesso:**
- Entropia final > 0.1 (vs 0.000 na L5)
- Ação dominante < 150/200 (vs 200/200 na L5)
- DoD relativo ≥50% (vs 21% na L5)

### L12 — Treinamento com labirintos aleatórios ⏸️ SUSPENSO (sem labirintos até o gigante)

**Objetivo:** Forçar generalização, evitando overfitting ao labirinto A.

**Hipótese:** Se o agente treina com labirintos aleatórios, ele vai aprender a navegar em qualquer labirinto, não só no A.

**Implementação:**
- [ ] Criar labirintos aleatórios (ex: 5-10 labirintos diferentes)
- [ ] Treinar com labirintos aleatórios (não só A)
- [ ] Avaliar em B (nunca visto)
- [ ] Comparar com baseline L5 (treino só em A)

**Métricas de sucesso:**
- Entropia final > 0.1 (vs 0.000 na L5)
- Ação dominante < 150/200 (vs 200/200 na L5)
- DoD relativo ≥50% (vs 21% na L5)

### L13 — Avaliação final

**Objetivo:** Validar se a reformulação estrutural resolve o problema de colapso.

**Implementação:**
- [ ] Rodar 4 configs × 4 seeds com novelty bonus + action diversity + arquitetura robusta + labirintos aleatórios
- [ ] Comparar com baseline L5 (sem reformulação)
- [ ] Se DoD relativo ≥80%: promover Fase 14 como atingida
- [ ] Senão: documentar limite e considerar alternativas radicais (DQN, behavioral cloning)

**Métricas de sucesso:**
- Entropia final > 0.1 (vs 0.000 na L5)
- Ação dominante < 150/200 (vs 200/200 na L5)
- DoD relativo ≥80% (vs 21% na L5)

### L14 — Alternativas radicais (INICIADA 2026-10-05)

**Motivo:** L5 (16/16) + L7 (3/3) + L8 (3/3) + L9–L13 provam colapso **estrutural**
do policy-gradient softmax — entropia → 0 independente de recompensa,
exploração, rede ou labirinto. Trocar a dinâmica, não os parâmetros.

**L14a — Behavioral Cloning puro (baseline de representação, dims 8-dim na era atual):**
- Só `imitate()` (CE supervisionada), `autonomy=0` no treino, `autonomy=1` no eval.
- Pergunta: a rede 11→32→16→5 ao menos **representa** o professor?
- Uso: `python train_food_headless.py --episodes=30 --seed=42 --bc --save=docs/historico/pesos/pesos_L14_bc.json` (comando da era-com-muros, sem `--maze` hoje)
- Smoke 2026-10-05: treino 50% (movimento = professor), eval B **0%** (ação 2
  200/200). Até o supervisionado colapsa no distribution-shift (treina em
  trajetórias do professor, avalia nas próprias). Falha de representação/generalização,
  não só de RL.

**L14b — DQN value-based + ε-greedy (dinâmica nova, dims 8-dim na era atual):**
- `QNetwork` em `mlp.py` (11→32→16→5 linear, TD `r+γ·maxQ'`, rede-alvo c/ sync,
  `act()` ε-greedy). Exploração externa, sem depender de entropia softmax.
- Uso: `python train_food_headless.py --episodes=15 --seed=42 --dqn --dqn-epsilon=0.3` (comando da era-com-muros, sem `--maze` hoje)
- Smoke 2026-10-05: treino 6.7%, tops variam (0/3/1/2/4) — **sem colapso
  estrutural** no treino, mas eval greedy ainda 0% (15 eps insuficientes).
- Nota: `QNetwork.td_update` usa sinal correto `(target−Q)` p/ `θ+=lr·grad`;
  o `Critic` legado usa sinal invertido e diverge (verificado 2026-10-05) —
  não mexer no Critic sem revalidar Fases 8/9.

**Próximos passos L14:**
- [x] QNetwork + teste `test_mlp.py` 16/16 + flags `--bc/--dqn` (este passo)
- [ ] BC 100 eps × 3 seeds + eval B (confirma 0% ou mede representação real)
- [ ] DQN 100 eps × 3 seeds (`--dqn-epsilon=0.2–0.3`, `--target-sync=10`) + eval B
- [ ] Se DQN > 21% relativo: promover como trilha principal; senão, aceitar
  limite do RL puro sem frameworks (documentar + congelar).

### L15 — Consolidação honesta (INICIADA 2026-10-05, via agentificação)

**O que é, p/ quem chegou agora:** antes de rodar 100 eps caros, três
investigadores-robôs (subagentes) releram `train_food_headless.py`, `mlp.py` e
`main.py` procurando mentiras nas métricas. Acharam 5. L15 corrige as 5 sem
mudar o legado PPO — só os caminhos novos `--bc/--dqn`.

**Ação L15-1 — s_next antes do respawn + done ao comer (DQN aprende a verdade):**
- Antes: `s_next` era lido DEPOIS de teleportar a comida → transição
  `(s,a,r,s')` misturava dois objetivos (comida velha deu `r`, comida nova
  aparece em `s'`). `done` era `step==H-1` (timeout virava "morte").
- Agora: `s_next` capturado antes do respawn; `done=True` só ao comer
  (subgoal terminou: `target=r`, sem bootstrap p/ goal novo). Timeout é
  truncamento (`done=False`, mantém bootstrap).
- Risco previsto: muda a dinâmica vs smoke L14 — curvas antigas não comparáveis.
  Reavaliado: sensores 11-dim não usam `prev_dist/hunger`, então ler `s_next`
  após `calculate_food_reward` é seguro.

**Ação L15-2 — act_count da ação executada (placar honesto no BC):**
- Antes: contava a ação amostrada e somava `+0` no override → `top` do log era
  da política aleatória, não do professor executado.
- Agora: conta após `action=best`. Só afeta log, não aprendizado. Smoke 8 eps:
  tops variados (200/200, 126/200, 141/200…) = professor real.

**Ação L15-3 — td_update com passo médio (anti-explosão):**
- Antes: `accumulate` soma N grads e `apply` 1x → passo efetivo `lr×N`
  (200×0.05=10.0). Herdado de `update_episode`/`ppo_from_buffer`.
- Agora: `apply(lr/N)`. Smoke DQN 8 eps: tops ainda diversos (0/1/2), 0%
  (curto, esperado), sem explosão. Reavaliar em 100 eps antes de concluir.

**Ação L15-4 — load_brain checa n_outputs (anti-corrupção silenciosa):**
- Antes: só checava `n_inputs/n_hidden` → peso Q(5 saídas) carregava em
  Critic(1 saída) e corrompia; Q↔Policy carregava silencioso.
- Agora: recusa `n_outputs` divergente com AVISO (verificado: Q→Critic=False,
  Q→Policy=True mesma forma). Q↔Policy ainda passa (limitação documentada:
  forma igual, semântica Q≠logits).

**Ação L15-5 — guards + divergência oficial GUI vs headless:**
- `--bc+--dqn` juntos → DQN vence com AVISO (eram indefinidos).
- `--epsilon` genérico só no treino (antes randomizava o eval).
- `brain.temperature` só p/ Policy (Q não tem softmax).
- Divergência oficial: `main.py` continua PPO-only (sem `--bc/--dqn`),
  professor/física/CSV diferentes — números headless BC/DQN NÃO transferem
  p/ GUI sem portar. CSV unificado adiado p/ L16 (mudança grande).

**Pendente L15 (resolvido na reconstrução):** CSV headless = GUI em **13 cols**
(sem `colisao_obs`); item "15 cols" cancelado. Restam: professor headless = GUI
(só borda, sem muros), sync por step, `reward_scale` default 1.0 p/ Q (`q_reward_scale`), teste `done=False` + `scale` default.
Critics legado com sinal invertido NÃO mexido (revalidaria Fases 8/9).

### L16 — Protocolo comparável (FEITA 2026-10-05)

**O que é, p/ quem chegou agora:** L14 criou BC/DQN só no headless, L15
consertou mentiras. Faltava tornar o headless *comparável* ao jogo visual:
mesmo professor, mesmo CSV, mesmos defaults seguros. Sem isso, 100 eps não
provam nada.

**Ação L16-1 — professor headless = GUI (borda; muros removidos na era atual):**
- Antes: headless só atraía p/ comida; GUI atrai + foge borda (`wall_margin`)
  + contorna muros (repulsão 1.4 + tangente 0.6, raio ×2.2).
- Agora: `teacher_dir()` headless soma os 3 vetores com a mesma fórmula.
  `teacher_action/best` e estágios A/B voltam a significar o mesmo.
- Risco: muda o professor → BC e PPO-híbrido mudam junto. Smoke 8 eps BC:
  50% (vs 37.5% L15) — variação esperada, não regressão.

**Ação L16-2 — CSV headless 13 cols = GUI (`--csv=`):**
- Header idêntico a `main.py`: episodio, estagio, recompensa_total/media,
  retorno_medio (= media r; BC/DQN sem GAE, documentado), entropia_media
  (DQN = epsilon proxy), lr, lambda, autonomia, encontro, passos_ate_comer,
  colisao_obs (fração steps no raio), ação, semente, maze.
- Uso: `--csv=episodios_headless.csv`. Validado 8+8 eps (header + linhas OK).
- Não quebra legado: default `csv_path=None` (sem arquivo).

**Ação L16-3 — Q defaults seguros + teste bootstrap:**
- `reward_scale` default Q: 5.0 (PPO) → 1.0 (`q_reward_scale`; override via
  `--set`). Q em escala crua [-1,1] não explode o target.
- `td_update` sem `target_net` agora avisa (self-bootstrap instável).
- `test_mlp.py`: novo caso `done=False` prova `target=r+γ·maxQ'`.
- Sync por steps: `--target-sync-steps=N` (0 = legado por episódio).

**Protocolo oficial L16 (HISTÓRICO era-com-muros — bloco preservado como registro;
flags `--maze` removidas do código depois; para a era atual ver Fase 16):**
```bash
python train_food_headless.py --episodes=100 --maze=A --seed=42 --csv=ppo_A100.csv --save=pesos_L16_ppo.json
python train_food_headless.py --episodes=100 --maze=A --seed=42 --bc --csv=bc_A100.csv --save=pesos_L16_bc.json
python train_food_headless.py --episodes=100 --maze=A --seed=42 --dqn --dqn-epsilon=0.2 --csv=dqn_A100.csv --save=pesos_L16_dqn.json
python train_food_headless.py --episodes=100 --maze=B --seed=99 --eval --weights=pesos_L16_ppo.json --csv=ppo_Beval.csv
python train_food_headless.py --episodes=100 --maze=B --seed=99 --eval --weights=pesos_L16_bc.json --csv=bc_Beval.csv
python train_food_headless.py --episodes=100 --maze=B --seed=99 --eval --dqn --weights=pesos_L16_dqn.json --csv=dqn_Beval.csv
python train_food_headless.py --episodes=100 --maze=B --seed=99 --teacher-only  # teto
```
- Não mexido: Critic (sinal), física grade vs deslize (limitação documentada).

**Revisão L16-refaz (2026-10-05, avaliação contínua):**
- `debug_food.py` C1 voltava a FALHAR (`r=+0.200` sem alimento): teste
  pré-L9, não bug — novidade L9 dá +0.2 por estado novo mesmo sem comida.
  Fix no teste (desliga `novelty_bonus` só no C1); C2–C6 seguem reais.
  Base agora: `test_mlp` 16/16 + `debug_food` TUDO OK + `debug_maze` OK.
- Professor: `limit` agora do CONFIG (era `env.limit`); zona-morta em L2;
  muros headless 1 ponto/muro = GUI (era 3; repelia ~3x). Quebra smokes
  L14/L15 — intencional, antes do protocolo de 100 eps.
- `config.py`: `q_reward_scale=1.0` centralizada (era `get` solto).
- `--target-sync-steps` não-múltiplo de H agora avisa (só vale em fronteira).
- Smoke pós-refaz 8 eps A seed42: PPO 37.5%, BC 37.5% (estágio A/BC ambos
  autonomy=0: mesmo movimento-professor, confere), DQN 0% tops diversos.
  CSVs 15 cols salvos e conferidos.

### L17 — Veredito 100 eps: PPO vs BC vs DQN (HISTÓRICO era-com-muros; gate conceitual mantido)

**O que é, p/ quem chegou agora:** L14 criou dois cérebros novos (BC que copia
o professor, DQN que aprende valores), L15–L16 consertaram as réguas para a
disputa ser justa (mesmo professor, mesmo CSV, escala segura). L17 é a luta:
3 treinos de 100 episódios no labirinto A + 3 avaliações no B (nunca visto) +
o teto do professor. No fim, uma frase decide o futuro do projeto.

**Hipóteses (o que esperamos, p/ cobrar depois):**
- H1 PPO repete L5: treino A ~30% e eval B ~11–17%, entropia → 0 (colapso).
- H2 BC repete L14a: eval B ~0% (copia bem o professor, dirige mal sozinho —
  distribution-shift clássico do supervisionado).
- H3 DQN é a incógnita: treino diverso (tops variados), eval B incerto —
  qualquer coisa ≥21% relativo já é melhor que tudo desde L5.

**Tarefas (nesta ordem, 1 seed primeiro — barato antes do caro):**
- [ ] L17-1 Treino PPO 100 eps A seed42 (+CSV): `... --episodes=100 --maze=A --seed=42 --csv=ppo_A100.csv --save=pesos_L17_ppo.json`
- [ ] L17-2 Treino BC 100 eps A seed42 (+CSV): `... --bc --csv=bc_A100.csv --save=pesos_L17_bc.json`
- [ ] L17-3 Treino DQN 100 eps A seed42 ε=0.2 (+CSV): `... --dqn --dqn-epsilon=0.2 --csv=dqn_A100.csv --save=pesos_L17_dqn.json`
- [ ] L17-4 Eval PPO 100 eps B seed99 (+CSV): `... --eval --weights=pesos_L17_ppo.json --csv=ppo_Beval.csv`
- [ ] L17-5 Eval BC 100 eps B seed99 (+CSV): `... --eval --weights=pesos_L17_bc.json --csv=bc_Beval.csv`
- [ ] L17-6 Eval DQN 100 eps B seed99 (+CSV, note o `--dqn`): `... --eval --dqn --weights=pesos_L17_dqn.json --csv=dqn_Beval.csv`
- [ ] L17-7 Teto do professor 100 eps B seed99: `... --teacher-only` (régua do DoD relativo)
- [ ] L17-8 Linha #32 no `EXPERIMENTOS.md` + veredito abaixo (1 frase)

**Métricas (as mesmas 15 cols do CSV; comparar maçã com maçã):**

| Métrica | Onde ler | Sucesso |
|---------|----------|---------|
| % com encontro (eval B) | `encontro_food` | DQN ≥ 21% relativo ao professor (supera L5) |
| DoD relativo | eval / teto professor | ≥80% = promove Fase 14; senão, congela |
| Diversidade | `acao_principal` + `entropia_media` | top < 150/200, ent > 0.1 (sem colapso) |
| Tempo-até-comer | `passos_ate_comer` mediana | menor = melhor (pergunta-base F8) |

**Regra de decisão (gate, sem meio-termo):**
- DQN ≥80% do professor → trilha principal vira DQN (L18 = multi-seed + ε-decay).
- DQN 21–79% → L18 = melhorias focadas (ε-decay, DAGGER `λc`, sync por step).
- Tudo ≤21% (repete L5) → aceita o limite do RL puro sem frameworks:
  documenta, congela treinos longos, projeto vira material didático de
  "como o RL colapsa e por quê".

**Travas (p/ não estragar a disputa):**
- Congela código durante os 7 runs (só `--set`/`--csv`/`--save`; nada de editar
  `.py` no meio — senão os 3 cérebros correm em pistas diferentes).
- Eval DQN sem `--dqn` é inválido (carrega Q como Policy); eval PPO/BC sem
  `--dqn` (óbvio, mas já confundimos no L14-smoke).
- Pesos `pesos_L17_*.json` + CSVs guardados juntos (reproduzível = seed fixa).

**O que NÃO entra (fica p/ L18+/Fase 16):** multi-seed (só do vencedor), ε-decay no
DQN, DAGGER/interleaving professor, fix do Critic (revalidaria Fases 8/9),
porte `--bc/--dqn` p/ GUI, física deslize-vs-grade, `gitignore` de pesos.
> Era atual: **válidos** multi-seed do vencedor, ε-decay, DAGGER `λc`, fix Critic;
> **suspensos até o labirinto gigante**: porte GUI-bc/dqn-muros, random-maze,
> física deslize-vs-grade, treino A→B.

### L17 — Veredito (EXECUTADA 2026-10-05, mapa 32 COM muros — histórico)

**Resultado:** treino A100 s42: PPO 36%/0.47ep, BC 80%/1.23ep (= professor),
DQN 6%/0.06ep. Eval B100 s99: PPO **8%**, BC **5%**, DQN **7%**; teto
professor **89%**/1.32ep. DoD relativo: 9% / 6% / 8% (meta 80%).
**Gate: tudo ≤21% → limite do RL puro sem frameworks ACEITO.**
Treinos longos congelados; projeto preservado como material didático de
"como policy-gradient, clonagem e Q colapsam/estagnam e por quê".
Nuance nova: BC 100eps **não** colapsou (ent 0.43, tops 0/2/4) — o
supervisionado preserva diversidade mas não resolve (shift de distribuição
confirmado); PPO colapsou (0-400/400); DQN greedy colapsou p/ "frente" no
eval apesar de diverso no treino (ε=0.2 mascara). Ver `#34`.

> **Mapa 32 (pré-L17, 2026-10-05):** `map_limit` 16→32, `ground_scale` 72,
> `sensor_max_dist` 60, `food_limit` 24, currículo 10+0.3/ep, H 200→400,
> `hunger_rate` 0.001, A 3 muros+2 pedras / B 4 muros+1 pedra, câmera GUI
> reenquadrada. Pesos antigos carregam (mesma 11-dim) mas jogam outro jogo:
> smokes #18–#31 incomparáveis; L17 roda no mapa novo. Ver `#32`.

---

## Fase 16 — Terreno limpo 8-dim + protótipo 🟢 ATUAL (2026-10-08)

**O que mudou:** reconstrução removeu pedras/muros/labirinto/dificuldade;
estado 11→8-dim; CSV 13 cols; `load_brain` recusa pesos antigos (correto).
P0/P1 de código corrigidos (action centralizado, `step(dt)`, renames
`EPISODE_LIMIT`/`SPAWN`, header `encontros_food`, defaults `limit=24`).

**Baseline #35 (era atual, PPO 50eps seed42 H=400):** 56% (28/50), média 0.78/ep;
estágio C 8/20=40% (ent 0.000, colapso ação 0); passos-ate-1o mediana 163/min 16.
CSV `baseline_limpo.csv`. **Leitura honesta:** colapso persiste sem obstáculos —
labirinto não era a causa; hipótese vigente é estrutural (softmax on-policy,
20k passos). Próximo teste barato: `--dqn` e `--bc` no 8-dim (ainda não rodados).

**Definição do protótipo (conclusão visada):**
1. Demo GUI: verme encontra alimento em terreno limpo com HUD/grade funcionando.
2. Treino headless reproduzível 50eps com CSV 13 cols + `debug_food/maze` + `test_mlp` verdes.
3. Docs sem passado-como-presente (este replanejamento).
4. (Depois) Labirinto gigante `map_limit` 64 especificado em `PLANO_LABIRINTO_COMIDA.md §9`.

**Próximos passos (ordem):**
- [x] Baseline #35 PPO 50eps (+CSV)
- [ ] `--dqn` 50eps + `--bc` 50eps no 8-dim (barato; decide se DQN/BC viram trilha)
- [x] Arquivar CSVs 15-cols e pesos 11/17-dim como histórico (`docs/historico/`)
- [x] Reescrever `README.md` p/ era 8-dim
- [ ] Labirinto gigante §9 (só após P0 zerado)

## Resultados até agora (Fases 0-9, HISTÓRICO eras luz/chuva e labirinto)

| Fase | Algoritmo | Melhor seed (50 eps) | Eval longo (100 eps) | Observação |
|------|-----------|---------------------|---------------------|------------|
| 7 | REINFORCE | 40% | 20% | Política colapsa para 1 ação |
| 8 | A2C | 45% | 39% | 2x mais estável que REINFORCE |
| 9 | PPO | 40% | 40% | ≈ A2C, ambos 2x REINFORCE |

**Diagnóstico:** os algoritmos melhoram (10% → 40%) mas estagnam. O problema **não é o algoritmo** — é a quantidade de dados e a capacidade da rede. On-policy com20k passos e229 parâmetros é insuficiente para este domínio 3D.

---

## Mapa conceitual

| Termo | No projeto | Onde aparece |
|-------|-----------|--------------|
| Estado `s` | 8 features (food_dir/dist + smell + vel/borda) | `get_food_sensor_inputs()` |
| Ação `a` | uma das 5 direções | `sample_action(π)` |
| Política `π(a|s)` | softmax do MLP | ponta do `forward` |
| Retorno `G_t` | soma descontada `Σγ^k r` | `compute_returns()` |
| Baseline `b` | média dos retornos (REINFORCE) ou V(s) (A2C/PPO) | `mean(G)` ou `critic.value()` |
| Advantage `A_t` | GAE: `δ_t + γλ·δ_{t+1} + ...` | `compute_gae()` |
| Gradiente `∇logπ` | backprop do cross-entropy | `backward()` |
| Professor | campo de potencial: atração alimento + fuga de borda (sem muros) | `get_target_direction()` |
| Currículo | `λ_imitação` decaído + `teacher_influence` | Fase 4 |
| Clipping | PPO: limita `π_new/π_old` em [1-ε, 1+ε] | `ppo_grad_log_prob_z()` |
| Replay Buffer | fila de transições reutilizáveis | Fase 13 (código pronto, OFF por padrão) |

---

## Referências

- Williams (1992): REINFORCE — Simple statistical gradient-following algorithms for connectionist reinforcement learning
- Schulman et al. (2016): PPO — Proximal Policy Optimization Algorithms
- Mnih et al. (2015): DQN — Human-level control through deep reinforcement learning
- Sutton & Barto: Reinforcement Learning: An Introduction (cap. 13: Policy Gradient Methods)
