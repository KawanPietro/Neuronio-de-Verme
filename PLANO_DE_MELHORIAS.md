# Plano de Melhorias — Verme Autônomo

> ❄️ **ESTADO CONGELADO (2026-09-24):** treino encerrado e documentado.
> Trilha legada (luz/chuva, 17-dim) preservada por padrão; trilha alimento
> (11-dim, `--maze=A/B`) funcional em GUI + headless. Veredito Fase 14:
> DoD relativo **não atingido** (aprendido 21–26% do professor; #15–16).
> Reprodução em `EXPERIMENTOS.md #15–16` e seção "Como rodar".
> Novos treinos longos desaconselhados sem hipótese nova (ver T5–T8).

> Roteiro de evolução do projeto para chegar a um **verme 100% autônomo** (decide sozinho, sem professor), **didático e funcional**.
>
> Decisões de escopo alinhadas:
> - **Cérebro**: Puro Python do zero (sem NumPy/PyTorch) — cada equação fica visível.
> - **Autonomia final**: rede neural decide 100% das ações; o professor só existe nas fases de treino.
> - **Comportamentos**: trilha legada com chuva (alvo) e luz (perigo);
>   trilha atual com **alimento** (objetivo único) atrás de `--maze=A/B`
>   (ver seção F8 do `README.md`).
> - **Entregável**: este documento de rota por etapas.

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
| 12 | Espaço de estado expandido | Features adicionais (velocidade, borda, ângulo) | M | ✅ concluída (código, 11→17) |
| 13 | Replay Buffer (off-policy) | Reutilizar dados de episódios passados | L | ✅ concluída (código) — ver nota |
| 14 | Avaliação e_DoD | Validar ≥80% em 100+ eps com multi-seed | M | ❄️ congelada (legado 10% + alimento 21–26% do prof; DoD não atingido, #13/#15–16) |
| 15 | Obstáculos e níveis | Pedras com colisão, 4 níveis, 11 sensores | M | ✅ concluída |

> Estimativa de esforço: S = 1 sessão, M = 2-4, L = 4+.

**Critério de conclusão (Definition of Done):**
1. `teacher_influence = 0` de forma permanente. ✅
2. Legado: ≥80% chegada à chuva E sem perigo (eval 100+ eps, multi-seed). ❌ **Melhor: 40% (PPO)**.
   Alimento (DoD relativo ≥80% do professor): ❌ **A 26% / B 21% (#15)**.
   Teto do oráculo: A 66% / B 80% (#16) — DoD absoluto em B é loteria.
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

#### 3. Rede neural subdimensionada

- Atual: `8 → 16 → 5` (16 neurônios na camada oculta)
- Parâmetros: 8×16 + 16 + 16×5 + 5 = **229 pesos**
- Muito pequena para capturar a relação nonlinear entre sensores e ações em 3D

#### 4. Espaço de estado limitado (8 features)

| Feature | O que falta |
|---------|-------------|
| `luz_dir.x/z` | OK — direção para luz |
| `luz_dist` | OK — distância |
| `luz_perigo` | OK — binário |
| `chuva_dir.x/z` | OK — direção para chuva |
| `chuva_dist` | OK — distância |
| `pulso_chuva` | OK — vibração |

**Não temos**: velocidade atual, histórico de movimento, ângulo relativo à direção, distância até a borda. O verme não sabe "para onde estava indo" — só "para onde está a chuva".

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

### Arquivos importantes
- `config.py` — todas as constantes (+ chaves alimento: `food_*`, `n_inputs_food`,
  `epsilon_greedy`, `lambda_c`, `food_curriculum`)
- `mlp.py` — PolicyNetwork + CriticNetwork + REINFORCE/A2C/PPO (+ whitening T6,
  ε-greedy opt-in T6)
- `main.py` — loop do jogo, currículo, HUD, editor (+ `FOOD_MODE` via `--maze=A/B`,
  professor unimodal, CSV `encontro_food/passos_ate_comer/maze`)
- `perception.py` — sensores legados 17-dim + modo alimento 11-dim (`FOOD_DIM`,
  olfato, fome)
- `worm.py` — corpo do verme (cabeça + segmentos)
- `environment.py` — chuva/luz/fontes + randomização (+ `food_sources`,
  `load_maze`, `eat_and_respawn`, currículo proximal T8)
- `test_mlp.py` — 13 testes (gradientes, A2C, PPO, save/load)
- `debug_sensores.py` — headless legado (17-dim)
- `debug_maze.py` / `debug_food.py` / `train_food_headless.py` — validador A≠B,
  contrato alimento, treino/eval headless (`--teacher-only`, `--eval`,
  `--weights/--save`, `--set`, `--epsilon`)
- `labirintos.json` — A_treino ≠ B_teste (métrica de generalização)
- `VIABILIDADE/PLANO_MUDANCAS` — reformulação alimento: incorporados à seção
  F8 do `README.md` (documentos avulsos removidos na unificação 5→3).
- `EXPERIMENTOS.md` — tabela de ciência (16 experimentos; #15–16 = Fase 14 alimento)
- `PLANO_DE_MELHORIAS.md` — este arquivo
- `episodios.csv` — métricas por episódio (gerado pelo jogo)
- `pesos.json` — cérebro salvo (actor + critic)

### Como rodar
```bash
# Treino com limite
python main.py --episodes=100

# Treino alimento no labirinto A com currículo (trilha atual)
python main.py --maze=A --episodes=100 --set=food_curriculum=1 --set=epsilon_greedy=0.1 --set=lambda_c=0.05

# Avaliacao oficial (labirinto B, nunca visto no treino)
python main.py --eval --maze=B --episodes=100

# Headless rápido (mesma pergunta-base, sem GUI)
python train_food_headless.py --episodes=100 --maze=A --seed=42 --set=food_curriculum=1 --save=pesos_food.json
python train_food_headless.py --episodes=100 --maze=B --seed=99 --eval --weights=pesos_food.json
python train_food_headless.py --episodes=100 --maze=B --seed=99 --teacher-only  # teto do oráculo

# Avaliacao (carrega pesos.json, sem treino)
python main.py --eval --episodes=50

# Override de CONFIG
python main.py --episodes=100 --set=learning_rate=0.05 --set=gamma=0.95

# Testes
python test_mlp.py
python debug_sensores.py

# Compilacao
python -m py_compile config.py mlp.py main.py perception.py worm.py environment.py
```

---

## Fase 10 — Treino longo (PRÓXIMA)

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

**Hipótese:** com 3 camadas (8→32→16→5) e ~1.700 parâmetros, a rede pode capturar relações não-lineares mais sutis entre sensores e ações.

**Tarefas:**
- [x] Adicionar camada intermediária ao MLP (11→32→16→5) — `mlp.py` 3 camadas, numeric grad 1e-7
- [x] Adaptar `forward/backward` para 3 camadas + `n_hidden2` em Policy/Critic
- [x] Atualizar `PolicyNetwork` e `CriticNetwork` (11→32→16→1)
- [x] Testar com `test_mlp.py` (15/15 OK) + deep forward/backward
- [x] Treinar 100 eps com rede nova (dif 1, 11→32→16→5) — 0% (colapso esquerda 200)
- [ ] Treinar 500 eps com rede nova + multi-seed — rede maior precisa 2-3× mais dados
- [ ] Comparar com rede de 2 camadas (Fase 10: 15% shallow vs 0% deep em 100 eps)

**Resultado:** rede profunda (+13% params → 1.7k) não evita colapso com poucos dados; precisa Fase 12 (estado 16) + treino 500+.

---

## Fase 12 — Espaço de estado expandido ✅ concluída (código)

**Objetivo:** dar ao agente informação mais rica sobre si mesmo e sobre o mundo.

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
- [ ] Treinar 500+ eps com 17 features + rede maior (Fase 11) — pesos antigos incompatíveis, treino novo necessário
- [ ] Comparar com11 features (Fase 10/11)

**Validação 2026-09 (headless, sem GUI):** 35 eps × 200 passos, PPO puro 17-dim, sem crash (ver `EXPERIMENTOS.md #11`).

**Treino GUI 2026-09 (dif FACIL, ver `EXPERIMENTOS.md #12`):** 35eps (A10/B20/C5) + eval 20eps → 15% seguro (3/20). Estágio C 1/5 seguro. Colapso persiste (entropia 0). Conclusão: 17-dim exige 500+ eps; 35eps insuficiente.

**Métrica de sucesso:** ≥60% (+5pp sobre Fase 11).

---

## Fase 13 — Replay Buffer (off-policy) ✅ concluída (código)

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

---

## Fase 15 — Obstáculos e níveis de dificuldade ✅ concluída

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

## Fase 14 — Avaliação e DoD

**Objetivo:** validar se o verme atinge o critério de80% de chegada sem perigo.

**Tarefas:**
- [ ] Treinar a melhor combinação (Fase10-13) com3 sementes × 500 eps (pendente — exige nightly run)
- [x] Avaliar 1 semente com `--eval --episodes=100` (pesos 17-dim #12, dif FACIL)
- [x] Calcular: % de episódios com `chegada_chuva > 0` E `perigo_luz == 0` → **10% (10/100)**
- [ ] Se ≥80%:DoD atingido — **NÃO atingido**
- [ ] Se60-80%: considerar "aceitável" e documentar limitações — **NÃO atingido**
- [x] Se <60%: considerar alternativas radicais (ver abaixo) — **CASO ATUAL**

**Resultado 2026-09 (ver `EXPERIMENTOS.md #13`):** eval 100eps, 14/100 chegada>0, 10/100 seguro, entropia 0.000. DoD <60% → vale avaliar DQN/entorno simplificado/clonagem ou aceitar limitação do RL puro.

**Trilha alimento — Fase 14 headless (ver `EXPERIMENTOS.md #15–16`):**
treino 100eps maze A c/ currículo → eval 100eps: A **17%** (prof 66%), B **17%**
(prof 80%). DoD relativo (≥80% do professor): A 26% / B 21% — **NÃO atingido**,
política = acaso. Teto do professor oscila (B 60–80%): DoD absoluto em B é
loteria; usar DoD relativo. Fixes travados: whitening + `epsilon_greedy` +
`lambda_c` opt-in + currículo de distância (`food_curriculum`).

**Alternativas se DoD não for atingido:**
1. **DQN com ε-greedy**: off-policy puro, replay buffer nativo
2. **Entorno simplificado**: 1 fonte de chuva, 1 de luz, mapa menor
3. **Behavioral Cloning puro**: treinar o professor com mais fontes, depois clonar
4. **Aceitar limitação**: documentar que RL puro sem frameworks é insuficiente para este domínio

---

## Resultados até agora (Fases 0-9)

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
| Estado `s` | 17 features (dir/dist/perigo/pulso + obs + vel/borda/angulos) | `get_sensor_inputs()` |
| Ação `a` | uma das 5 direções | `sample_action(π)` |
| Política `π(a|s)` | softmax do MLP | ponta do `forward` |
| Retorno `G_t` | soma descontada `Σγ^k r` | `compute_returns()` |
| Baseline `b` | média dos retornos (REINFORCE) ou V(s) (A2C/PPO) | `mean(G)` ou `critic.value()` |
| Advantage `A_t` | GAE: `δ_t + γλ·δ_{t+1} + ...` | `compute_gae()` |
| Gradiente `∇logπ` | backprop do cross-entropy | `backward()` |
| Professor | campo de potencial (direção ideal) | `get_target_direction()` |
| Currículo | `λ_imitação` decaído + `teacher_influence` | Fase 4 |
| Clipping | PPO: limita `π_new/π_old` em [1-ε, 1+ε] | `ppo_grad_log_prob_z()` |
| Replay Buffer | fila de transições reutilizáveis | Fase 13 (pendente) |

---

## Referências

- Williams (1992): REINFORCE — Simple statistical gradient-following algorithms for connectionist reinforcement learning
- Schulman et al. (2016): PPO — Proximal Policy Optimization Algorithms
- Mnih et al. (2015): DQN — Human-level control through deep reinforcement learning
- Sutton & Barto: Reinforcement Learning: An Introduction (cap. 13: Policy Gradient Methods)
