# Plano de Melhorias — Verme Autônomo

> Roteiro de evolução do projeto para chegar a um **verme 100% autônomo** (decide sozinho, sem professor), **didático e funcional**.
>
> Decisões de escopo alinhadas:
> - **Cérebro**: Puro Python do zero (sem NumPy/PyTorch) — cada equação fica visível.
> - **Autonomia final**: rede neural decide 100% das ações; o professor só existe nas fases de treino.
> - **Comportamentos**: continuar com chuva (alvo) e luz (perigo).
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
| 10 | Treino longo | 500+ episódios com PPO + semente fixa | S | ⏳ pendente |
| 11 | Expansão de rede | Arquitetura maior (8→32→16→5) | M | ⏳ pendente |
| 12 | Espaço de estado expandido | Features adicionais (velocidade, histórico, ângulo) | M | ⏳ pendente |
| 13 | Replay Buffer (off-policy) | Reutilizar dados de episódios passados | L | ⏳ pendente |
| 14 | Avaliação e_DoD | Validar ≥80% em 100+ eps com multi-seed | M | ⏳ pendente |

> Estimativa de esforço: S = 1 sessão, M = 2-4, L = 4+.

**Critério de conclusão (Definition of Done):**
1. `teacher_influence = 0` de forma permanente. ✅
2. ≥80% dos episódios: chegada à chuva E sem perigo (eval 100+ eps, multi-seed). ❌ **Melhor: 40% (PPO)**
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
- `config.py` — todas as constantes
- `mlp.py` — PolicyNetwork + CriticNetwork + REINFORCE/A2C/PPO
- `main.py` — loop do jogo, currículo, HUD, editor
- `perception.py` — sensores (8 features) + recompensa por progresso
- `worm.py` — corpo do verme (cabeça + segmentos)
- `environment.py` — chuva/luz/fontes + randomização
- `test_mlp.py` — 13 testes (gradientes, A2C, PPO, save/load)
- `debug_sensores.py` — headless: contrato recompensa + casos-limite
- `EXPERIMENTOS.md` — tabela de ciência (5 experimentos)
- `PLANO_DE_MELHORIAS.md` — este arquivo
- `episodios.csv` — métricas por episódio (gerado pelo jogo)
- `pesos.json` — cérebro salvo (actor + critic)

### Como rodar
```bash
# Treino com limite
python main.py --episodes=100

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
- [ ] Rodar PPO com `--episodes=500` (seed fixa)
- [ ] Rodar PPO com `--episodes=1000` (seed fixa)
- [ ] Reduzir `lr_decay` (0.998 em vez de 0.995) — lr não decai tanto
- [ ] Reduzir `temperature_decay` (0.998 em vez de 0.995) — explora por mais tempo
- [ ] Aumentar `min_temperature` para 0.2 (permite exploração residual)
- [ ] Avaliar os pesos finais com `--eval --episodes=100`
- [ ] Comparar com baseline de40% (Fase 9)

**Métrica de sucesso:** ≥50% (melhoria de +10pp sobre Fase 9).

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
- [ ] Adicionar camada intermediária ao MLP (8→32→16→5)
- [ ] Adaptar `forward/backward` para 3 camadas
- [ ] Atualizar `PolicyNetwork` e `CriticNetwork` (8→32→16→1)
- [ ] Testar com `test_mlp.py` (gradientes OK?)
- [ ] Treinar 500 eps com rede nova + multi-seed
- [ ] Comparar com rede de 2 camadas (Fase 10)

**Métrica de sucesso:** ≥55% (+5pp sobre Fase 10).

---

## Fase 12 — Espaço de estado expandido

**Objetivo:** dar ao agente informação mais rica sobre si mesmo e sobre o mundo.

**Hipótese:** com8 features extras (16 total), o agente pode aprender a "navegar" (não só reagir) — considerando velocidade, bordas, e histórico.

**Novas features (16-dim):**

| Feature | Descrição |
|---------|-----------|
| `luz_dir.x/z` | direção XZ unitária para luz |
| `luz_dist` | distância normalizada [0,1] |
| `luz_perigo` | 1 se dentro do arrival_radius |
| `chuva_dir.x/z` | direção XZ unitária para chuva |
| `chuva_dist` | distância normalizada [0,1] |
| `pulso_chuva` | vibração (tato) |
| **NOVO:** `vel_x/z` | vetor velocidade normalizado |
| **NOVO:** `borda_x/z` | distância normalizada às bordas |
| **NOVO:** `angulo_luz` | ângulo relativo (atan2) entre direção atual e luz |
| **NOVO:** `angulo_chuva` | ângulo relativo entre direção atual e chuva |

**Tarefas:**
- [ ] Expandir `get_sensor_inputs()` para16 features
- [ ] Atualizar `n_inputs=16` no CONFIG
- [ ] Adaptar todas as redes (MLP, PolicyNetwork, CriticNetwork)
- [ ] Treinar 500+ eps com 16 features + rede maior (Fase 11)
- [ ] Comparar com8 features (Fase 10/11)

**Métrica de sucesso:** ≥60% (+5pp sobre Fase 11).

---

## Fase 13 — Replay Buffer (off-policy)

**Objetivo:** reutilizar dados de episódios passados em vez de descartá-los.

**Hipótese:** com um buffer de10.000 transições, o PPO pode treinar em mini-batches de dados antigos, aumentando a eficiência de dados em 10-50x.

**Tarefas:**
- [ ] Criar classe `ReplayBuffer` (deque de tuplas `(s, a, r, s', done)`)
- [ ] Modificar `update_episode` para amostrar do buffer
- [ ] Adaptar PPO para treinar em mini-batches (K epochs por batch)
- [ ] Treinar 200 eps com buffer (deve ser mais rápido que500 eps sem buffer)
- [ ] Comparar eficiência: eps até convergir vs. Fase 10

**Métrica de sucesso:** ≥65% com50% menos episódios que Fase 10.

---

## Fase 14 — Avaliação e DoD

**Objetivo:** validar se o verme atinge o critério de80% de chegada sem perigo.

**Tarefas:**
- [ ] Treinar a melhor combinação (Fase10-13) com3 sementes × 500 eps
- [ ] Avaliar cada semente com `--eval --episodes=100`
- [ ] Calcular: % de episódios com `chegada_chuva > 0` E `perigo_luz == 0`
- [ ] Se ≥80%:DoD atingido
- [ ] Se60-80%: considerar "aceitável" e documentar limitações
- [ ] Se <60%: considerar alternativas radicais (ver abaixo)

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
| Estado `s` | 8 features (dir/dist/perigo/pulso) | `get_sensor_inputs()` |
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
