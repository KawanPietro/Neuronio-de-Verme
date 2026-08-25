# ESTADO DA SESSÃO — 2026-08-20

## Status atual
- Fases 0–8 completas
- **DoD item 2 NÃO atingido** (80%): melhor = 45% seed 3 (A2C), 39% eval longo
- Último commit: `93753b2` (push pendente — sem autenticação GitHub)
- Evolução: REINFORCE 10% → A2C 39% (4x melhoria)

## O que foi feito (Fases 0–6)

### Fase 0 — Higiene do código
- `config.py` centralizado, `Verme.py` vira shim, `.gitignore`, `requirements.txt`, ASCII-safe prints

### Fase 1 — Sensores
- 8 features em `perception.py`: relações vetoriais (dot/cross), distâncias normalizadas, pulsos

### Fase 2 — Ações discretas + backprop
- `mlp.py`: PolicyNetwork com softmax, sample_action, backprop camada a camada
- 5 ações: `{−θ, −θ/2, 0, +θ/2, +θ}` (`TURN_MULTIPLIERS`)
- `test_mlp.py`: 8 verificações (gradientes ~1e-8)

### Fase 3 — REINFORCE com baseline
- `compute_returns`, `update_episode`, `finish_episode`, CSV, `env.randomize_sources()`

### Fase 4 — Currículo de autonomia
- 3 estágios: A (imitação, 10 eps) → B (híbrido, 20 eps) → C (autônomo)
- `brain.imitate()` (CE), `teacher_action()`, tecla A force autonomy

### Fase 5 — Métricas, memória e visualização
- `save_weights`/`load_weights` (JSON), HUD (`Text`), grade de setas (P), `--eval --episodes=N`

### Fase 6 — Robustez e calibração
- `temperature_decay`, `min_temperature`, `wall_margin` (fuga de bordas)
- `--set chave=valor` (override de CONFIG via CLI)
- `debug_sensores.py` com casos-limite (C1/C2/C3)
- `EXPERIMENTOS.md` (tabela de ciência)

## Fase 7 — Engenharia de recompensa (EM ANDAMENTO)

### Mudanças já feitas

**config.py:**
- `learning_rate`: 0.01 → **0.05**
- `entropy_coef`: 0.01 → **0.05**
- `reward_scale`: 1.0 → **5.0**
- `progress_scale`: 1.0 → **3.0**
- `arrival_bonus`: 0.5 → **5.0**
- `danger_penalty`: 0.5 → **5.0**
- `inside_rain_reward`: 0.01 → **0.5**
- `inside_light_penalty`: 0.02 → **0.5**
- `idle_cost`: 0.005 → **0.0** (desativado)
- `temperature_decay`: 0.99 → **0.995** (mais lento)
- `min_temperature`: 0.2 → **0.3** (mais alto)
- NOVOS: `proximity_rain_bonus=0.3`, `proximity_light_penalty=0.3`, `action_repeat_penalty=0.1`, `action_repeat_window=20`

**perception.py:**
- `calculate_reward()`: adicionado potencial de proximidade (1/dist) — sinal contínuo para puxar verme perto da chuva
- Anti-farniente condicional (`if CONFIG['idle_cost'] > 0`)

**main.py:**
- `state['action_history']` tracking (anti-colapso)
- Penalidade por repetição: se últimas 20 ações são iguais, `reward -= 0.1`
- Reset do `action_history` a cada episódio

**debug_sensores.py:**
- C3 atualizado: idle_cost desativado, teste espera `r >= 0`

### O que foi feito (Fase 8 — A2C)
- `mlp.py`: CriticNetwork (8→16→1, V(s)), compute_gae(), save_brain/load_brain
- `config.py`: value_coef=0.5, gae_lambda=0.95
- `main.py`: critic integrado no loop de treino + save/load combinado
- `test_mlp.py`: 12 testes (4 novos: critic, GAE, A2C update, brain save/load)
- Multi-seed eval (5 seeds × 100 eps): seed 3 = 45% (50 eps) / 39% (100 eps)
- A2C é 2x mais estável que REINFORCE (39% vs 20% eval longo)
- Política ainda colapsa para 1 ação, mas com mais variância que REINFORCE

### O que falta fazer (Fase 9 — se necessário)
1. **PPO** (Proximal Policy Optimization): clipping de gradientes, mais estável
2. **Treinar 200+ episódios** com A2C + multi-seed (pode convergir mais)
3. **Ou aceitar resultado** e documentar limitações (REINFORCE/A2C puros são fracos para este domínio)

### DoD (Definition of Done) do projeto
1. `teacher_influence = 0` permanente → ✅ (Estágio C + `--eval`)
2. ≥80% dos episódios: chegada à chuva + sem perigo → ⏳ (Fase 7 em andamento)
3. Tempo real + métricas sem travar → ✅ (HUD + CSV)

## Convenções do projeto

### Convensão de gradiente
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
- `é` funciona mas vira `�` no capture

### Arquivos importantes
- `config.py` — todas as constantes
- `mlp.py` — PolicyNetwork (MLP + REINFORCE)
- `main.py` — loop do jogo, currículo, HUD, editor
- `perception.py` — sensores + recompensa
- `worm.py` — corpo do verme
- `environment.py` — chuva/luz/fontes
- `test_mlp.py` — 8 testes (gradientes, save/load, imitate, returns)
- `debug_sensores.py` — headless, testa contrato recompensa + casos-limite
- `EXPERIMENTOS.md` — tabela de ciência
- `PLANO_DE_MELHORIAS.md` — roadmap completo
- `episodios.csv` — métricas por episódio (gerado pelo jogo)
- `pesos.json` — cérebro salvo

### Como rodar
```bash
# Treino com limite
python main.py --episodes=100

# Avaliação (carrega pesos.json, sem treino)
python main.py --eval --episodes=50

# Override de CONFIG
python main.py --episodes=100 --set=learning_rate=0.05 --set=gamma=0.95

# Testes
python test_mlp.py
python debug_sensores.py

# Compilação
python -m py_compile config.py mlp.py main.py perception.py worm.py environment.py
```
