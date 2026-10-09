# EXPERIMENTOS — tabela de ciência

Registro das experiências feitas no projeto. A cultura é: **o que mudou → o que
aconteceu → conclusão**. Cada linha deve ser reproduzível (a semente está no
`CONFIG['seed']`; qualquer treino é determinístico).

> ERAS DO PROJETO (leia antes de comparar números — **não são comparáveis entre si**):
> - **Era luz/chuva** (#1–14): sensores luz/chuva, H=200, mapa 16. Código removido.
> - **Era labirinto 11-dim** (#15–31): alimento + muros/pedras A/B, H=200.
> - **Era mapa 32** (#32–34): `map_limit` 32, H=400, ainda com muros/pedras.
> - **Era terreno limpo 8-dim** (#35+): sem obstáculos, H=400, CSV 13 cols
>   (`encontros_food`). **Canônico atual** — ver `PLANO_LABIRINTO_COMIDA.md §7`.

## Como rodar um experimento (era terreno limpo 8-dim)

```bash
# Treino com limite (A10+B20+C20 = 50 episodios), salva pesos e fecha
python main.py --episodes=50

# Avaliacao com N episodios (carrega pesos.json, politica near-greedy, sem treino)
python main.py --eval --episodes=50

# Variar hiperparametros sem editar codigo (reproduzivel com seed=42)
python main.py --episodes=50 --set=learning_rate=0.005 --set=gamma=0.95
python main.py --eval --episodes=50 --set=learning_rate=0.005 --set=gamma=0.95

# Headless rapido (sem GUI)
python train_food_headless.py --episodes=50 --seed=42 --csv=baseline_limpo.csv
```

- `--episodes=N` funciona em **treino** e **avaliacao** (salva `pesos.json` e fecha).
  ⚠️ Em `--eval` ele SOBRESCREVE `pesos.json` ao fechar — faça backup antes.
- `--eval` carrega automaticamente `pesos.json` e usa `min_temperature` (politica decisiva).
- `--set=chave=valor` sobrescreve o `CONFIG` sem editar codigo. Chaves removidas
  (`difficulty`, `obstacle_*`) são aceitas em silêncio e **não fazem nada** — não use.
- Metricas por episodio ficam em `episodios.csv` (13 cols: `encontros_food`,
  `passos_ate_comer`; sem `colisao_obs`). Curvas das eras antigas (15 cols) em
  `docs/historico/csv/`; pesos antigos em `docs/historico/pesos/` — ver
  `docs/historico/README.md`. Não comparar eras entre si.

## Tabela de experimentos

| # | Data | Semente | α | γ | λ_reg | Temperatura | H | Estágios | Chegada chuva | Perigo luz | Recompensa média | Observações |
|---|------|---------|-----|-----|-------|-------------|-----|----------|---------------|------------|------------------|-------------|
| 1 | 2026-08 | 42 | 0.01 | 0.99 | 0.001 | 1.0→0.2 (decay 0.99) | 200 | A10/B20/C70 | 30% (6/20 C) | 25% (5/20 C) | +0.01 | POLÍTICA TRAVADA: ação `frente` em 100% dos episódios C. RL não convergiu em virar. Necessário Fase 7. |
| 2 | 2026-08 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.3 (decay 0.995) | 200 | A10/B20/C70 | 30% (6/20 C) | 25% (5/20 C) | +0.01 | Fase 7: rewards 5x, entropy 5x, proximity 1/dist, repeat penalty. Colapsou p/ `esquerda`. |
| 3 | 2026-08 | 1–5 | 0.05 | 0.99 | 0.001 | 1.0→0.3 (decay 0.995) | 200 | A10/B20/C70 | **40% (8/20 C)** seed 5 | 15% (3/20 C) | — | Multi-seed: seed 5 melhor. Avaliação 100 eps = 20% (14/70 C). Política colapsa p/ 1 ação. |
| 4 | 2026-08 | 1–5 | 0.05 | 0.99 | 0.001 | 1.0→0.3 (decay 0.995) | 200 | A10/B20/C70 | **45% (9/20 C)** seed 3 | 10% (2/20 C) | — | **A2C** (CriticNetwork + GAE): seed 3 melhor. Eval 100 eps = **39%** (27/70 C). |
| 5 | 2026-08 | 1–5 | 0.05 | 0.99 | 0.001 | 1.0→0.3 (decay 0.995) | 200 | A10/B20/C70 | **40% (8/20 C)** seed 2 | 25% (5/20 C) | — | **PPO** (clip=0.2): seed 2 melhor. Eval 100 eps = **40%** (28/70 C). PPO ≈ A2C, ambos 2x REINFORCE. |
| 6 | 2026-08 | 1–5 | 0.05 | 0.99 | 0.001 | 1.0→0.3 (decay 0.995) | 200 | A10/B20/C70 | 25-35% (bug) | — | — | **Replay Buffer** (pré-fix): ratio=1.0, critic=advantage → clipping inoperante; overflow no ep.34. |
| 7 | 2026-08 | 3 | 0.05 | 0.99 | 0.001 | 1.0→0.3 (decay 0.995) | 200 | A10/B20/C70 | — | — | — | **Replay Buffer pós-fix**: 15/15 testes OK, 4.3s/ep (7× mais caro). 200 eps ≈14 min/seed. Pendente multi-seed completo (nightly). |
| 8 | 2026-08 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | A10/B20/C70 | 15% (3/20 C) | 30% (6/20 C) | — | **Fase 10 treino longo** (100 eps, dif 1, decay 0.998): colapsou p/ `direita` 200/200, entropia 0.000. Mesmo com 500 eps (253 eps parcial) colapso persiste. Obstáculos dificultam — sem Fase 11/12 não há ganho. |
| 9 | 2026-08 | — | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | A10/B20/C70 | — | — | — | Laboratorio 32×32 (72x72 ground), obstáculos com muros Fase 15+, Aquario --visual separado. |
| 10 | 2026-08 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | A10/B20/C70 | **0% (0/20 C)** | 40% (8/20 C) | — | **Fase 11 deep 11→32→16→5**: 100 eps dif 1, colapso `esquerda` 200/200, 0% seguro vs 15% shallow. Rede 1.7k precisa mais dados. |
| 11 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | headless PPO puro 35eps | 5.1% steps (média) | 13.9% steps (média) | — | **Fase 12 validação integração 17-dim (headless, sem GUI/professor/obs)**: 35×200 passos, sem crash, shapes OK, entropia→0 (colapso esperado em random-walk). Treino GUI 500+ eps pendente. |
| 12 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | A10/B20/C5 treino + 20 eval | 15% (3/20 seguro) | 4.4% média steps | — | **Fase 12 GUI 17→32→16→5, dif FACIL**: treino 35eps C=1/5 seguro (ep35 0.10/0.00), eval 20eps 3/20 seguro (média chegada 0.053). Política colapsa (entropia 0.000, só esquerda/direita). 17-dim não resolve com 35eps — precisa 500+ eps. |
| 13 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | eval 100 (pesos #12) | 10% (10/100 seguro) | 4.1% média perigo | — | **Fase 14 DoD (1 seed)**: 14/100 chegada>0, 10/100 seguro (chegada>0 e perigo==0). Entropia 0.000. DoD ≥80% NÃO atingido. |
| 14 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | A10/B20/C389 (419/500, timeout) | 12.6% C seguro (49/389) | — | — | **Fase 10 treino longo 17-dim (parcial)**: 419eps em 60min, interrompido no ep417. Últimos 19eps 3/19 seguro. Colapso persiste (direita 200/200, entropia 0). Ganho marginal 10%→12.6%. Pesos não salvos (sem conclusão). |
| 15 | 2026-09 | 42→99 | 0.05 | 0.99 | 0.001 | 1.0→0.2 + whitening + eps0.1 + λc0.05 | 200 | **Fase 14 headless alimento**: treino 100eps maze A c/ currículo (44% encontro, C 1/5) → eval 100eps A **17%** (prof 66%) e B **17%** (prof 80%). Relativo A 26% / B 21% (meta 80% do prof). DoD NAO atingido; política = acaso (20%). Pesos `pesos_food_A100.json`. |
| 16 | 2026-09 | 99 | — | — | — | — | 200 | **Teto do professor headless** (autonomy=0, uniforme): A **66%**/1.32ep, B **80%**/1.56ep (100eps). B varia 60–80% entre runs — oráculo instável; DoD absoluto ≥80% em B é loteria. DoD relativo ao professor é a métrica correta. |
| 17 | 2026-09 | 42→99/123 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **Gate F14 15-dim enxuto** (`gate_f14.py`): treino 35eps A10/B20/C5 seed42 9s -> `pesos_gate15.json` → eval 20eps seed42 **15%**, 99 **15%**, 123 **20%** (media 16.7%, GAP 5pp). DoD≥80% NAO; overfit NAO. Mantém 15% do 17-dim (#12) com -5.5% params (2310→2182). |
| 18 | 2026-09 | 42→99 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L4 alimento currículo puro** (sem eps/lambda, `food_curriculum=1`): treino 100eps A 34%/0.41ep mediana 126 → eval A **15%** e B **11%** (prof B 64% hoje; relativo B 17%). Colapso total (top 0 200/200, ent 0.000). Pior que #15 (21% relativo c/ eps0.1+λc0.05). Pesos `pesos_food_L4.json`. |
| 19 | 2026-09 | 42/1/2/3→99 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L5 ablação 4 configs × 4 seeds** (16 runs): sem currículo (30-38% A, 11-17% B), com currículo (33-41% A, 11-17% B), sem fome (27-34% A, 11-17% B), sem ambos (27-34% A, 11-17% B). **Colapso total em 16/16 runs** (ent 0.000, ação dominante 200/200). Currículo e fome NÃO evitam colapso. Seeds diferentes → políticas similares (sempre ação 0 ou 4). Eval B 11-17% vs professor 64-80%. **Overfitting A**: treino A 27-41% mas eval B 11-17%. Pesos `pesos_L5_*.json`. |
| 20 | 2026-09 | 42→99 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L7 anti-colapso 3 configs** (seed 42): Config 1 (entropy=0.15, min_T=0.4, lr_decay=0.999) 31% A / 11% B; Config 2 (entropy=0.2, min_T=0.5, lr_decay=0.999) 31% A / 11% B; Config 3 (entropy=0.15, min_T=0.4, lr_decay=0.999, repeat_penalty=0.2) 31% A / 11% B. **Colapso total em 3/3 configs** (ent 0.000, ação 0 200/200). Anti-colapso estrutural FALHOU — parâmetros de exploração não são suficientes. Pesos `pesos_L7_config*.json`. |
| 21 | 2026-09 | 42→99 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L8 recompensa densa 3 configs** (seed 42): Config 1 (smell=0.5, progress=1.5) 28% A / 17% B; Config 2 (smell=1.0, progress=1.0) 28% A / 17% B; Config 3 (smell=0.5, progress=2.0) 31% A / 11% B. **Colapso total em 3/3 configs** (ent 0.000, ação dominante 200/200). Recompensa densa FALHOU — problema é estrutural. Pesos `pesos_L8_config*.json`. |
| 22 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L9 novelty bonus** (seed 42, 100 eps): novelty_bonus=0.2, novelty_decay=0.99, novelty_window=100. Treino A 31% encontros. Entropia final 0.000 (colapso estágio C), MAS entropia 0.5-0.6 nos eps 60-80 (vs 0.000 sem novelty). Novelty bonus atrasa colapso mas não evita. Implementado em `config.py` + `perception.py`. Pesos `pesos_L9_novelty.json`. |
| 23 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L10 action repeat penalty** (seed 42, 100 eps): action_repeat_penalty=0.2, action_repeat_window=50. Treino A 33% encontros. Entropia final 0.000 (colapso estágio C), MAS entropia 0.6-0.7 nos eps 22-29 (vs 0.000 sem penalidade). Penalidade por repetição atrasa colapso mas não evita. Implementado em `perception.py` + `train_food_headless.py`. Pesos `pesos_L10_repeat.json`. |
| 24 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L11 arquitetura robusta** (seed 42, 100 eps): n_hidden=64, n_hidden2=32 (rede 11→64→32→5, ~3.5k params). Treino A 29% encontros. Entropia final 0.000 (colapso estágio C). Rede maior NÃO evita colapso — problema é estrutural. Pesos `pesos_L11_big.json`. |
| 25 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L12 labirintos aleatórios** (seed 42, 100 eps, --random-maze): treino A 40% encontros (vs 27-41% L5). Entropia final 0.000 (colapso estágio C). Labirintos aleatórios NÃO evitam colapso, mas ajudam generalização (40% treino A). Implementado em `train_food_headless.py`. Pesos `pesos_L12_random.json`. |
| 26 | 2026-09 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 200 | **L13 avaliação final** (seed 42, 100 eps): combinação novelty_bonus=0.2 + action_repeat_penalty=0.2 + n_hidden=64 + n_hidden2=32 + random_maze. Treino A 48% encontros (vs 27-41% L5). Entropia final 0.000 (colapso estágio C). Combinação NÃO evita colapso, mas ajuda generalização (48% treino A). Pesos `pesos_L13_config1_s42.json`. |
| 27 | 2026-10 | 42→99 | — | — | — | — | 200 | **L14a BC puro smoke** (`--bc`, 30 eps A seed42 → eval 20 eps B seed99): treino 50%/0.73ep (movimento=professor) → eval B **0%** (ação 2 200/200, ent 0.000). Até supervisionado colapsa no shift treino-professor/eval-próprio. Pesos `pesos_L14_bc.json`. |
| 28 | 2026-10 | 42 | 0.05 | 0.99 | 0.001 | ε=0.3 externo | 200 | **L14b DQN smoke** (`--dqn`, 15 eps A seed42): treino 6.7%/0.07ep, tops variam (0/3/1/2/4, 134–160/200) — sem colapso único no treino; eval B 0% greedy (15 eps insuficientes). QNetwork TD c/ sinal correto + alvo sync=10. Pesos `pesos_L14_dqn.json`. |
| 29 | 2026-10 | 42 | 0.05 | 0.99 | 0.001 | ε=0.3 / BC | 200 | **L15 consolidação smoke** (8 eps A seed42, pós-fix): BC 37.5%/0.38ep tops honestos (200/200,126/200…); DQN 0%/0.00ep tops diversos (0/1/2) sem explosão (lr médio). Fixes: s_next pré-respawn + done=comer, act_count executada, lr/N, load checa n_outputs, guards epsilon/temperatura/bc+dqn. `test_mlp.py` 16/16. |
| 30 | 2026-10 | 42 | 0.05 | 0.99 | 0.001 | ε=0.3 / BC | 200 | **L16 protocolo smoke** (8 eps A seed42): BC 50%/0.50ep (professor c/ borda+muros), DQN 0% tops diversos sem explosão (q_scale=1.0). CSV 15 cols = GUI validado (`--csv=`), `test_mlp.py` 16/16 c/ bootstrap. Pronto p/ 100eps comparáveis (PPO vs BC vs DQN + eval B + teto). |
| 31 | 2026-10 | 42 | 0.05 | 0.99 | 0.001 | PPO/BC/ε=0.3 | 200 | **L16-refaz avaliação** (8 eps A seed42, pós-paridade): PPO 37.5%, BC 37.5% (A/BC autonomy=0, confere), DQN 0% diverso. Base verde (`test_mlp` 16/16, `debug_food` OK pós-C1-novidade, `debug_maze` OK). Refazes: muros 1pto=GUI, limit CONFIG, L2, `q_reward_scale` central, aviso sync, C1 novidade. |
| 32 | 2026-10 | 99→42 | — | — | — | professor/PPO/DQN | 400 | **Mapa 32 (pré-L17)**: limit 16→32, muros A 2→3/B 3→4, pedras A 1→2/B 0→1, foods espalhados, H 200→400, `sensor_max_dist` 60, currículo 10+0.3/ep (teto 24), fome 0.001. Teto professor B 90%/1.20ep; PPO-A 100% estágio A (colapso ep4 ent→0); DQN 0% diverso. Pesos/smokes antigos incomparáveis (reescala, mesma arquitetura 11-dim: carregam, mas jogam outro jogo). |
| 33 | 2026-10 | 42 | 0.05 | 0.99 | 0.001 | PPO/ε=0.3 | 400 | **Revisão mapa 32 (compat total)**: 7 fallbacks atualizados (perception/environment/headless/maze/C1→32/60/24/10/0.3/0.001), `novelty_window` 100→200, `max_zoom` 80→140, docs L2/L3. Verdes: `test_mlp` 16/16, `debug_food` OK, `debug_maze` OK, smoke PPO 100%/DQN diverso + CSVs. `action_repeat_window` 20 mantida (sensibilidade), físicas absolutas verificadas. |
| 34 | 2026-10 | 42→99 | 0.05 | 0.99 | 0.001 | PPO/BC/ε=0.2 | 400 | **L17 veredito mapa 32** (100eps A s42 → eval 100eps B s99): treino PPO 36%/0.47, BC 80%/1.23 (movimento=professor), DQN 6%/0.06; eval B: PPO **8%** (colapso 0-400/400 ent 0.000), BC **5%** (diverso ent 0.43, tops 0/2/4), DQN **7%** (greedy 2, mediana 40 passos); teto professor **89%**/1.32ep. Relativo: 9%/6%/8% (meta 80%). Gate: tudo ≤21% → limite do RL puro aceito, treinos longos congelados. Pesos `pesos_L17_*.json` + 7 CSVs. |
| 35 | 2026-10 | 42 | 0.05 | 0.99 | 0.001 | 1.0→0.2 (decay 0.998) | 400 | **Baseline terreno limpo 8-dim** (PPO 50eps seed42, sem curriculo, H=400): **56%** (28/50), media 0.78/ep; estagio C 8/20=**40%** (ent 0.000, colapso acao 0 esquerda 400/400); passos-ate-1o mediana **163**/min 16/media 189. Colapso persiste SEM obstaculos (ent→0 ja no ep5) — o labirinto nao era a causa. CSV `baseline_limpo.csv` (13 cols, `encontros_food`). |

**Diagnóstico final:** REINFORCE puro com softmax colapsa. A2C/PPO dobram (20%→40%) mas **Fase 10 (decay 0.998, 500 eps) não resolve colapso** — política ainda converge p/ 1 ação (direita) com 4 pedras. Obstáculos + mapa maior aumentam dificuldade. Requer **Fase 11 (rede maior) + 12 (estado 17-dim)** para representar desvio.

> Atualização terreno limpo (#35, 2026-10): o diagnóstico acima era do mundo com
> muros/pedras. No 8-dim limpo o colapso **persiste** (ent 0.000, 1 ação) — as
> pedras não eram a causa raiz. "Fase 11+12 p/ desvio" está superado: não há mais
> desvio a representar (sem obstáculos) e a 17-dim foi revertida. Hipótese vigente:
> colapso estrutural do policy-gradient softmax on-policy com poucos dados
> (50eps×400 = 20k passos). Próximo teste barato: `--dqn` / `--bc` no 8-dim.

> Instruções: após rodar um treino, preencha uma linha com `% com encontro`,
> `passos_ate_comer` (mediana/min/média) e `media encontros/ep` dos episódios finais
> (era terreno limpo; colunas `chegada_chuva`/`perigo_luz` não existem mais).

## Robustez e casos-limite (Fase 6)

| Caso | Comportamento esperado | Status |
|------|------------------------|--------|
| Sem fontes no mapa | Sensores zerados, recompensa só-novidade (neutra com `novelty_bonus=0`, sem crash) | ✅ `debug_food.py` C1 |
| Fonte além do alcance | Distâncias saturam em `1.0` (sem NaN/overflow) | ✅ `debug_food.py` C2 |
| Verme parado | Sem punição (`idle_cost=0`, `r >= -0.05`) | ✅ `debug_food.py` C3 |
| Verme perto da borda | Professor foge da parede (anti-encalhe) | ✅ fuga de bordas em `get_target_direction` |

## Anti-saturação

- `tanh` do MLP tem **clamp em ±500** na pré-ativação (evita overflow).
- Entradas já nascem normalizadas (direções em `[-1,1]`, distâncias em `[0,1]`).
- Bônus de **entropia** (`entropy_coef`) + **decay de temperatura** **atrasam mas não
  impedem** o colapso numa única ação (prova: colapso 16/16 em #19–26 e #35, ent 0.000);
  a temperatura tem piso (`min_temperature`) para a política nunca ficar 100% greedy.

## Exploração estruturada

- Ação **amostrada** da softmax (não argmax) — exploração por amostragem.
- `temperature_decay`: a cada episódio de treino `T ← max(min_T, T·decay)`.
- No `--eval`, a temperatura **não** decai — a avaliação usa a política treinada.
- Exploração vigente extra: `--set=epsilon_greedy` (hook opt-in), `--bc` (professor),
  `--dqn` + `--dqn-epsilon` (ε-greedy externo, sem softmax).
## Congelamento (2026-09-24) + reabertura parcial (2026-10-08)

Treino encerrado **na era labirinto 11-dim**. Veredito: DoD relativo não atingido
em nenhuma trilha daquela era (legado 10% absoluto #13; alimento 21–26% do
professor #15; teto do oráculo A66/B80 #16). Fixes estruturais travados
(whitening, `epsilon_greedy`, `lambda_c` opt-in, currículo de distância,
anti-sorte de spawn) — todos default-safe, regressão verde (`test_mlp.py 16/16`,
`debug_food/maze`).

Reabertura 2026-10: o congelamento **não vale p/ era terreno limpo 8-dim** —
números antigos incomparáveis (outro estado, outro mapa, outro CSV). Roda-se
baseline novo e barato (#35) antes de qualquer treino longo.

Reprodução (era atual):
```bash
python test_mlp.py && python debug_food.py && python debug_maze.py
python train_food_headless.py --episodes=50 --seed=42 --csv=baseline_limpo.csv
python main.py --episodes=3   # smoke GUI
```
