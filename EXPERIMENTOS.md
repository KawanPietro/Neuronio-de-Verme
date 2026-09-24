# EXPERIMENTOS — tabela de ciência

Registro das experiências feitas no projeto. A cultura é: **o que mudou → o que
aconteceu → conclusão**. Cada linha deve ser reproduzível (a semente está no
`CONFIG['seed']`; qualquer treino é determinístico).

## Como rodar um experimento

```bash
# Treino com limite (A10+B20+C5 = 35 episodios), salva pesos e fecha
python main.py --episodes=35

# Avaliacao com N episodios (carrega pesos.json, politica near-greedy, sem treino)
python main.py --eval --episodes=50

# Variar hiperparametros sem editar codigo (reproduzivel com seed=42)
python main.py --episodes=35 --set=learning_rate=0.005 --set=gamma=0.95
python main.py --eval --episodes=50 --set=learning_rate=0.005 --set=gamma=0.95
```

- `--episodes=N` funciona em **treino** e **avaliacao** (salva `pesos.json` e fecha).
- `--eval` carrega automaticamente `pesos.json` e usa `min_temperature` (politica decisiva).
- `--set=chave=valor` sobrescreve o `CONFIG` sem editar codigo.
- Metricas por episodio ficam em `episodios.csv`.

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

**Diagnóstico final:** REINFORCE puro com softmax colapsa. A2C/PPO dobram (20%→40%) mas **Fase 10 (decay 0.998, 500 eps) não resolve colapso** — política ainda converge p/ 1 ação (direita) com 4 pedras. Obstáculos + mapa maior aumentam dificuldade. Requer **Fase 11 (rede maior) + 12 (estado 17-dim)** para representar desvio.

> Instruções: após rodar um treino, preencha uma linha com a média de
> `chegada_chuva`/`perigo_luz` e a recompensa média dos episódios finais.

## Robustez e casos-limite (Fase 6)

| Caso | Comportamento esperado | Status |
|------|------------------------|--------|
| Sem fontes no mapa | Sensores zerados, recompensa neutra (sem crash) | ✅ `debug_sensores.py` C1 |
| Fonte além do alcance | Distâncias saturam em `1.0` (sem NaN/overflow) | ✅ `debug_sensores.py` C2 |
| Verme parado | Paga o custo anti-farniente (`r < 0`) | ✅ `debug_sensores.py` C3 |
| Verme perto da borda | Professor foge da parede (anti-encalhe) | ✅ fuga de bordas em `get_target_direction` |

## Anti-saturação

- `tanh` do MLP tem **clamp em ±500** na pré-ativação (evita overflow).
- Entradas já nascem normalizadas (direções em `[-1,1]`, distâncias em `[0,1]`).
- Bônus de **entropia** (`entropy_coef`) + **decay de temperatura** impedem o
  colapso numa única ação; a temperatura tem piso (`min_temperature`) para a
  política nunca ficar 100% greedy.

## Exploração estruturada

- Ação **amostrada** da softmax (não argmax) — exploração por amostragem.
- `temperature_decay`: a cada episódio de treino `T ← max(min_T, T·decay)`.
- No `--eval`, a temperatura **não** decai — a avaliação usa a política treinada.
## Congelamento (2026-09-24)

Treino encerrado. Veredito: DoD relativo não atingido em nenhuma trilha
(legado 10% absoluto #13; alimento 21–26% do professor #15; teto do oráculo
A66/B80 #16). Fixes estruturais travados (whitening, `epsilon_greedy`,
`lambda_c` opt-in, currículo de distância, anti-sorte de spawn) — todos
default-safe, regressão verde (`test_mlp.py 15/15`, `debug_food/sensores/maze`).

Reprodução:
```bash
python test_mlp.py && python debug_food.py && python debug_sensores.py && python debug_maze.py
python train_food_headless.py --episodes=100 --maze=A --seed=42 --set=food_curriculum=1 --save=pesos_food.json
python train_food_headless.py --episodes=100 --maze=B --seed=99 --eval --weights=pesos_food.json
python main.py --maze=A --episodes=3 --set=food_curriculum=1   # smoke GUI
```
