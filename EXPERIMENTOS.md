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
| 4 | 2026-08 | 1–5 | 0.05 | 0.99 | 0.001 | 1.0→0.3 (decay 0.995) | 200 | A10/B20/C70 | **45% (9/20 C)** seed 3 | 10% (2/20 C) | — | **A2C** (CriticNetwork + GAE): seed 3 melhor. Avaliação 100 eps = **39%** (27/70 C). 2x mais estável que REINFORCE. |

**Diagnóstico final:** REINFORCE puro com softmax colapsa quando UMA ação gera reward positiva consistente. O gradiente acumulado reforça essa ação em todos os 200 passos, e a entropia/penalidade não são suficientes para quebrar o loop. **Solução necessária:** algoritmo de exploração mais forte (PPO, A2C com value function, ou DQN com ε-greedy).

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