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
| 0 | Higiene do código | Base limpa e configurável para trabalhar | M | ✅ concluída |
| 1 | Contrato de comportamento | Sensores vão além de escalares; recompensa vira "gradiente" claro | M | ✅ concluída |
| 2 | Cérebro de verdade | MLP com backprop; política estocástica (escolha de ação) | M | ✅ concluída |
| 3 | REINFORCE com baseline | Aprendizado real por Policy Gradient → convergência | S | ✅ concluída |
| 4 | Currículo de autonomia | Do professor ao piloto automático; annealing do professor | M | ✅ concluída |
| 5 | Métricas e memória | Medir, salvar e visualizar o aprendizado (cultura de dados) | S | ✅ concluída |
| 6 | Robustez e calibração | Tuning, exploração, correção de casos-limite | S | ✅ concluída |

> Estimativa de esforço relativa: S = pequena (1–2 sessões), M = média (2–4), L = grande (4+).

**Critério de conclusão (Definition of Done) do projeto:**
1. `teacher_influence = 0` de forma permanente. ✅ (Estágio C + tecla `A` + `--eval`)
2. Na avaliação (configs fixos e "nunca vistos"), o verme chega à chuva e **não** entra na zona de perigo em ≥ 80% dos episódios, ao longo de várias sementes aleatórias. ⏳ **validação final**: `python main.py --eval --episodes=50` após um treino completo (ver `EXPERIMENTOS.md`).
3. O jogo roda em tempo real e exibe métricas de aprendizado sem travar. ✅ (HUD + grade de setas + CSV)

---

## Fase 0 — Higiene do código ✅ concluída

**Para quê:** antes de ensinar novas técnicas ao cérebro, a base precisa ser limpa. Hoje há código morto e constantes mágicas espalhadas.

**Tarefas:**
- [x] Remover o `update()` duplicado em `Verme.py`. Unificar o loop em uma única versão (agora em `main.py`).
- [x] Mover constantes para um dicionário único `CONFIG` em `config.py` (`num_segments`, `speed`, `arrival_radius`, `autonomy_scale`, `sensor_max_dist`, `learning_rate`, `regularization`, `cam`, ...).
- [x] Modularizar: separar em arquivos coesos:
  - `config.py` — todas as constantes e a `CONFIG` global.
  - `Rede_Neural.py` — apenas a rede neural (demo movida para `if __name__ == '__main__'`).
  - `environment.py` — luzes, chuva, partículas, colisões (classe `Environment`).
  - `worm.py` — o corpo do verme (cabeça + segmentos + animação, classe `Worm`).
  - `main.py` — orquestra o jogo (`Ursina()`, `update()`, `input()`); `Verme.py` virou shim de compatibilidade.
- [x] Adicionar `.gitignore` (exclui `__pycache__`, `*.pyc`, `.venv/`, `venv/`) e `requirements.txt`.
- [x] Corrigir compatibilidade: usar `camera.raycast(...)` no editor (mais seguro em Ursina novo).
- [x] Bônus: prints do terminal agora são ASCII-safe (caracteres `─`, `→` falhavam em terminais cp1252 no Windows).

**Conceito didático:** *separação de responsabilidades* e *configuração declarativa* — mudar um número não deve exigir caçar o código.

**Critério de aceite:** o jogo roda com o mesmo comportamento visual, mas sem código morto; todas as constantes em um só lugar.

---

## Fase 1 — Contrato de comportamento (sensores + recompensa) ✅ concluída

**Para quê:** um agente RL só converge se (a) o estado contiver info suficiente para agir e (b) a recompensa indicar *progresso*, não só posição absoluta.

### 1.1 Sensores ricos (substituem os 2 escalares) ✅

Novo estado com geometria, implementado em `perception.py` (`get_sensor_inputs`):

| Feature | Descrição | Dimensões |
|---------|-----------|-----------|
| `luz_dir` | Vetor XZ normalizado da cabeça até a fonte de luz mais próxima | 2 |
| `luz_dist` | Distância normalizada `[0,1]` até essa fonte | 1 |
| `luz_perigo` | 1 se dentro do `ARRIVAL_RADIUS` (danger), senão 0 | 1 |
| `chuva_dir` | Vetor XZ normalizado até a fonte de chuva mais próxima | 2 |
| `chuva_dist` | Distância normalizada até essa fonte | 1 |
| `pulso_chuva` | Vibração (sinal do "tato") | 1 |

Total: **8 entradas**. A `WormBrain` foi generalizada (arquitetura configurável) para receber as 8 features (`8 → 16 → 3`).

### 1.2 Recompensa por progresso (shaping) ✅

Recompensa por **passo** (cada frame), em `perception.py` (`calculate_reward`):

```
r = + progresso_em_direção_à_chuva  (Δ distância curvada)
    − progresso_em_direção_à_luz     (aproximar da luz é ruim)
    + BONUS  ao chegar na chuva (entrar no ARRIVAL_RADIUS)
    − PENALIDADE ao entrar na zona de perigo da luz
    − pequeno CUSTO de ficar parado/andando em círculo (anti-farniente)
```

- Usado **Δdistância** (melhora vs. passo) — o sinal diz "ficou melhor" mesmo sem evento.
- Intervalo `[-1, +1]` normalizado.
- A "órbita" foi **removida** do professor (`get_target_direction`): o comportamento desejado é **chegar à chuva e permanecer perto**, e fugir da luz.
- `autonomy_scale` rebalanceado (30 → 120) para a recompensa por frame.

**Critério de aceite:** ✅ `debug_sensores.py` (headless) imprime `[sensores 8-dim] + r` por passo e confirma que `r` é maior quando o verme se aproxima da chuva (média +0.18 aproximando vs −0.21 afastando no cenário de teste).

---

## Fase 2 — Cérebro de verdade (MLP + política) ✅ concluída

**Para quê:** a regra atual (`lr × recompensa × entrada × saída`) não é um gradiente de verdade — não converge de forma confiável. Vamos construir um MLP reutilizável com **backpropagation** e um **cabeçote de política**.

### 2.1 MLP com backprop (classe reutilizável) ✅

- Arquitetura configurável: `8 → 16 → num_ações`.
- Guardar as ativações intermediárias no `forward` (para o `backward`).
- Funções de ativação: `tanh` e `relu` à escolha (didático: mostrar a diferença).
- Implementar `backward(target_ou_grad)` com as regras da cadeia, **camada a camada** (SEM loops genéricos de matriz — preservar o didatismo).
- ✅ **Implementado em `mlp.py`**: classe `MLP` (`forward`, `backward_from_output_grad`, `apply_gradients`, `grads`) e classe `PolicyNetwork` (softmax + amostragem + `update` com policy gradient e bônus de entropia).

### 2.2 Política estocástica — espaço de ação ✅

Recomendação: **ações discretas** (muito mais didáticas para policy gradient):

| Ação | Efeito |
|------|--------|
| `esquerda` | girar −θ no plano XZ |
| `frente_esquerda` | girar −θ/2 |
| `frente` | seguir em frente |
| `frente_direita` | girar +θ/2 |
| `direita` | girar +θ |

(5 ações; velocidade constante. Se quiser, 3×3 = `[velocidade] × [direção]`.)

- Saída: **softmax** sobre as ações → distribuição de probabilidade `π(a|s)`.
- Ação escolhida por **amostragem** (exploração) — é isso que permite aprender por tentativa.
- ✅ **Integrado em `main.py`**: `brain.sample_action(sensors)` → giro discreto `{−θ, −θ/2, 0, +θ/2, +θ}` com `turn_rate` (rad/s); `brain.update(reward, action, sensors)` por frame; `Rede_Neural.py` marcado como legado.

> Alternativa contínua (mais difícil): gaussiana sobre `(vx, vz)` com log-prob e reparametrização. Fica como *desafio opcional* no fim de Fase 3.

**Conceito didático:** *backpropagation na mão* e *política estocástica* (por que amostrar em vez de pegar o argmax).

**Critério de aceite:** um teste unitário manual mostra que `backward` calcula o gradiente correto comparado à estimativa numérica de diferenças finitas.
✅ **`test_mlp.py` aprovado** — 4 verificações OK (regressão MSE e política `−log π`, com `tanh` e `relu`), erro relativo máximo ~1e-8.

**Observação (ponte para a Fase 3):** o `update` atual é um *policy gradient por passo* (`r·∇logπ + β·∇H`), válido como demonstração, mas sofre de alta variância. A Fase 3 substitui por episódios + retornos descontados + baseline. ✅ **Implementado em `update_episode()`.**

---

## Fase 3 — REINFORCE com baseline ✅ concluída

**Para quê:** este é o coração do verme autônomo. Implementação clássica de Williams (1992), do zero.

```
Loop de treino:
  coletar H passos de um episódio  →  estado, ação, recompensa
  computar retornos descontados    G_t = Σ γ^k · r_{t+k}
  baseline                        b = média dos retornos do episódio
  gradiente                       ∇θ J ≈ Σ (G_t − b) · ∇θ log π(a_t|s_t) + β·∇H(π)
  atualizar pesos                 θ ← θ + α · gradiente − α·λ·θ  (L2)
```

- **Baseline** reduz a variância (lição central: "compare com a média, não com número absoluto"). ✅ `b = média(G)` em `update_episode`.
- **γ (desconto)** ensino de horizonte: ações perto da recompensa pesam mais. ✅ `compute_returns(rewards, γ)`.
- Extras didáticos no caminho:
  - termo de **entropia** para manter exploração → impede colapso prematuro numa ação só; ✅ `β·∇H` somado ao gradiente.
  - **decay do α** (learning rate schedule); ✅ `brain.learning_rate *= lr_decay` a cada episódio.
  - `reward scaling` para tirar o jogo de regime. ✅ parâmetro `reward_scale` (default 1.0).
- **Sessões episódicas**: cada episódio posiciona fonte de luz/chuva em posições **aleatórias** e o treino roda depois de `H` passos. O verme aprende a generalizar (não decora um cenário). ✅ `env.randomize_sources()` em `finish_episode()`.
- No fim do episódio → opcionalmente **resetar** a posição/`history` e pedir nova cena (para diversidade). ✅ verme volta ao centro em posição aleatória.

**Conceito didático:** *política estocástica + gradiente do log-prob* + *redução de variância* — a base de todo RL moderno (REINFORCE, A2C, PPO).

**Critério de aceite:** a curva de recompensa média por episódio **sobe** (log/CSV), e com o professor desligado o verme já mostra, sozinho, tendência de ir à chuva.

✅ **`episodios.csv`** registra `recompensa_total, recompensa_media, retorno_medio, entropia_media, learning_rate, autonomia_forcada` por episódio. Teste manual com 3 episódios: média subiu `+0.008 → +0.052` (mesmo com cenas aleatórias). Tecla **A** desliga o professor (autonomia forçada = 1) para avaliar o verme sozinho. `test_mlp.py` validou `compute_returns` e `update_episode`.

---

## Fase 4 — Currículo de autonomia (tirar o professor) ✅ concluída

**Para quê:** os pasos de *behavioral cloning* bootstrap rápido e a Fase 3 finaliza sem professor.

### Currículo em 3 estágios

1. **Estágio A — Imitação (warm-up):** o professor demonstra `H` passos (estado → ação ideal, via campo de potencial). Treinar por **cross-entropy supervisionada** (imitation learning). Rede começa com pesos "bons". ✅ `brain.imitate(sensors, acao_professor)` — um passo de CE por passo do episódio; `teacher_action()` converte a direção do professor na ação discreta ideal.
2. **Estágio B — Híbrido (REINFORCE + professor):** continuar com policy gradient, mas com a **entrada do professor como "ação de referência"** que entra na função de perda com peso `λ_imitação` que **decai** a cada episódio:
   `loss = −REINFORCE + λ_imitação · CE(π, ação_professor)` ✅ `update_episode(imitation_weight=λ)`; autonomia do movimento sobe `0→1` ao longo do estágio.
3. **Estágio C — Autonomia plena:** `λ_imitação = 0`, `teacher_influence = 0`. A rede controla tudo. Simular vários episódios em posições variadas. ✅ movimento 100% rede; tecla **A** força esse modo a qualquer momento.

Analogy didática: aula com professor, exercícios com tutor dizendo "a resposta é X", e então prova sem ajuda.

**Critério de aceite:** rodar 50 episódios no estágio C e registrar: taxa de chegada à chuva, taxa de entrada no perigo e recompensa média. Devem bater o DoD do projeto.
✅ O CSV (`episodios.csv`) registra por episódio: `estagio, recompensa_*, retorno, entropia, learning_rate, lambda_imitacao, chegada_chuva, perigo_luz`. Validação manual do estágio A: imitação reduz a entropia da política (o verme "cola" no professor) e o verme já alcança a chuva. O `test_mlp.py` validou `imitate` (CE diminui) e o termo híbrido de `update_episode`.

---

## Fase 5 — Métricas, memória e visualização ✅ concluída

**Para quê:** "autônomo" precisa ser *mensurável*. E o projeto é didático — **ver** o que a rede aprendeu é tesouro.

- [x] **Log/CSV**: por episódio — `episodio, estagio, recompensa_total, recompensa_media, retorno_medio, entropia_media, learning_rate, lambda_imitacao, autonomia_forcada, chegada_chuva, perigo_luz, acao_principal, semente`.
- [x] **Painel HUD no Ursina**: episódio, estágio do currículo, recompensa média rolling (últimos 10), entropia, learning rate, modo treino/avaliação.
- [x] **Visualização da política aprendida**: grade de setas sobre o mapa (tecla `P`) — em cada célula, a ação mais provável; verde = direção p/ chuva, vermelho = p/ luz.
- [x] **Salvar/carregar pesos**: `save_weights`/`load_weights` (JSON, `pesos.json`), teclas `S`/`L`, e salvamento automático no fim do `--eval`.
- [x] **Modo avaliação**: `python main.py --eval [--episodes=N]` — professor sempre desligado, **sem treino**, com semente fixa (`seed = 42` no `CONFIG`) → números reproduzíveis e comparáveis entre treinos.

**Critério de aceite:** após um treino, é possível abrir o `CSV` e ver a curva de melhora; o HUD acompanha em tempo real. ✅ validado: `--eval --episodes=2` roda, salva `pesos.json` e fecha; CSV com todas as colunas; HUD atualiza sem travar.

---

## Fase 6 — Robustez e calibração ✅ concluída

**Para quê:** botar o verme "no limite" e deixá-lo estável.

- [x] **Tuning de hiperparâmetros** (tabela de experimentos): `α, γ, λ_reg, temperatura, decaimento de exploração, H` — via **CLI** (`--set chave=valor`) + tabela de ciência no **`EXPERIMENTOS.md`**.
- [x] **Anti-saturação**: `tanh` com clamp em ±500 na pré-ativação (evita overflow); entradas já normalizadas (direções `[-1,1]`, distâncias `[0,1]`); bônus de entropia + piso de temperatura impedem colapso.
- [x] **Exploração estruturada**: amostragem da softmax + `temperature_decay` por episódio (`T ← max(min_T, T·decay)`); no `--eval` a temperatura não decai.
- [x] **Casos-limite**: sem fontes no mapa, fonte no limite do alcance, worm parado (idle) — validados em `debug_sensores.py` (C1/C2/C3); **fuga de bordas** no professor (`get_target_direction`) para não encalhar na parede.
- [x] **Documentar experimentos** no `EXPERIMENTOS.md` (o que mudou, o que aconteceu, conclusão) — cultura de "tabela de ciência".

**Critério de aceite:** o jogo roda estável nos casos-limite; os hiperparâmetros variam sem editar código; os experimentos são registrados e reproduzíveis (semente fixa). ✅ **Validado**: `debug_sensores.py` (3 casos-limite OK), `python main.py --eval --episodes=2 --set=learning_rate=0.005 --set=gamma=0.95` (override aplicado, avaliação completa).

---

## Mapa conceitual (glossário para os comentários de código)

| Termo | No projeto | Onde aparece |
|-------|-----------|--------------|
| Estado `s` | 8 features (dir/dist/perigo/pulso) | `get_sensor_inputs()` |
| Ação `a` | uma das 5 direções | `sample_action(π)` |
| Política `π(a|s)` | softmax do MLP | ponta do `forward` |
| Retorno `G_t` | soma descontada `Σγ^k r` | `compute_returns()` |
| Baseline `b` | média dos retornos | `mean(G)` |
| Gradiente `∇logπ` | backprop do cross-entropy | `backward()` + Fase 3 |
| Professor | campo de potencial (direção ideal) | `get_target_direction()` |
| Currículo | `λ_imitação` decaído + `teacher_influence` | Fase 4 |

---

## Ordem de execução sugerida (caminho crítico)

```
Fase 0 (limpeza)
   ↓
Fase 1 (sensores 8-dim + recompensa por progresso)
   ↓
Fase 2 (MLP+backprop+softmax) ✅
   ↓
Fase 3 (REINFORCE + baseline) ✅  ← já dá um verme "quase autônomo"
   ↓
Fase 4 (currículo → autonomia plena) ✅  ← 🏆 verme autônomo (estágio C)
   ↓
Fase 5 (métricas/HUD/eval) ✅  e  Fase 6 (robustez) ✅  ← amadurecimento completo
```

> Nota: Fases 5 e 6 podem ser adiantadas (parcialmente) para acompanhar o desenvolvimento desde o começo — logging básico e configuração já no Fase 0 ajudam muito.

---

## Experimentos sugeridos (ao longo do caminho)

- E1 — Comparar recompensa absoluta vs. por progresso (mesma rede, mesma seed).
- E2 — Comparar ações discretas (5) vs. contínuas (desafio opcional).
- E3 — Com e sem baseline: ver a variância e a velocidade de convergência.
- E4 — Com e sem entropia: observar o colapso prematuro.
- E5 — γ baixo (0.9) vs. γ alto (0.99): horizonte de planejamento.
- E6 — 1 fonte vs. 3 fontes de chuva: generalização.

Cada experimento anotado no `EXPERIMENTOS.md` com: hipótese → resultado → conclusão.