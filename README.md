# Neuronio-de-Verme

Uma simulação 3D educativa em Python que usa **redes neurais** e **aprendizado por reforço** para ensinar um verme virtual a se comportar: ele deve **fugir da luz** e **procurar a chuva** — um cenário inspirado em como vermes reais se comportam ao sair do solo.

> O projeto funciona como um "laboratório vivo": você assiste ao verme aprendendo em tempo real, pode posicionar estímulos (luz/chuva) pela cena e acompanhar a evolução da sua autonomia.

---

## Sumário

- [Conceito](#-conceito)
- [Tecnologias](#-tecnologias)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Como executar](#-como-executar)
- [Controles](#-controles)
- [Como o projeto funciona](#-como-o-projeto-funciona)
  - [1. O cérebro: `Rede_Neural.py`](#1-o-cérebro-rede_neuralpy)
  - [2. O ambiente: `Verme.py`](#2-o-ambiente-verme.py)
  - [3. O ciclo de aprendizagem](#3-o-ciclo-de-aprendizagem)
- [Aprendizado por reforço na prática](#-aprendizado-por-refoço-na-prática)
- [Aprendizado por currículo (Curriculum Learning)](#-aprendizado-por-currículo-curriculum-learning)
- [A demonstração `DiaNoite.py`](#-a-demonstração-dianoitepy)
- [Observações e limitações](#-observações-e-limitações)
- [Estado atual e guia de estudo](#️-estado-atual-e-guia-de-estudo-para-o-grupo)
- [Ideias para melhorias futuras](#-ideias-para-melhorias-futuras)

---

## 🪱 Conceito

O projeto une duas áreas de estudo de forma visual e divertida:

| Área | Aplicação no projeto |
|------|----------------------|
| **Redes neurais** | Um MLP (Multi-Layer Perceptron) de camada única oculta, implementado do zero (sem bibliotecas de ML). |
| **Aprendizado por reforço (RL)** | O verme recebe recompensas/penalidades dependendo de onde está, e ajusta os pesos do "cérebro" a partir delas. |
| **Aprendizado por currículo** | O verme começa sendo guiado por um "professor" determinístico e, conforme acumula recompensa, vai ganhando autonomia para decidir sozinho. |
| **Campos de potencial** | A direção "professor" é calculada com uma técnica clássica de navegação de robótica. |
| **Game engine** | O ambiente 3D é renderizado e interativo usando `Ursina`. |

**O objetivo do verme:**
- 🟦 **Procurar chuva** → estar perto de fontes de chuva gera **recompensa positiva** (+)
- ☀️ **Evitar luz** → estar perto de fontes de luz gera **recompensa negativa** (−)

Douglas Adams estaria orgulhoso: o verme tem sensores de **fotorecepção** (luz), **tato/vibração** (chuva), **propriocepção** (direção/velocidade) e **desvio de obstáculos**, um **cérebro** de 17 entradas → 32 → 16 neurônios ocultos → 5 ações discretas (softmax, Fase 12), e aprende por **backpropagation + PPO/A2C**.

---

## 🧰 Tecnologias

- **Python 3.8+** (testado no 3.12)
- **[Ursina](https://www.ursinaengine.org/)** — engine 3D construída sobre Panda3D
- **Apenas a biblioteca padrão** para a rede neural (`math`, `random`)

---

## 📁 Estrutura do projeto

```
Neuronio-de-Verme/
├── main.py              # Ponto de entrada: orquestra câmera, editor, update e input
├── config.py            # Todas as constantes em um dicionário CONFIG
├── perception.py        # Contrato de comportamento: sensores 17-dim (Fase 12) + recompensa por progresso
├── mlp.py               # MLP + Policy/Critic + PPO/A2C + Replay Buffer (softmax, 5 ações)
├── test_mlp.py          # 15 testes: gradientes, REINFORCE, A2C, PPO, buffer, save/load
├── Rede_Neural.py       # Cérebro LEGADO (Fase 1, Hebbian-like) — referência didática
├── worm.py              # O corpo do verme (cabeça + segmentos + animação + colisão)
├── environment.py       # Fontes de luz/chuva + obstáculos/muros + partículas de chuva
├── debug_sensores.py    # Debug headless do contrato (sensores 17-dim + recompensa por passo)
├── Verme.py             # Shim de compatibilidade (executa main.py)
├── DiaNoite.py          # Demonstração separada: ciclo de dia/noite
├── Instalar Ursina.py   # Script de teste rápido para verificar se o Ursina funciona
├── episodios.csv        # Curva de aprendizado por episódio (gerado pelo jogo)
├── pesos.json           # Cérebro salvo (teclas S/L, ou fim de --eval)
├── EXPERIMENTOS.md      # Tabela de ciência: o que mudou, o que aconteceu, conclusão
├── requirements.txt     # Dependência: ursina
└── README.md            # Este documento
```

---

## 🚀 Como executar

> ⚠️ Recomenda-se usar um ambiente virtual (venv) para não poluir o Python do sistema.

**1. Instale a dependência (Ursina):**

```bash
pip install -r requirements.txt
```

Você pode conferir se ficou tudo certo rodando `Instalar Ursina.py`:

```bash
python "Instalar Ursina.py"
```

Se aparecer um cubo laranja numa janela, o Ursina está ok.

**2. Rode a simulação principal:**

```bash
python main.py
```

> `Verme.py` continua existindo como atalho de compatibilidade — ele apenas delega para `main.py`.

**3. Rode a demonstração do ciclo dia/noite (opcional):**

```bash
python DiaNoite.py
```

**4. Valide os gradientes da rede neural (linha de comando):**

```bash
python test_mlp.py
```

Compara o `backward` com diferenças finitas (regressão + política) — é o critério de aceite da Fase 2.

**5. Teste o cérebro legado isoladamente (opcional):**

```bash
python Rede_Neural.py
```

Este arquivo (referência didática da Fase 1) define a antiga classe `WormBrain` e tem um bloco de teste que cria um cérebro e aplica 10 reforços aleatórios para exibir os pesos.

---

## 🎮 Controles

### main.py (simulação principal)

| Tecla | Ação |
|-------|------|
| **Botão direito + mouse** | Orbitar a câmera |
| **W / A / S / D** | Mover o foco da câmera (pan vertical/horizontal) |
| **Scroll** | Zoom (aproximar/afastar) |
| **R** | Reiniciar o verme e o cérebro (zera pesos e recompensa) |
| **A** | Ligar/desligar o **professor** (autonomia forçada = 100%) |
| **P** | Mostrar/ocultar a **grade de setas** da política aprendida |
| **S / L** | **Salvar / carregar** os pesos do cérebro (`pesos.json`) |
| **1** | Alternar modo *colocar fonte de LUZ* |
| **2** | Alternar modo *colocar fonte de CHUVA* |
| **3** | Alternar modo *deletar fonte* |
| **Clique esquerdo** | Executar a ação do modo atual (colocar/deletar) |
| **ESC** | Cancelar o modo ativo; apertar de novo fecha o programa |

### DiaNoite.py

| Tecla | Ação |
|-------|------|
| **Espaço** | Alternar modo automático / manual |
| **→ ou D** | Avançar o sol (modo manual) |
| **← ou A** | Recuar o sol (modo manual) |
| **O** | Aumentar velocidade (modo automático) |
| **P** | Diminuir velocidade (modo automático) |

> 💡 Observação: as instruções no terminal do `DiaNoite.py` correspondem às teclas reais (`O`/`P`).

---

## 🧠 Como o projeto funciona

### 1. O cérebro: `mlp.py`

O cérebro usado pelo jogo é uma **política estocástica**: um MLP com **backpropagation** implementado do zero, com um cabeçote **softmax** sobre ações discretas. (A `Rede_Neural.py` antiga, com regra Hebbian-like, ficou como referência didática.)

```
        ENTRADAS (17)              OCULTAS              AÇÕES (softmax)
   ┌───────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │ luz_dir (x,z)     │    │                  │    │  esquerda        │  girar −θ
   │ luz_dist/perigo   │    │  h1[0]...h1[31]  │───▶│  frente_esquerda │  girar −θ/2
   │ chuva_dir (x,z)   │───▶│        ↓         │    │  frente          │  seguir em frente
   │ chuva_dist/pulso  │    │  h2[0]...h2[15]  │───▶│  frente_direita  │  girar +θ/2
   │ obs_dir/dist (x,z)│    │                  │    │  direita         │  girar +θ
   │ vel, borda, ângulos│   └──────────────────┘    └──────────────────┘
   └───────────────────┘   (Fase 11: 2 camadas + Fase 12: 17-dim)
```

**Arquitetura (Fase 12):** `17 entradas → 32 → 16 ocultos → 5 ações discretas` (actor ~1189 + critic ~1121 params)

| Classe / método | Função |
|-----------------|--------|
| `MLP(n_inputs, n_hidden, n_outputs)` | MLP genérico com **backpropagation camada a camada** (regra da cadeia visível). Ativações ocultas à escolha: `tanh` ou `relu`. |
| `MLP.forward(inputs)` | Propagação direta; guarda as ativações intermediárias para o `backward`. |
| `MLP.backward_from_output_grad(grad)` | Backprop do gradiente da saída (`dL/dz`) até todos os parâmetros. |
| `PolicyNetwork` | MLP + **softmax** sobre as ações → `π(a|s)`; `sample_action()` **amostra** (exploração), `imitate()` treina por **cross-entropy** (Fase 4A), `update_episode()` treina por REINFORCE/A2C/PPO (+ `λ·CE` híbrido na Fase 4B). |
| `CriticNetwork` | MLP `17→32→16→1` que estima `V(s)`; usado no GAE + loss MSE (Fases 8–9). |
| `ReplayBuffer` | Guarda até 50 episódios e treina PPO em mini-batches × epochs (Fase 13). |
| `save_brain()` / `load_brain()` | **Persistência**: salva/carrega actor+critic em `pesos.json` (teclas `S`/`L`); descarta pesos com `n_inputs` incompatível (ex: 11→17 exige treino novo). |
| `compute_returns` / `compute_gae` | Retornos `G_t` (Fase 3) e advantages GAE `A_t` (Fase 8). |

**Atualização (Fases 3/8/9):** REINFORCE → A2C (GAE + critic) → PPO (clipped surrogate), treinado ao fim de cada episódio de `H` passos:

```
G_t = Σ γ^k · r_{t+k}          (retornos descontados)
b   = média(G)                  (baseline: reduz a variância)
∇θ J ≈ Σ (G_t − b) · ∇logπ(a_t|s_t) + β·Σ∇H(π_t)
θ ← θ + α·∇θJ − α·λ·θ          (α decai a cada episódio)
```

- `(G_t − b)` é a **vantagem**: "compare com a média do episódio, não com um número absoluto".
- `β·∇H(π)` = **bônus de entropia** (evita colapso prematuro numa única ação).
- Ação escolhida por **amostragem**, não argmax — é assim que se explora.

> Cada episódio reposiciona luz/chuva em lugares **aleatórios** (`env.randomize_sources()`) — o verme generaliza, não decora uma cena. A curva por episódio vai para `episodios.csv` (recompensa média, entropia, `λ` de imitação, taxa de **chegada na chuva** e de **perigo da luz**).

> O currículo da Fase 4 roda em **estágios**: A (imitação por CE), B (REINFORCE + `λ·CE` com λ decaindo) e C (autonomia plena, `λ = 0`).

- `∇logπ(a|s)` = derivada da softmax: `(δ_{a,k} − π_k) / T`.
- Verifique a correção dos gradientes com `python test_mlp.py` (compara o `backward` com diferenças finitas numéricas).

---

### 2. O ambiente: `main.py` + `perception.py`

`main.py` é o programa principal (monta a cena 3D e conecta tudo). O *contrato de comportamento* — sensores e recompensa — vive isolado em `perception.py`:

**Cenário:**
- Chão com textura de grama, céu (`sky_sunset`), luzes direcional/ambiente.
- Uma luz pontual ciano que segue o verme.
- Sistema de **partículas de chuva** que caem e se reposicionam ao redor das fontes de chuva.
- **Fontes editáveis**: esferas amarelas (luz) e cianas (chuva) posicionadas pelo usuário no modo editor.

**O verme animado:**
- Cabeça (esfera) + **12 segmentos** com gradiente de cor (preto → ciano) e animação de ondulação.
- Os segmentos seguem o histórico de posições da cabeça (efeito "cobra").

**Sensores** (`get_sensor_inputs()` em `perception.py`) — **estado 17-dim (Fase 12 = 11 + 6)**:

| # | Feature | Significado |
|---|---------|-------------|
| 0–3 | `luz_dir (x,z)`, `luz_dist`, `luz_perigo` | Direção, distância `[0,1]` e flag de perigo da luz |
| 4–7 | `chuva_dir (x,z)`, `chuva_dist`, `pulso_chuva` | Direção, distância `[0,1]` e vibração da chuva |
| 8–10 | `obs_dir (x,z)`, `obs_dist` | Direção e distância do obstáculo (Fase 15) |
| 11–12 | `vel_x/z` | Direção atual do verme — propriocepção `[-1,1]` (Fase 12) |
| 13–14 | `borda_x/z` | Distância à parede `[0,1]` — `1`=centro, `0`=borda (Fase 12) |
| 15–16 | `angulo_luz/chuva` | Quanto precisa virar `/π` em `[-1,1]` — `0`=alinhado (Fase 12) |

> Evolução: 8-dim (Fase 1: só luz/chuva) → 11-dim (Fase 15: +obstáculos) → 17-dim (Fase 12: +vel/borda/ângulos). Antes o verme sabia "onde está a chuva", agora sabe também "para onde estou indo e quanto falta virar".

**Recompensa** (`calculate_reward()` em `perception.py`) — **por progresso**, calculada a cada **frame** e normalizada em `[-1, +1]`:
| Termo | Efeito |
|-------|--------|
| `+ progresso` chuva / `− progresso` luz | **Δ distância** × `progress_scale` (melhorou → positivo) |
| `+ proximidade` chuva / `− proximidade` luz | `1/dist` contínuo — puxa mesmo de longe (Fase 7) |
| `+ BÔNUS` chegada / `− PENALIDADE` perigo | Eventos ao cruzar `arrival_radius` + bônus por permanecer |
| `− colisão/proximidade` obstáculo | Penalidade ao bater + por estar perto; `+` ao se afastar (Fase 15) |
| `− repetição` de ação | Quebra colapso "sempre mesma ação" (Fase 7, janela 20) |

**Movimento:**
1. Lê o estado 17-dim → `brain.sample_action(sensors)` amostra uma das **5 ações discretas**.
2. Calcula a direção do "professor" (`get_target_direction()` — atração chuva + repulsão luz + fuga de bordas + contorno de pedras).
3. A ação vira o verme por um múltiplo da taxa máxima: `{−θ, −θ/2, 0, +θ/2, +θ}`.
4. Mistura o giro do professor com o giro da política conforme a **autonomia** (estágios A/B/C).
5. Move a cabeça (`SPEED × tempo`) com velocidade constante, com **colisão/deslize** em obstáculos, limitado ao mapa `[-32, 32]`.
6. Cada passo é guardado no episódio atual; ao completar `H` passos, treina (PPO/A2C via buffer ou direto), reposiciona fontes+obstáculos e registra no terminal e em `episodios.csv`.

> Para validar o contrato sem abrir a janela, rode `python debug_sensores.py` — ele imprime `[sensores 17-dim] + r` por passo e confirma que `r` é maior quando o verme se aproxima da chuva (testes C1–C4).

---

### 3. O ciclo de aprendizagem

Um resumo visual do loop que roda a cada frame:

```
 Sensores 17-dim (geometria + propriocepção + obstáculos)
        │
        ▼
 Política π(a|s) ──sample──▶ ação (5 discretas) ──▶ giro + velocidade constante
        │
        ▼
 Professor (potencial: chuva/luz/borda/pedras) ──▶ giro_professor
        │
        ▼
 Mistura: giro = lerp(professor, política, autonomia A/B/C)
        │
        ▼
 Move o verme (colisão/deslize) ──▶ Recompensa por progresso ──▶ guarda (s, a, r)
        ▲                                                              │
        └────────────  atualiza total_reward ◀─────────────────────────┘
                        ↑                              H passos coletados ▼
              estágio A/B/C                   PPO/A2C: GAE, clipping, critic
                                              θ←θ+α·∇θJ (cena nova por episódio)
```

---

## 🎓 Aprendizado por reforço na prática

Os conceitos-chave implementados:

| Termo | O que significa | Onde está no código |
|-------|------------------|---------------------|
| **Agente** | O verme | `head` e seus segmentos |
| **Ambiente** | A cena 3D com luzes e chuva | `main.py` / `environment.py` |
| **Estado (sensores)** | Estado 17-dim: luz/chuva + obstáculos + vel/borda/ângulos | `get_sensor_inputs()` em `perception.py` |
| **Ação** | Uma de 5 direções discretas (giro `−θ..+θ`) | `sample_action()` em `mlp.py` |
| **Recompensa** | Progresso + proximidade + eventos + anti-colapso, em `[-1,1]` | `calculate_reward()` em `perception.py` |
| **Política / Valor** | `π(a|s)` (actor) + `V(s)` (critic) + GAE + PPO clipping | `PolicyNetwork`/`CriticNetwork` em `mlp.py` |
| **Professor** | Potencial: atração chuva + repulsão luz + fuga borda + contorno pedras | `get_target_direction()` em `main.py` |

---

## 📈 Aprendizado por currículo (Curriculum Learning)

A ideia central e mais interessante do projeto:

> Em vez de a rede neural tentar acertar sozinha desde o início (o que demoraria muito), um **professor** mostra a direção correta a cada frame. O verme observa e reforça. Com o tempo, a **autonomia** aumenta e ele passa a decidir sozinho.

Na Fase 4 a autonomia é ditada por **estágios do currículo** (não mais pelo tempo nem pela recompensa acumulada):

| Estágio | Episódios | Movimento | Treino |
|---------|-----------|-----------|--------|
| **A — Imitação** | 1º ao `stage_a_episodes` | 100% professor | **Cross-entropy supervisionada** (`brain.imitate`) — o verme "cola" no professor |
| **B — Híbrido** | seguintes `stage_b_episodes` | autonomia sobe `0→1` | **PPO/A2C + λ·CE** (`update_episode(imitation_weight=λ)`), λ decai a cada episódio |
| **C — Autônomo** | depois | **100% rede** (professor desligado) | PPO/A2C puro (`λ = 0`) |

- `λ` começa em `lambda_start` e decai (`lambda_decay`) por episódio até 0 — o professor vira tutor e depois some.
- `teacher_action()` converte a direção do campo de potencial na **ação discreta ideal** (alvo da imitação).
- A tecla **A** força o Estágio C (professor desligado) a qualquer momento, para avaliar o verme sozinho.

**Direção do professor** (`get_target_direction()`) — dois comportamentos (desde a Fase 1, a "órbita" foi removida do contrato):
1. **Atração pela chuva** — vetor em direção às fontes de chuva, ponderado por `max(0, 1 − dist/30)` (fontes próximas puxam mais).
2. **Repulsão pela luz** — vetor de fuga das luzes, ponderado por `max(0, 1 − dist/20)`.

Isso é inspirado em **campos de potencial** (*potential fields*), técnica clássica de navegação robótica.

---

## 📊 Métricas, memória e visualização (Fase 5)

**HUD na tela** — painel fixo mostrando em tempo real: episódio, estágio do currículo, recompensa média *rolling* (últimos 10 episódios), entropia da política, learning rate e modo (treino/avaliação).

**Grade de setas (tecla `P`)** — em cada célula do mapa, uma seta mostra **para onde a política quer ir** (ação mais provável, assumindo o verme virado para `+Z`): verde se a direção aponta para a chuva, vermelho se aponta para a luz, cinza se neutra. É o "pensamento" da rede virando imagem.

**Persistência (teclas `S`/`L`)** — `save_weights()`/`load_weights()` exportam e importam todos os pesos para `pesos.json`. No fim de uma avaliação com limite, os pesos são salvos automaticamente.

**Modo avaliação `--eval`** — professor sempre desligado e **sem treino**; mede apenas. Permite comparar treinos de forma justa:

```bash
python main.py --eval --episodes=20   # roda 20 episódios, salva pesos e fecha
```

O CSV (`episodios.csv`) registra por episódio: `estagio, recompensa_*, retorno, entropia, learning_rate, lambda_imitacao, chegada_chuva, perigo_luz, colisao_obs, acao_principal, semente, dificuldade` — com a semente fixa no `CONFIG`, qualquer treino é **reproduzível**. Pesos antigos com `n_inputs` diferente são ignorados com aviso (ex: Fase 12 exige treino novo 11→17).

**Robustez e calibração (Fase 6):**
- **Exploração estruturada** — a temperatura do softmax **decai** a cada episódio de treino (`temperature_decay`), com piso `min_temperature`; a política nunca fica 100% greedy. No `--eval` a temperatura não decai.
- **Anti-encalhe** — perto da borda, o professor foge da parede (`wall_margin` em `get_target_direction`).
- **Tabela de ciência** — hiperparâmetros podem ser variados **sem editar código**:
  ```bash
  python main.py --eval --episodes=50 --set=learning_rate=0.005 --set=gamma=0.95
  ```
  Registre os resultados em `EXPERIMENTOS.md`.

---

## 🌅 A demonstração `DiaNoite.py`

Um mini-projeto **separado** (não integrado ao `Verme.py`) que simula o ciclo do dia com um sol arcando no céu (0° a 360°), dividido em 4 fases:

| Fases (t de 0–1) | Transição |
|------------------|-----------|
| 0.00 → 0.25 | Amanhecer (laranja → ciano) |
| 0.25 → 0.50 | Tarde (ciano → laranja) |
| 0.50 → 0.75 | Entardecer → noite (laranja → preto) |
| 0.75 → 1.00 | Noite → amanhecer (preto → laranja) |

Tem modo **automático** (o ângulo avança com `time.dt`) e modo **manual** (teclas de seta). É um bom exemplo didático de separação de responsabilidades: a função `apply_cycle(angle)` concentra toda a lógica de cores/luzes, e os modos automático/manual apenas a chamam com ângulos diferentes.

---

## 🔍 Observações e limitações

Analisando o código, alguns pontos merecem atenção para quem for continuar o projeto:

1. ~~**`update()` duplicado em `Verme.py`.**~~ **Resolvido** na Fase 0: o loop foi unificado em `main.py` e o código morto removido.
2. ~~**Regra de aprendizado simplificada.**~~ **Resolvido** na Fase 2 (backprop + política estocástica), **aprimorado na Fase 3** (REINFORCE com baseline episódico) e **completado na Fase 4** (currículo em 3 estágios: imitação → híbrido → autônomo). O heurístico antigo ficou em `Rede_Neural.py`, como referência.
3. ~~**`ny` da saída ignorada.**~~ **Resolvido** na Fase 2: a saída agora é uma distribuição softmax sobre **5 ações discretas** (não há eixo desperdiçado).
4. ~~**Incompatibilidade de teclas no `DiaNoite.py`.**~~ **Resolvido**: os textos agora dizem `O`/`P` e os caracteres especiais foram trocados por ASCII (seguro em qualquer terminal).
5. ~~**Compatibilidade com versões novas do Ursina.**~~ **Resolvido**: o editor usa `camera.raycast(...)`.
6. ~~**`brain` criado duas vezes.**~~ **Resolvido**: a demo do módulo foi movida para `if __name__ == '__main__'` e não roda na importação.
7. ~~**Sem persistência.**~~ **Resolvido** na Fase 5: `save_weights`/`load_weights` (JSON, teclas `S`/`L`) e salvamento automático no fim do `--eval`.
8. ~~**Sem arquivo de requisitos.**~~ **Resolvido**: há `requirements.txt` e `.gitignore`.

---

## 🗺️ Estado atual e guia de estudo (para o grupo)

**Onde estamos (Fase 12 concluída — código):** estado 17-dim + rede `17→32→16→5` + PPO/A2C + obstáculos/muros.

| Fase | O que foi | Status | Como ver no código |
|------|-----------|--------|-------------------|
| 0–4 | Base, sensores, MLP, REINFORCE, currículo A/B/C | ✅ | `main.py: current_stage/lambda_imitation`, `perception.py` |
| 5–7 | HUD/grade/persistência, robustez, reward shaping | ✅ | `main.py: HUD/grid`, `test_mlp.py`, `debug_sensores.py` |
| 8–9 | A2C (critic+GAE) e PPO (clipping) — 40% eval | ✅ | `mlp.py: CriticNetwork/compute_gae/ppo_grad` |
| 10–11 | Treino longo + rede profunda `→32→16→` | ⚠️ parcial | `config.py: lr_decay/temp`, `EXPERIMENTOS.md: #8–10` |
| 12 | Estado 17-dim (vel/borda/ângulos) | ✅ código | `perception.py: get_sensor_inputs`, `config.py: n_inputs=17` |
| 13 | Replay Buffer off-policy | ✅ código | `mlp.py: ReplayBuffer/ppo_update_from_buffer` |
| 15 | Obstáculos/muros + 4 níveis + colisão | ✅ | `environment.py: randomize_obstacles`, tecla `O` |
| 14 | Avaliação DoD ≥80% | ⚠️ parcial (10/100 seguro, DoD não atingido) | Ver `EXPERIMENTOS.md #13`; multi-seed 500+ eps pendente |

**Roteiro de estudo (30 min para apresentar):**
1. `python test_mlp.py` — prova que o backprop está certo (5 min).
2. `python debug_sensores.py` — prova que a recompensa ensina "ir para chuva" + testes C1–C4 da Fase 12 (5 min).
3. `python main.py` — mostre HUD, tecla `P` (grade), tecla `A` (professor on/off), tecla `O` (níveis) (10 min).
4. `EXPERIMENTOS.md` + `episodios.csv` — conte a história 20%→40% e por que colapsa sem Fase 10/11/12 (10 min).
5. Detalhe de referência: `PLANO_DE_MELHORIAS.md` tem o diagnóstico completo (treino curto, dados descartados, rede pequena).

**Como registrar a próxima mudança (padrão do grupo):** edite `config.py` ou código → rode `test_mlp.py` + `debug_sensores.py` → treino curto `--episodes=35` → eval `--eval --episodes=50` → adicione 1 linha em `EXPERIMENTOS.md` → atualize esta seção + `PLANO_DE_MELHORIAS.md`.

## 💡 Ideias para melhorias futuras

**Simplicidade / manutenção**
- [x] Remover o `update()` morto e unificar o loop em `main.py`.
- [x] Refatorar em módulos (`worm.py`, `environment.py`, `perception.py`, `mlp.py`).
- [x] Adicionar `requirements.txt` e `.gitignore`.
- [x] Consertar as teclas do `DiaNoite.py` e usar `camera.raycast`.
- [x] Centralizar as constantes mágicas em `config.py`.

**Aprendizado**
- [x] REINFORCE (Fases 3–4) → A2C (Fase 8) → PPO (Fase 9).
- [x] Rede profunda `17→32→16→5` (Fase 11) + estado 17-dim (Fase 12).
- [x] Replay Buffer (Fase 13, código) + obstáculos (Fase 15).
- [ ] Treino longo 500+ eps com 17-dim × multi-seed (próximo passo da Fase 12).
- [ ] Adicionar **memória/recorrência** (RNN) e novos sensores (temperatura, cheiro, predadores).

**Ambiente / visual**
- [x] HUD, grade de setas (`P`), níveis de dificuldade (`O`), anti-colapso (entropia + repetição).
- [ ] Integrar o ciclo de dia/noite do `DiaNoite.py` na simulação principal.
- [ ] Adicionar múltiplos vermes competindo/cooperando.

**Experimentação**
- [x] Multi-seed reproduzível + `episodios.csv` + tabela em `EXPERIMENTOS.md`.
- [ ] Fase 14: validar DoD ≥80% em 100+ eps (ou documentar limite do RL puro sem frameworks).

---

## 📄 Licença / Notas

Projeto educacional para estudo de redes neurais e aprendizado por reforço. Sinta-se livre para experimentar e modificar — a melhor forma de aprender RL é mexendo no cérebro de um verme. 🪱