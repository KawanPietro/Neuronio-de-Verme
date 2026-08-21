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

Douglas Adams estaria orgulhoso: o verme tem sensores de **fotorecepção** (luz) e de **tato/vibração** (chuva), um **cérebro** de 8 entradas → 16 neurônios ocultos → 5 ações discretas (softmax), e aprende por **backpropagation + política estocástica**.

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
├── perception.py        # Contrato de comportamento: sensores 8-dim + recompensa por progresso
├── mlp.py               # MLP com backprop + política estocástica (softmax, 5 ações)
├── test_mlp.py          # Teste dos gradientes (backward vs. diferenças finitas)
├── Rede_Neural.py       # Cérebro LEGADO (Fase 1, Hebbian-like) — referência didática
├── worm.py              # O corpo do verme (cabeça + segmentos + animação)
├── environment.py       # Fontes de luz/chuva e partículas de chuva
├── debug_sensores.py    # Debug headless do contrato (sensores 8-dim + recompensa por passo)
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
        ENTRADAS              OCULTA              AÇÕES (softmax)
   ┌───────────────┐    ┌──────────────┐    ┌──────────────────┐
   │ luz_dir (x,z) │    │              │    │  esquerda        │  girar −θ
   │ luz_dist      │    │  h[0]...h[15]│───▶│  frente_esquerda │  girar −θ/2
   │ luz_perigo    │───▶│              │    │  frente          │  seguir em frente
   │ chuva_dir(x,z)│    └──────────────┘    │  frente_direita  │  girar +θ/2
   │ chuva_dist    │                        │  direita         │  girar +θ
   │ pulso_chuva   │                        └──────────────────┘
   └───────────────┘
```

**Arquitetura:** `8 entradas → 16 neurônios ocultos → 5 ações discretas`

| Classe / método | Função |
|-----------------|--------|
| `MLP(n_inputs, n_hidden, n_outputs)` | MLP genérico com **backpropagation camada a camada** (regra da cadeia visível). Ativações ocultas à escolha: `tanh` ou `relu`. |
| `MLP.forward(inputs)` | Propagação direta; guarda as ativações intermediárias para o `backward`. |
| `MLP.backward_from_output_grad(grad)` | Backprop do gradiente da saída (`dL/dz`) até todos os parâmetros. |
| `PolicyNetwork` | MLP + **softmax** sobre as ações → `π(a|s)`; `sample_action()` **amostra** (exploração), `imitate()` treina por **cross-entropy supervisionada** (imitação), `update_episode()` treina por REINFORCE com baseline (+ opcional `λ·CE` híbrido). |
| `save_weights()` / `load_weights()` | **Persistência**: salva/carrega todos os pesos em `pesos.json` (teclas `S`/`L`, ou automático no fim da avaliação). |
| `compute_returns(rewards, γ)` | Retornos descontados `G_t = Σ γ^k·r_{t+k}` (do fim para o início). |

**Atualização (`PolicyNetwork.update_episode`, Fase 3):** *REINFORCE com baseline* (Williams 1992), treinado ao fim de cada episódio de `H` passos:

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

**Sensores** (`get_sensor_inputs()` em `perception.py`) — **estado com geometria, 8 dimensões**:

| Feature | Significado |
|---------|-------------|
| `luz_dir` (x, z) | Vetor unitário no plano XZ até a fonte de luz mais próxima |
| `luz_dist` | Distância normalizada `[0,1]` até essa luz |
| `luz_perigo` | `1` se dentro do raio de perigo da luz, senão `0` |
| `chuva_dir` (x, z) | Vetor unitário no plano XZ até a fonte de chuva mais próxima |
| `chuva_dist` | Distância normalizada `[0,1]` até essa chuva |
| `pulso_chuva` | Vibração (sinal de "tato") |

> Antes o verme só via dois escalares ("estar perto de X"), o que não dizia **em que direção** ir. Com a geometria, a rede pode aprender a navegar.

**Recompensa** (`calculate_reward()` em `perception.py`) — **por progresso**, calculada a cada **frame** e normalizada em `[-1, +1]`:
| Termo | Efeito |
|-------|--------|
| `+ progresso` em direção à chuva | **Δ distância curvada** (melhorou → positivo) |
| `− progresso` em direção à luz | Aproximar da luz é ruim |
| `+ BÔNUS` ao entrar no raio da chuva | Recompensa de evento de chegada |
| `− PENALIDADE` ao entrar no perigo da luz | Recompensa de evento de perigo |
| `− CUSTO` por ficar parado | Anti-farniente (evita girar em círculo) |

**Movimento:**
1. Lê o estado 8-dim → `brain.sample_action(sensors)` amostra uma das **5 ações discretas**.
2. Calcula a direção do "professor" (`get_target_direction()` — atração pela chuva + repulsão da luz, sem órbita).
3. A ação vira o verme por um múltiplo da taxa máxima: `{−θ, −θ/2, 0, +θ/2, +θ}`.
4. Mistura o giro do professor com o giro da política conforme a **autonomia**.
5. Move a cabeça (`SPEED × tempo`) com velocidade constante, limitando a posição ao mapa `[-18, 18]`.
6. Cada passo é guardado no episódio atual; ao completar `H` passos, `brain.update_episode()` treina (retornos descontados + baseline), as fontes são reposicionadas aleatoriamente e um log do episódio vai para o terminal e para `episodios.csv`.

> Para validar o contrato sem abrir a janela, rode `python debug_sensores.py` — ele imprime `[sensores 8-dim] + r` por passo e confirma que `r` é maior quando o verme se aproxima da chuva.

---

### 3. O ciclo de aprendizagem

Um resumo visual do loop que roda a cada frame:

```
 Sensores 8-dim (geometria)
        │
        ▼
 Política π(a|s) ──sample──▶ ação (5 discretas) ──▶ giro + velocidade constante
        │
        ▼
 Professor (campos de potencial) ──▶ giro_professor
        │
        ▼
 Mistura: giro = lerp(professor, política, autonomia)
        │
        ▼
 Move o verme  ──▶  Recompensa por progresso (a cada frame)  ──▶  guarda (s, a, r)
        ▲                                                        │
        └────────────  atualiza total_reward ◀───────────────────┘
                        ↑                              H passos coletados ▼
              aumenta a AUTONOMIA                  update_episode(): G_t, baseline,
                                                   vantagem, θ←θ+α·∇θJ (novo episódio)
```

---

## 🎓 Aprendizado por reforço na prática

Os conceitos-chave implementados:

| Termo | O que significa | Onde está no código |
|-------|------------------|---------------------|
| **Agente** | O verme | `head` e seus segmentos |
| **Ambiente** | A cena 3D com luzes e chuva | `main.py` / `environment.py` |
| **Estado (sensores)** | Estado 8-dim com geometria (direções, distâncias, perigo, pulso) | `get_sensor_inputs()` em `perception.py` |
| **Ação** | Uma de 5 direções discretas (giro `−θ..+θ`) | `sample_action()` em `mlp.py` |
| **Recompensa** | Sinal de progresso que guia o aprendizado | `calculate_reward()` em `perception.py` |
| **Política** | Como o agente escolhe a ação | `PolicyNetwork` em `mlp.py` |
| **Professor** | Guia determinístico que demonstra o comportamento ideal | `get_target_direction()` |

---

## 📈 Aprendizado por currículo (Curriculum Learning)

A ideia central e mais interessante do projeto:

> Em vez de a rede neural tentar acertar sozinha desde o início (o que demoraria muito), um **professor** mostra a direção correta a cada frame. O verme observa e reforça. Com o tempo, a **autonomia** aumenta e ele passa a decidir sozinho.

Na Fase 4 a autonomia é ditada por **estágios do currículo** (não mais pelo tempo nem pela recompensa acumulada):

| Estágio | Episódios | Movimento | Treino |
|---------|-----------|-----------|--------|
| **A — Imitação** | 1º ao `stage_a_episodes` | 100% professor | **Cross-entropy supervisionada** (`brain.imitate`) — o verme "cola" no professor |
| **B — Híbrido** | seguintes `stage_b_episodes` | autonomia sobe `0→1` | **REINFORCE + λ·CE** (`update_episode(imitation_weight=λ)`), λ decai a cada episódio |
| **C — Autônomo** | depois | **100% rede** (professor desligado) | REINFORCE puro (`λ = 0`) |

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

O CSV (`episodios.csv`) registra por episódio: `estagio, recompensa_*, retorno, entropia, learning_rate, lambda_imitacao, chegada_chuva, perigo_luz, acao_principal, semente` — com a semente fixa no `CONFIG`, qualquer treino é **reproduzível**.

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

## 💡 Ideias para melhorias futuras

Pontos de partida para evoluir o projeto, do mais simples ao mais ambicioso:

**Simplicidade / manutenção**
- [ ] Remover o `update()` morto e unificar o loop em uma única versão.
- [ ] Refatorar `Verme.py` em módulos (`worm.py`, `environment.py`, `editor.py`, `camera.py`).
- [ ] Adicionar `requirements.txt` e `.gitignore` (excluir `__pycache__`).
- [ ] Consertar as teclas do `DiaNoite.py` e usar `camera.raycast`.
- [ ] Centralizar as constantes mágicas (30, 20, 5, etc.) em um dicionário de configuração.

**Aprendizado**
- [x] Implementar **REINFORCE / policy gradient** com episódios, baseline e desconto (Fases 3–4) — evoluir para A2C/PPO fica como desafio.
- [ ] Adicionar **memória/recorrência** (RNN) para o verme lembrar estados anteriores.
- [x] Salvar/carregar pesos (`json` via `save_weights`/`load_weights`) e medir o progresso entre execuções.
- [ ] Adicionar mais sensores (temperatura, "cheiro" de alimento, predadores).

**Ambiente / visual**
- [x] Render os dados de aprendizado na tela (HUD com episódio, estágio, recompensa média, entropia).
- [x] Visualizar a política aprendida (grade de setas coloridas sobre o mapa, tecla `P`).
- [ ] Integrar o ciclo de dia/noite do `DiaNoite.py` na simulação principal.
- [ ] Adicionar múltiplos vermes competindo/cooperando.
- [x] Evitar colapso numa ação só — bônus de entropia (Fase 2) + anti-farniente (Fase 1).

**Experimentação**
- [x] Rodar múltiplas execuções ("sementes") — semente fixa no `CONFIG` (`seed = 42`) torna o treino reproduzível.
- [x] Criar um csv/log com a curva de recompensa ao longo do tempo (`episodios.csv`).

---

## 📄 Licença / Notas

Projeto educacional para estudo de redes neurais e aprendizado por reforço. Sinta-se livre para experimentar e modificar — a melhor forma de aprender RL é mexendo no cérebro de um verme. 🪱