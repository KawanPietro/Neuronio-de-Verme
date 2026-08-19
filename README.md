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

Douglas Adams estaria orgulhoso: o verme tem um sensor de **fotorecepção** (luz) e um de **tato/vibração** (chuva), um **cérebro** de 2 entradas → 4 neurônios ocultos → 3 saídas, e aprende a mover-se.

---

## 🧰 Tecnologias

- **Python 3.8+** (testado no 3.12)
- **[Ursina](https://www.ursinaengine.org/)** — engine 3D construída sobre Panda3D
- **Apenas a biblioteca padrão** para a rede neural (`math`, `random`)

---

## 📁 Estrutura do projeto

```
Neuronio-de-Verme/
├── Rede_Neural.py        # Implementa a rede neural (2-4-3) e a regra de aprendizado
├── Verme.py              # Simulação principal: ambiente 3D, verme, sensores, recompensa e loop de aprendizado
├── DiaNoite.py           # Demonstração separada: ciclo de dia/noite (não integrado à simulação)
├── Instalar Ursina.py    # Script de teste rápido para verificar se o Ursina funciona
└── README.md             # Este documento
```

---

## 🚀 Como executar

> ⚠️ Recomenda-se usar um ambiente virtual (venv) para não poluir o Python do sistema.

**1. Instale a dependência (Ursina):**

```bash
pip install ursina
```

Você pode conferir se ficou tudo certo rodando `Instalar Ursina.py`:

```bash
python "Instalar Ursina.py"
```

Se aparecer um cubo laranja numa janela, o Ursina está ok.

**2. Rode a simulação principal:**

```bash
python Verme.py
```

**3. Rode a demonstração do ciclo dia/noite (opcional):**

```bash
python DiaNoite.py
```

**4. Teste a rede neural isoladamente (linha de comando):**

```bash
python Rede_Neural.py
```

Este arquivo, além de definir a classe `WormBrain`, tem um bloco de teste no final que cria um cérebro e aplica 10 reforços aleatórios para exibir os pesos.

---

## 🎮 Controles

### Verme.py

| Tecla | Ação |
|-------|------|
| **Botão direito + mouse** | Orbitar a câmera |
| **W / A / S / D** | Mover o foco da câmera (pan vertical/horizontal) |
| **Scroll** | Zoom (aproximar/afastar) |
| **R** | Reiniciar o verme e o cérebro (zera pesos e recompensa) |
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

> 💡 Observação: o texto impresso no terminal do `DiaNoite.py` menciona as teclas `+`/`-`, mas o código de fato usa `O`/`P`.

---

## 🧠 Como o projeto funciona

### 1. O cérebro: `Rede_Neural.py`

A classe `WormBrain` implementa uma rede *feedforward* simples, do zero:

```
        ENTRADAS           OCULTA          SAÍDAS
   ┌────────────┐    ┌──────────────┐    ┌────────────┐
   │  light      │───▶│  h[0]        │───▶│  out[0]    │───▶ eixo X do movimento
   │  touch      │    │  h[1]        │    │  out[1]    │───▶ eixo Y (ignorado na prática)
   └────────────┘    │  h[2]        │    │  out[2]    │───▶ eixo Z do movimento
                    └──────────────┘    └────────────┘
```

**Arquitetura:** `2 entradas → 4 neurônios ocultos → 3 saídas`

| Método | Função |
|--------|--------|
| `__init__()` | Inicializa pesos com **Xavier Initialization** (`√(6/(n_in + n_out))`) e vieses em `[-0.5, 0.5]`. |
| `forward(light, touch)` | **Propagação direta**: calcula as ativações da camada oculta e de saída usando a função de ativação **tanh** (resultado entre -1 e 1). Guarda o último estado em `last_inputs`, `last_hidden`, `last_output`. |
| `reinforce(reward, ...)` | **Regra de aprendizado**: ajusta pesos e vieses proporcionalmente à recompensa recebida, com regularização **L2** para evitar que os pesos cresçam sem limite. |
| `get_weights()` / `set_weights()` | Exporta/importa todos os pesos — útil para salvar e carregar um "cérebro treinado". |

**A regra de aprendizado de `reinforce`:** para cada peso, a atualização é aproximadamente

```
peso  ←  peso  +  (learning_rate × recompensa × entrada × saída)  −  (regularização × peso)
```

Essa é uma forma simplificada de *Hebbian/REINFORCE-like*: multiplicamos a recompensa pelo sinal de entrada e pela ativação. É didática e rápida, mas **não é uma implementação exata de gradiente** (ver [Limitações](#-observações-e-limitações)).

---

### 2. O ambiente: `Verme.py`

É o programa principal. Ele monta a cena 3D e conecta tudo:

**Cenário:**
- Chão com textura de grama, céu (`sky_sunset`), luzes direcional/ambiente.
- Uma luz pontual ciano que segue o verme.
- Sistema de **partículas de chuva** que caem e se reposicionam ao redor das fontes de chuva.
- **Fontes editáveis**: esferas amarelas (luz) e cianas (chuva) posicionadas pelo usuário no modo editor.

**O verme animado:**
- Cabeça (esfera) + **12 segmentos** com gradiente de cor (preto → ciano) e animação de ondulação.
- Os segmentos seguem o histórico de posições da cabeça (efeito "cobra").

**Sensores** (`get_sensor_inputs()`), valores entre `0.0` e `1.0`:
- **light**: `1 − (distância até a fonte de luz mais próxima / 30)`. Perto da luz → valor alto.
- **touch**: `1 − (distância até a fonte de chuva mais próxima / 30)`, somado a um "pulso de chuva" ponderado. Perto da chuva → valor alto.

**Recompensa** (`calculate_reward()`), normalizada em `[-1, +1]`, calculada a cada 1 segundo:
| Situação | Efeito na recompensa |
|----------|----------------------|
| Longe da luz | **+** (positivo, quanto mais longe melhor) |
| Dentro do `ARRIVAL_RADIUS` da luz (perigo) | **−0.5** (penalidade) |
| Longe da chuva | **−** (negativo) |
| Dentro do `ARRIVAL_RADIUS` da chuva (alvo) | **+0.5** (bônus) |

**Movimento:**
1. Lê sensores → `brain.forward(light, touch)`.
2. Calcula a direção do "professor" (`get_target_direction()`).
3. Calcula a direção sugerida pela rede neural.
4. Faz a mistura entre as duas conforme a **autonomia**.
5. Suaviza a direção e move a cabeça (`SPEED × tempo`), limitando a posição ao mapa `[-18, 18]`.
6. A cada segundo, aplica a recompensa no cérebro (`brain.reinforce`) e mostra um log no terminal.

---

### 3. O ciclo de aprendizagem

Um resumo visual do loop que roda a cada frame:

```
 Sensores (luz, toque)
        │
        ▼
 Rede Neural ──forward──▶ (nx, ny, nz) ──▶ direção_neuronal
        │
        ▼
 Professor (campos de potencial) ──▶ direção_professor
        │
        ▼
 Мistura: direção = lerp(professor, neuronal, autonomia)
        │
        ▼
 Move o verme  ──▶  Recompensa (a cada 1s)  ──▶  brain.reinforce()
        ▲                                            │
        └────────────  atualiza total_reward ◀────────┘
                        ↑
              aumenta a AUTONOMIA
```

---

## 🎓 Aprendizado por reforço na prática

Os conceitos-chave implementados:

| Termo | O que significa | Onde está no código |
|-------|------------------|---------------------|
| **Agente** | O verme | `head` e seus segmentos |
| **Ambiente** | A cena 3D com luzes e chuva | `Verme.py` |
| **Estado (sensores)** | Intensidade de luz e toque | `get_sensor_inputs()` |
| **Ação** | Direção de movimento (vetor) | saída da rede neural |
| **Recompensa** | Sinal que guia o aprendizado | `calculate_reward()` |
| **Política** | Como o agente escolhe a ação | a própria rede neural (`WormBrain`) |
| **Professor** | Guia determinístico que demonstra o comportamento ideal | `get_target_direction()` |

---

## 📈 Aprendizado por currículo (Curriculum Learning)

A ideia central e mais interessante do projeto:

> Em vez de a rede neural tentar acertar sozinha desde o início (o que demoraria muito), um **professor** mostra a direção correta a cada frame. O verme observa e reforça. Com o tempo, a **autonomia** aumenta e ele passa a decidir sozinho.

A autonomia é calculada com base no **desempenho real** (recompensa acumulada), não no tempo:

```python
AUTONOMY_SCALE = 30.0
autonomy = min(1.0, max(0.0, state['total_reward'] / AUTONOMY_SCALE))
```

- `autonomy = 0` → **100% professor** (rede só observa)
- `autonomy = 1` → **100% rede neural** (rede decidiu tudo sozinha)

Nos estágios intermediários, a direção é uma **interpolação** (`lerp`) entre as duas. Isso é análogo a técnicas usadas em RL moderno (como *DAgger* ou *behavioral cloning progressivo*).

**Direção do professor** (`get_target_direction()`) — três comportamentos:
1. **Atração pela chuva** — vetor em direção às fontes de chuva, ponderado por `max(0, 1 − dist/30)` (fontes próximas puxam mais).
2. **Órbita** — quando bem perto da chuva (`< ARRIVAL_RADIUS`), o professor ensina a orbitar em vez de parar (`tangente × 0.7 + radial × 0.3`).
3. **Repulsão pela luz** — vetor de fuga das luzes, ponderado por `max(0, 1 − dist/20)`.

Isso é inspirado em **campos de potencial** (*potential fields*), técnica clássica de navegação robótica.

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

1. **`update()` duplicado em `Verme.py`.** Existem duas definições da função `update` (linhas ~211 e ~407). Como em Python a última definição vence, apenas a segunda executa. Toda a primeira versão (com `update_camera`, `update_rain_particles`, `update_segments`, `update_rewards`, `spawn_energy_particles`) é **código morto**, inclusive a criação de partículas de energia.

2. **Regra de aprendizado simplificada.** O `reinforce()` usa um heurístico *Hebbian-like* (`lr × recompensa × entrada × saída`), não o gradiente real. Um `REINFORCE`/`policy gradient` verdadeiro, ou mesmo `backpropagation`, provavelmente aprenderia de forma mais estável.

3. **`ny` da saída ignorada.** A rede gera 3 saídas, mas o movimento usa apenas `x` e `z` (`Vec3(nx, 0, nz)`). A saída do meio é essencialmente desperdiçada.

4. **Incompatibilidade de teclas no `DiaNoite.py`.** O print de instruções diz `+`/`-`, mas o código escuta `O`/`P`.

5. **Compatibilidade com versões novas do Ursina.** A função global `raycast()` (usada no modo editor) foi removida/movida para `camera.raycast()` em versões recentes do Ursina — pode precisar de ajuste.

6. **`brain` criado duas vezes.** `Rede_Neural.py` cria um `brain` global no módulo (e treina com recompensas aleatórias ao rodar), enquanto `Verme.py` cria sua própria instância. O bloco de demonstração do módulo não interfere no jogo, mas pode confundir.

7. **Sem persistência.** `get_weights`/`set_weights` existem, mas nada salva/carrega o "cérebro treinado" em disco.

8. **Sem arquivo de requisitos.** Não há `requirements.txt`. Recomenda-se adicionar com `pip freeze > requirements.txt`.

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
- [ ] Implementar **REINFORCE / policy gradient** real ou **backpropagation**.
- [ ] Usar as 3 saídas da rede (permitir movimento no eixo Y, por exemplo).
- [ ] Adicionar **memória/recorrência** (RNN) para o verme lembrar estados anteriores.
- [ ] Salvar/carregar pesos (`pickle`/`json`) e medir o progresso entre execuções.
- [ ] Adicionar mais sensores (temperatura, "cheiro" de alimento, predadores).

**Ambiente / visual**
- [ ] Render os dados de aprendizado na tela (HUD com recompensa, autonomia e gráfico simples).
- [ ] Integrar o ciclo de dia/noite do `DiaNoite.py` na simulação principal.
- [ ] Adicionar múltiplos vermes competindo/cooperando.
- [ ] Evitar que o score cresça só por andar em círculos (termo de entropia/custo de energia).

**Experimentação**
- [ ] Rodar múltiplas execuções ("sementes") para comparar a convergência do aprendizado.
- [ ] Criar um csv/log com a curva de recompensa ao longo do tempo.

---

## 📄 Licença / Notas

Projeto educacional para estudo de redes neurais e aprendizado por reforço. Sinta-se livre para experimentar e modificar — a melhor forma de aprender RL é mexendo no cérebro de um verme. 🪱