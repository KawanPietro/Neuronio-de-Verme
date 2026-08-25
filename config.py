# Configuração central do projeto.
#
# Todos os números "mágicos" do jogo vivem aqui. Mudar um valor neste arquivo
# deve ser suficiente para ajustar o comportamento — sem caçar o código.

CONFIG = {
    # ── Verme ────────────────────────────────────────────────────────────────
    'num_segments'     : 12,   # Quantidade de segmentos do corpo
    'segment_size'     : 2.5,  # Tamanho de cada segmento (e da cabeça)
    'segment_gap'      : 2.5,  # Espaçamento entre segmentos
    'speed'            : 6.0,  # Velocidade de movimento
    'map_limit'        : 18,   # Limite do mapa em X e Z

    # ── Sensores / geometria ─────────────────────────────────────────────────
    'sensor_max_dist'  : 30.0,  # Distância de referência para normalizar sensores
    'light_repel_dist' : 20.0,  # Alcance da repulsão pela luz (campo de potencial)
    'arrival_radius'   : 7.0,   # Raio de "chegada" na chuva / "perigo" na luz

    # ── Recompensa por progresso (reward shaping, por passo) ─────────────────
    # Fase 7: sinais mais fortes para que o RL aprenda a VIRAR (não só andar reto).
    'progress_scale'      : 3.0,   # Ganho aplicado ao Δ distância (unidades de movimento)
    'arrival_bonus'       : 5.0,   # Bônus ao entrar no raio da chuva
    'danger_penalty'      : 5.0,   # Penalidade ao entrar na zona de perigo da luz
    'inside_rain_reward'  : 0.5,   # Recompensa por permanecer dentro da chuva
    'inside_light_penalty': 0.5,   # Penalidade por permanecer perto da luz
    'idle_cost'           : 0.0,   # Desativado — o verme não precisa de incentivo p/ andar
    'idle_threshold'      : 0.05,  # Deslocamento/frame abaixo disso conta como "parado"

    # ── Recompensa de proximidade (Fase 7) ───────────────────────────────────
    # Potencial que puxa o verme para perto da chuva e empurra da luz.
    'proximity_rain_bonus'  : 0.3,  # Bônus proporcional a 1/dist para chuva
    'proximity_light_penalty': 0.3, # Penalidade proporcional a 1/dist para luz

    # ── Penalidade por repetição de ação (Fase 7) ────────────────────────────
    # Quebra o colapso da política: se o verme escolhe a mesma ação muitas
    # vezes seguidas, sofre uma penalidade crescente — incentiva alternância.
    'action_repeat_penalty' : 0.1,
    'action_repeat_window'  : 20,

    # ── Cérebro ──────────────────────────────────────────────────────────────
    'n_inputs'         : 8,     # Features do estado (Fase 1: sensores com geometria)
    'n_hidden'         : 16,    # Neurônios da camada oculta
    'n_outputs'        : 3,     # Saídas contínuas (usado pela Rede_Neural legada)
    'n_actions'        : 5,     # Ações discretas da política (Fase 2)
    'learning_rate'    : 0.05,  # Taxa de aprendizado (alpha) — Fase 7: 5x maior
    'regularization'   : 0.001, # Fator L2

    # ── Política estocástica (Fase 2) ────────────────────────────────────────
    'temperature'      : 1.0,   # Temperatura do softmax (exploração)
    'entropy_coef'     : 0.05,  # Peso do bônus de entropia — Fase 7: 5x maior (anti-colapso)
    'turn_rate'        : 1.05,  # Taxa de giro máxima (rad/s) ≈ 60°/s
                                # ações: {−θ, −θ/2, 0, +θ/2, +θ}

    # ── REINFORCE episódico (Fase 3) ─────────────────────────────────────────
    'gamma'            : 0.99,  # Fator de desconto: ações perto da recompensa pesam mais
    'episode_steps'    : 200,   # H passos por episódio antes de treinar a política
    'lr_decay'         : 0.995, # Decay do learning rate a cada episódio
    'reward_scale'     : 5.0,   # Escala da recompensa — Fase 7: 5x maior (sinal forte)
    'log_csv'          : 'episodios.csv',  # Curva de recompensa média por episódio

    # ── A2C (Fase 8): Actor-Critic ───────────────────────────────────────────
    'value_coef'       : 0.5,   # Peso do critic loss no update combinado
    'gae_lambda'       : 0.95,  # λ do GAE: trade-off viés/variancia do advantage

    # ── Autonomia (curriculum learning) ──────────────────────────────────────
    # (a autonomia por recompensa da Fase 1 foi substituída pelos estágios da
    #  Fase 4; o valor fica como referência do ganho de autonomia esperado)
    'autonomy_scale'   : 120.0,

    # ── Currículo de autonomia (Fase 4) ──────────────────────────────────────
    # Estágio A: imitação pura (warm-up) → Estágio B: híbrido REINFORCE + λ·CE
    # → Estágio C: autonomia plena (λ = 0, professor desligado).
    'stage_a_episodes' : 10,   # Quantos episódios de imitação supervisionada
    'stage_b_episodes' : 20,   # Quantos episódios híbridos (autonomia sobe 0→1)
    'lambda_start'     : 0.5,  # Peso inicial da imitação (CE) no híbrido
    'lambda_decay'     : 0.95, # Decaimento de λ a cada episódio (→ 0 no C)

    # ── Métricas, memória e visualização (Fase 5) ────────────────────────────
    'seed'             : 42,    # Semente global de aleatoriedade (reproduzível)
    'weights_file'     : 'pesos.json',  # Salvar/carregar o cérebro (teclas S/L)
    'grid_cells'       : 9,     # Grade de visualização da política (N × N setas)
    'rolling_window'   : 10,    # Janela da recompensa média exibida no HUD

    # ── Robustez e calibração (Fase 6) ───────────────────────────────────────
    'temperature_decay': 0.995, # Fase 7: decay mais lento (exploração por mais tempo)
    'min_temperature'  : 0.3,   # Fase 7: piso mais alto (nunca 100% greedy)
    'wall_margin'      : 4.0,   # Faixa perto da borda em que o professor foge da parede

    # ── Log ──────────────────────────────────────────────────────────────────
    'log_interval'     : 1.0,    # Intervalo (s) entre linhas de log no jogo

    # ── Câmera ───────────────────────────────────────────────────────────────
    'cam': {
        'rot_speed' : 40.0,
        'pan_speed' : 20.0,
        'zoom_speed': 5.0,
        'min_zoom'  : 5.0,
        'max_zoom'  : 80.0,
    },
}