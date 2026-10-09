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
    # L17-mapa: dobrado 16→32 (mesma densidade temporal: passos e fome
    # reescalados abaixo). wall_margin mantido = faixa relativamente estreita.
    'map_limit'        : 32,   # Labirinto A/B ampliado (era 16)
    'ground_scale'     : 72,   # Tamanho visual do chão (deve ser >= 2*map_limit)

    # ── Sensores / geometria (alimento em terreno limpo) ─────────────────────────
    # L17-mapa: diagonal 32 ≈ 90; 60 mantém gradiente sem saturar tudo em 1.0.
    'sensor_max_dist'  : 60.0,  # Referência p/ normalizar (era 30 no mapa 16)
    'food_radius'       : 3.0,   # Raio de "comeu" (precisa chegar perto)

    # ── Recompensa por progresso (reward shaping, por passo) ─────────────────
    # Sinais fortes para que o RL aprenda a VIRAR (não só andar reto).
    # Objetivo único: só atrair (alimento). Sem termo negativo de luz.
    'progress_scale'      : 3.0,   # Ganho aplicado ao Δ distância (unidades de movimento)
    'eat_bonus'         : 5.0,   # Bônus ao cruzar food_radius
    'inside_food_reward': 0.5,   # Recompensa por permanecer sobre o alimento
    'smell_bonus'       : 0.3,   # Olfato E3: smell=1/(1+dist), bônus contínuo
    'idle_cost'           : 0.0,   # Desativado — o verme não precisa de incentivo p/ andar
    'idle_threshold'      : 0.05,  # Deslocamento/frame abaixo disso conta como "parado"

    # ── Alimento + fome (E3/E4) ───────────────────────────────────────────────
    'hunger_gain'       : 0.5,   # E4: r *= (1 + hunger_gain*hunger); hunger 0→1
    # L17-mapa: episódio dobrou (400 passos) → taxa cai à metade p/ mesma
    # fome por episódio (~0.4). P/ iniciante: fome enche no mesmo ritmo real.
    'hunger_rate'       : 0.001, # Incremento de fome por passo (400 passos ≈ 0.4)
    'n_inputs'          : 8,     # Estado canônico 8-dim: 4 food + 4 propriocepção (sem obstáculos)
    'n_inputs_food'     : 8,     # Alias de compat (train_headless/debug_food usam esta chave)
    'food_limit'        : 24,    # L17-mapa: teto do respawn (era 12 no mapa 16)
    # ── Currículo de distância T8 (Lean: barato e decisivo) ─────────────────
    # Alimento nasce perto e afasta aos poucos: max_d(ep) = start + growth*ep.
    # 0/desligado = legado (uniforme em food_limit). Via --set sem editar código.
    # L17-mapa: start 5→10 e growth dobrado (teto ~24 em ~47eps, mesmo ritmo).
    'food_start_dist'   : 10.0,  # distância máx inicial do spawn/worm
    'food_growth'       : 0.3,   # + por episódio (≈24 em ~47eps com start 10)
    'food_curriculum'   : 0,     # 0=desligado (legado); 1=ligado

    # ── Penalidade por repetição de ação (Fase 7) ────────────────────────────
    # Quebra o colapso da política: se o verme escolhe a mesma ação muitas
    # vezes seguidas, sofre uma penalidade crescente — incentiva alternância.
    'action_repeat_penalty' : 0.1,
    'action_repeat_window'  : 20,

    # ── Recompensa por novidade (L9) ─────────────────────────────────────────
    # Incentiva o agente a visitar estados novos, forçando exploração.
    # Estados novos dão bônus; estados já visitados dão menos bônus (decaimento).
    # Revisão mapa 32: janela 100→200 (50% do episódio de 400 passos, como era
    # 100/200 antes). Janela curta demais inflava o bônus no fim do episódio
    # (estados antigos evaporavam e viravam "novos" de novo).
    'novelty_bonus'        : 0.2,   # Bônus por visitar estado novo
    'novelty_decay'        : 0.99,  # Decaimento do bônus com o tempo
    'novelty_window'       : 200,   # Janela de estados visitados (tamanho do buffer)

    # ── Terreno limpo (sem obstáculos) ─────────────────────────────────────────
    # Labirinto gigante futuro usará labirintos.json; por enquanto só verme + alimento.

    # ── Cérebro ──────────────────────────────────────────────────────────────
    'n_hidden'         : 32,    # Primeira camada oculta
    'n_hidden2'        : 16,    # Segunda camada oculta — arquitetura 8→32→16→5
    'n_outputs'        : 3,     # Saídas contínuas (usado pela Rede_Neural legada)
    'n_actions'        : 5,     # Ações discretas da política (Fase 2)
    'learning_rate'    : 0.05,  # Taxa de aprendizado (alpha) — Fase 7: 5x maior
    'regularization'   : 0.001, # Fator L2

    # ── Política estocástica (Fase 2) ────────────────────────────────────────
    'temperature'      : 1.0,   # Temperatura do softmax (exploração)
    'entropy_coef'     : 0.05,  # Peso do bônus de entropia — Fase 7: 5x maior (anti-colapso)
    'epsilon_greedy'   : 0.0,   # T6: piso de exploração opt-in (0=legado). Food runs usam 0.1 via --set.
    'turn_rate'        : 1.05,  # Taxa de giro máxima (rad/s) ≈ 60°/s
                                # ações: {−θ, −θ/2, 0, +θ/2, +θ}

    # ── REINFORCE episódico (Fase 3) ─────────────────────────────────────────
    'gamma'            : 0.99,  # Fator de desconto: ações perto da recompensa pesam mais
    # L17-mapa: passos dobrados (200→400) p/ mesma cobertura por área
    # (speed 6 → 0.1/passo; 400 passos ≈ 40 unidades ≈ diâmetro do mapa 32).
    # Custo: 2x por episódio. Via --set=episode_steps=N sem editar código.
    'episode_steps'    : 400,   # H passos por episódio antes de treinar a política
    'lr_decay'         : 0.998, # Fase 10: decay mais lento (lr estável por 500 eps)
    'reward_scale'     : 5.0,   # Escala da recompensa — Fase 7: 5x maior (sinal forte)
    'log_csv'          : 'episodios.csv',  # Curva de recompensa média por episódio

    # ── A2C (Fase 8): Actor-Critic ───────────────────────────────────────────
    'value_coef'       : 0.5,   # Peso do critic loss no update combinado
    'gae_lambda'       : 0.95,  # λ do GAE: trade-off viés/variancia do advantage

    # ── PPO (Fase 9): Proximal Policy Optimization ───────────────────────────
    'ppo_clip'         : 0.2,   # ε do clipping: limita razão π_new/π_old em [1−ε, 1+ε]

    # ── DQN (L14b/L16): Q-Learning value-based ────────────────────────────────
    # P/ iniciante: o Q aprende valores em escala crua [-1,1] (estável).
    # O PPO usa reward_scale=5.0; herdar 5.0 explodia o target do Q.
    'q_reward_scale'   : 1.0,   # escala da recompensa SÓ p/ td_update

    # ── Replay Buffer (Fase 13) ──────────────────────────────────────────────
    'buffer_max_episodes'  : 50,    # quantos episódios inteiros o buffer guarda
    'buffer_min_transitions': 999999,  # Fase 10: desativa buffer para treino longo puro PPO (rapido)
    'buffer_epochs'        : 2,     # quantas vezes cada mini-batch é reutilizado (K epochs) — 2 para estabilidade pure-python
    'buffer_batch_size'    : 64,    # transições por mini-batch

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
    'lambda_c'         : 0.0,  # T7 DAGGER-âncora opt-in (0=legado). Food: 0.05 via --set.

    # ── Métricas, memória e visualização (Fase 5) ────────────────────────────
    'seed'             : 42,    # Semente global de aleatoriedade (reproduzível)
    'weights_file'     : 'pesos.json',  # Salvar/carregar o cérebro (teclas S/L)
    'checkpoint_interval': 50,  # Fase 10: salva pesos a cada N eps (anti-perda em run longo)
    'grid_cells'       : 9,     # Grade de visualização da política (N × N setas)
    'rolling_window'   : 10,    # Janela da recompensa média exibida no HUD

    # ── Robustez e calibração (Fase 6) ───────────────────────────────────────
    'temperature_decay': 0.998, # Fase 10: explora por mais tempo (500 eps)
    'min_temperature'  : 0.2,   # Fase 10: permite exploração residual (não congela)
    'wall_margin'      : 4.0,   # BORDA (nao muro): faixa em que o professor foge da parede

    # ── Labirinto gigante (opt-in; baseline limpo intacto) ──────────────────
    # maze_enabled=0 -> terreno limpo (default, 8-dim, sem colisao).
    # maze_enabled=1 -> carrega maze_key de maze_file, com colisao + spawn seguro.
    'maze_enabled'      : 0,     # 0=limpo (baseline); 1=gigante
    'maze_file'         : 'labirintos.json',
    'maze_key'          : 'gigante',
    'maze_seed'         : 42,    # seed do gerador DFS (reproduzivel)
    'maze_clearance'    : 2.5,   # folga verme-parede p/ ser "livre"
    'min_wall_dist'     : 2.5,   # distancia minima spawn/food das paredes
    'min_food_spawn_dist': 8.0,  # anti-colado: comida longe do nascimento
    'worm_radius'       : 1.5,   # raio de colisao da cabeca

    # ── Movimento livre (fix "sempre sobe") ──────────────────────────────────
    # Causa: heading fixo (0,0,1)=norte + giro ~1 grau/passo (180 passos p/
    # meia-volta). Fix: heading aleatorio por episodio + turn_rate maior.
    # turn_rate 1.05->2.4: 180 graus em ~75 passos (era ~180). Via --set.
    'spawn_heading'     : 'random',  # 'random'=qualquer direcao; 'fixed'=norte (legado)

    # ── Log ──────────────────────────────────────────────────────────────────
    'log_interval'     : 1.0,    # Intervalo (s) entre linhas de log no jogo

    # ── Câmera ───────────────────────────────────────────────────────────────
    'cam': {
        'rot_speed' : 40.0,
        'pan_speed' : 20.0,
        'zoom_speed': 5.0,
        'min_zoom'  : 5.0,
        'max_zoom'  : 140.0,  # Revisão mapa 32: cobre a câmera inicial (-125)
    },
}