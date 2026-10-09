# Histórico arquivado (eras anteriores — não usar no treino atual)

Arquivado em 2026-10-08 durante o replanejamento (era terreno limpo 8-dim).
Nada aqui carrega no canônico vigente (`8→32→16→5`, CSV 13 cols):
`load_brain` recusa com aviso. Preservado como prova do caminho, não como ativo.

## `csv/` — curvas das eras com labirinto (15 cols, com `colisao_obs`)

| Arquivo | Era | Leitura |
|---------|-----|---------|
| `ppo_A100.csv` / `ppo_Beval.csv` | L17 mapa 32 com muros | treino A / eval B |
| `bc_A100.csv` / `bc_Beval.csv` | L17 mapa 32 com muros | idem (behavioral cloning) |
| `dqn_A100.csv` / `dqn_Beval.csv` | L17 mapa 32 com muros | idem (DQN) |
| `teacher_B100.csv` | L17 mapa 32 com muros | teto do professor (89%) |

Vigente (na raiz): `baseline_limpo.csv` (13 cols, `encontros_food`) e
`episodios.csv` (log vivo do jogo, recriado a cada run).

## `pesos/` — cérebros das eras anteriores (todos incompatíveis hoje)

- `pesos.json` (era luz/chuva, 17-dim) — era o save/load vivo; arquivado.
- `pesos_food_A100.json`, `pesos_food_L4.json` — era labirinto 11-dim.
- `pesos_L5_*.json`, `pesos_L7_*`, `pesos_L8_*`, `pesos_L9_*`, `pesos_L10_*`,
  `pesos_L11_*`, `pesos_L12_*`, `pesos_L13_*` — ablações L5–L13 (11-dim).
- `pesos_L14_bc/dqn.json`, `pesos_L17_*.json` — BC/DQN e veredito L17 (11-dim).
- `pesos_a2c/ppo/buffer_seed_*.json` (8→16→5), `pesos_f10/f11_100.json` (11→16→5),
  `pesos_gate15.json` (15-dim), `pesos_seed_*.json` (legado Hebbian).
- Correspondência prova↔número: `docs/EXPERIMENTOS.md` #1–34.

Regra vigente (`.gitignore`): `episodios.csv` e `pesos.json` da raiz seguem
ignorados (arquivos vivos); a cópia arquivada de `pesos.json` tem exceção
(`!docs/historico/pesos/pesos.json`) — não apagar.
