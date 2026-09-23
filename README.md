# Mosca simulada

Simulação de um cérebro de mosca controlando um agente em **primeira pessoa**, renderizado com **raycasting** no estilo Doom / Wolfenstein 3D.

O agente enxerga o corredor pelos "olhos" da mosca, e a rede neural decide para onde virar. Ele aprende a desviar de paredes, buscar comida (laranja) e entregar no ninho (ciano).

## O que tem aqui

- **Motor de raycasting** em Python/pygame: paredes 3D com sombreamento por distância e sprites (comida e ninho) desenhados como billboards.
- **Cérebro baseado em conectoma real**: se os arquivos `neurons.csv` e `connections_princeton.csv` estiverem na pasta, a rede usa as conexões reais (fotorreceptores como entrada, neurônios `DNa` como saída motora). Sem eles, o programa gera uma rede sintética e continua funcionando.
- **Aprendizado por reforço**: dor ao bater na parede, dopamina ao pegar e entregar comida, com traço de elegibilidade nas sinapses.
- **Mini-mapa** visto de cima para acompanhar o que o cérebro está fazendo.

## Como rodar

Requer Python 3.9 ou superior.

```bash
pip install pygame numpy
# opcional, só para usar o conectoma real:
pip install pandas scipy

python moscardo_doom3.py
```

## Controles

| Tecla | Ação |
|-------|------|
| Setas | Estimula o cérebro (comida/obstáculo) |
| C | Choque |
| R | Reseta o aprendizado |
| Espaço | Reseta a posição |
| Tab | Liga/desliga o mini-mapa |
| Esc | Sair |

## Dados do conectoma (opcional)

Os arquivos `neurons.csv` e `connections_princeton.csv` usam colunas no formato do conectoma da mosca-da-fruta (`Root ID`, `Primary Cell Type`, `Soma side`, `Predicted NT type`, `pre_root_id`, `post_root_id`, `syn_count`). Eles **não estão incluídos** neste repositório. Baixe os dados na fonte original, respeite a licença e cite os autores do conectoma.

## Ajustes

As constantes no começo do arquivo controlam resolução (`LARGURA`, `ALTURA`), campo de visão, ganho sensorial, taxa de aprendizado e o tamanho da rede sintética.
