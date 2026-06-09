# Brazilian Lyrics Genre Classification

Sistema experimental para classificar generos musicais brasileiros usando apenas letras de musicas em portugues. O projeto segue o briefing do TCC em `prompt_codex_sistema_classificacao.md`.

## Escopo

- Entrada do modelo: somente `lyrics`.
- Saida do modelo: `genre`.
- Metadados como artista, nome da musica, emocoes ou valor numerico nao devem ser usados no treinamento.
- O dataset final esperado fica em `data/raw/letras_generos_balanceado_sem_funk.csv`.
- A base principal remove `funk carioca` e usa 1.480 letras por genero, balanceada pelo tamanho da classe `samba`.

## Estrutura

```text
data/
  raw/
  processed/
models/
results/
  analysis/
  figures/
  metrics/
src/
```

## Instalacao

Crie um ambiente virtual e instale as dependencias:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para treinar CNN ou LSTM com PyTorch:

```bash
pip install -r requirements-neural.txt
```

Para executar o BERTimbau:

```bash
pip install -r requirements-transformer.txt
```

## Preparar Dados

O arquivo de treino deve conter as colunas:

```csv
lyrics,genre
```

Gerar a base processada:

```bash
python -m src.data
```

O script remove nulos, letras muito curtas, duplicatas exatas e, por padrao, os generos `axe`, `funk carioca`, `pagode` e `trilha sonora` caso aparecam.

## Treinar Baseline TF-IDF

Treinamento rapido, sem validacao cruzada:

```bash
python -m src.train_baseline --no-cv
```

Treinamento com validacao cruzada k-fold:

```bash
python -m src.train_baseline --folds 5
```

Classificadores disponiveis:

```bash
python -m src.train_baseline --classifier logreg
python -m src.train_baseline --classifier linearsvc
python -m src.train_baseline --classifier nb
```

Exemplos de variacoes do TF-IDF:

```bash
python -m src.train_baseline --experiment-name word_1_1 --ngram-min 1 --ngram-max 1 --no-cv
python -m src.train_baseline --experiment-name word_1_3 --ngram-min 1 --ngram-max 3 --no-cv
python -m src.train_baseline --experiment-name char_wb_3_5_100k --analyzer char_wb --ngram-min 3 --ngram-max 5 --max-features 100000 --no-cv
python -m src.train_baseline --experiment-name word_char --feature-set word-char --no-cv
```

Artefatos gerados:

- modelo em `models/`;
- metricas em `results/metrics/`;
- matriz de confusao em `results/figures/`;
- comparacao de modelos em `results/metrics/comparacao_modelos.csv`.

## Ver Comparacao de Modelos

```bash
python -m src.evaluate --all
```

Gerar a comparacao final de desempenho, tempo e tamanho:

```bash
python -m src.final_comparison
```

Saidas:

- `results/metrics/comparacao_final_modelos.csv`
- `results/figures/comparacao_final_modelos.png`
- `results/analysis/recomendacao_modelo_final.json`

## Analisar Erros

Gerar os principais pares de generos confundidos pelo baseline principal:

```bash
python -m src.error_analysis
```

As tabelas sao salvas em `results/analysis/` e a matriz de confusao normalizada em `results/figures/`.

## Registro de Experimentos

As etapas de treinamento, configuracoes e metricas ficam registradas em:

```text
docs/registro_treinamento.md
```

As caracteristicas, limitacoes e usos recomendados do modelo final ficam em:

```text
docs/model_card.md
```

## Analise Lexical

```bash
python -m src.lexical_analysis --data data/processed/letras_processadas_balanceado_sem_funk.csv
```

Saidas:

- `results/analysis/top_tfidf_terms_by_genre.csv`
- `results/analysis/top_frequency_terms_by_genre.csv`
- `results/figures/class_distribution.png`
- `results/figures/top_terms_<genero>.png`

## Predizer Genero de Nova Letra

```bash
python -m src.predict --text "texto da letra aqui"
```

Usar o modelo principal recomendado pela comparacao final:

```bash
python -m src.predict --model best --text "texto da letra aqui"
```

Ou por arquivo:

```bash
python -m src.predict --file exemplo_letra.txt
```

## Modelos Neurais

Os modulos `src.train_cnn` e `src.train_lstm` implementam modelos PyTorch. O dispositivo e escolhido automaticamente, usando CPU quando CUDA nao esta disponivel.

Treinar CNN:

```bash
python -m src.train_cnn --epochs 8
```

Treinar LSTM:

```bash
python -m src.train_lstm --epochs 8
```

Treinar LSTM com GloVe NILC pre-treinado:

```bash
python -m src.train_lstm --experiment-name lstm_pytorch_nilc_glove_100d --pretrained-embeddings nilc-glove-100d --embedding-dim 100 --epochs 8
```

Treinar a configuracao regularizada, selecionando pelo F1 macro:

```bash
python -m src.train_lstm --experiment-name lstm_pytorch_nilc_glove_100d_tuned --pretrained-embeddings nilc-glove-100d --embedding-dim 100 --hidden-dim 64 --dropout 0.6 --weight-decay 0.02 --label-smoothing 0.05 --gradient-clip 1.0 --monitor valid_f1_macro --patience 3 --epochs 10
```

Predizer com a CNN:

```bash
python -m src.predict --model cnn_pytorch --text "texto da letra aqui"
```

Predizer com a LSTM bidirecional:

```bash
python -m src.predict --model lstm_pytorch --text "texto da letra aqui"
```

Treinar BERTimbau com o encoder congelado:

```bash
python -m src.train_bertimbau --experiment-name bertimbau_base_frozen --epochs 20 --batch-size 64 --classifier-batch-size 256 --max-len 128 --learning-rate 0.001 --trainable-encoder-layers 0 --cache-frozen-features --patience 4 --device cpu
```

Predizer com BERTimbau:

```bash
python -m src.predict --model bertimbau_base_frozen --text "texto da letra aqui"
```

O script tambem permite liberar as ultimas camadas do encoder com `--trainable-encoder-layers N`, mas essa configuracao exige muito mais tempo e memoria.

Proximas etapas sugeridas:

1. Considerar fine-tuning parcial do BERTimbau caso haja acesso a GPU.
2. Avaliar combinacao com atributos de audio em trabalhos futuros.

## Recomendacao Final

O modelo principal recomendado e o TF-IDF de palavras 1-2 com Logistic Regression. Ele apresentou o melhor F1 macro medio na validacao cruzada, treinou em 11,7 segundos e gerou um artefato de 5,3 MB.

O TF-IDF combinado de palavras e caracteres obteve o maior F1 macro no teste externo, mas teve media inferior na validacao cruzada e custo maior. O BERTimbau congelado foi o melhor neural, ainda abaixo dos modelos classicos.

## Observacao Sobre Reprodutibilidade

O baseline usa `random_state=42`, separacao estratificada treino/teste e, quando habilitada, validacao cruzada estratificada.
