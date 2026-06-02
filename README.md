# Brazilian Lyrics Genre Classification

Sistema experimental para classificar generos musicais brasileiros usando apenas letras de musicas em portugues. O projeto segue o briefing do TCC em `prompt_codex_sistema_classificacao.md`.

## Escopo

- Entrada do modelo: somente `lyrics`.
- Saida do modelo: `genre`.
- Metadados como artista, nome da musica, emocoes ou valor numerico nao devem ser usados no treinamento.
- O dataset final esperado fica em `data/raw/letras_generos.csv`.

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

As dependencias neurais (`tensorflow`, `torch`, `transformers`) estao comentadas em `requirements.txt` porque sao mais pesadas. Instale-as apenas quando for treinar CNN, LSTM ou BERTimbau.

## Preparar Dados

O arquivo de treino deve conter as colunas:

```csv
lyrics,genre
```

Gerar a base processada:

```bash
python -m src.data --input data/raw/letras_generos.csv --output data/processed/letras_processadas.csv
```

O script remove nulos, letras muito curtas, duplicatas exatas e, por padrao, os generos `axe`, `pagode` e `trilha sonora` caso aparecam.

## Treinar Baseline TF-IDF

Treinamento rapido, sem validacao cruzada:

```bash
python -m src.train_baseline --data data/processed/letras_processadas.csv --no-cv
```

Treinamento com validacao cruzada k-fold:

```bash
python -m src.train_baseline --data data/processed/letras_processadas.csv --folds 5
```

Classificadores disponiveis:

```bash
python -m src.train_baseline --classifier logreg
python -m src.train_baseline --classifier linearsvc
python -m src.train_baseline --classifier nb
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

## Analise Lexical

```bash
python -m src.lexical_analysis --data data/processed/letras_processadas.csv
```

Saidas:

- `results/analysis/top_tfidf_terms_by_genre.csv`
- `results/analysis/top_frequency_terms_by_genre.csv`
- `results/figures/class_distribution.png`
- `results/figures/top_terms_<genero>.png`

## Predizer Genero de Nova Letra

```bash
python -m src.predict --model baseline_tfidf_logreg --text "texto da letra aqui"
```

Ou por arquivo:

```bash
python -m src.predict --model baseline_tfidf_logreg --file exemplo_letra.txt
```

## Modelos Neurais

Os modulos `src.train_cnn` e `src.train_lstm` implementam modelos Keras. Para usa-los, instale TensorFlow removendo o comentario correspondente em `requirements.txt` ou instalando manualmente:

```bash
pip install tensorflow
```

Treinar CNN:

```bash
python -m src.train_cnn --data data/processed/letras_processadas.csv --epochs 5
```

Treinar LSTM:

```bash
python -m src.train_lstm --data data/processed/letras_processadas.csv --epochs 5
```

O modulo `src.train_bertimbau` esta documentado como etapa opcional porque o fine-tuning de Transformers pode exigir mais memoria, tempo ou GPU.

Proxima etapa sugerida:

1. Rodar o baseline TF-IDF e registrar metricas.
2. Rodar CNN e LSTM com poucas epocas para comparacao inicial.
3. Implementar/rodar BERTimbau como experimento opcional com Hugging Face Transformers.

## Observacao Sobre Reprodutibilidade

O baseline usa `random_state=42`, separacao estratificada treino/teste e, quando habilitada, validacao cruzada estratificada.
