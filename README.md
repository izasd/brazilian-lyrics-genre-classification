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

As dependencias de Transformers continuam opcionais porque sao mais pesadas.

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
python -m src.predict --model baseline_tfidf_logreg --text "texto da letra aqui"
```

Ou por arquivo:

```bash
python -m src.predict --model baseline_tfidf_logreg --file exemplo_letra.txt
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

Predizer com a CNN:

```bash
python -m src.predict --model cnn_pytorch --text "texto da letra aqui"
```

O modulo `src.train_bertimbau` esta documentado como etapa opcional porque o fine-tuning de Transformers pode exigir mais memoria, tempo ou GPU.

Proximas etapas sugeridas:

1. Rodar CNN e LSTM com poucas epocas para comparacao inicial.
2. Comparar custo computacional, acuracia e F1 macro dos modelos.
3. Implementar ou rodar BERTimbau como experimento opcional com Hugging Face Transformers.

## Observacao Sobre Reprodutibilidade

O baseline usa `random_state=42`, separacao estratificada treino/teste e, quando habilitada, validacao cruzada estratificada.
