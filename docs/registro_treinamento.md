# Registro de Treinamento

Este documento registra as etapas experimentais do sistema de classificacao de generos musicais a partir de letras. A ideia e manter um historico dos dados usados, configuracoes de treino, metricas e observacoes relevantes para o desenvolvimento do TCC.

## Base de Dados Principal

Arquivo bruto:

```text
data/raw/letras_generos_balanceado_sem_funk.csv
```

Arquivo processado:

```text
data/processed/letras_processadas_balanceado_sem_funk.csv
```

Configuracao da base:

| Item | Valor |
| --- | ---: |
| Total de letras | 13.320 |
| Total de generos | 9 |
| Letras por genero | 1.480 |
| Genero usado como limite de balanceamento | samba |
| Generos removidos | axe, funk carioca, pagode, trilha sonora |
| Colunas usadas no treinamento | lyrics, genre |

Distribuicao final:

| Genero | Quantidade |
| --- | ---: |
| forro | 1.480 |
| gospel/religioso | 1.480 |
| mpb | 1.480 |
| pop | 1.480 |
| rap | 1.480 |
| rock | 1.480 |
| romantico | 1.480 |
| samba | 1.480 |
| sertanejo | 1.480 |

Observacao metodologica: o nome do artista, titulo da musica, informacoes emocionais e demais metadados foram removidos da entrada do modelo para reduzir vieses. O modelo recebe apenas a letra da musica (`lyrics`). O genero (`genre`) e usado como rotulo durante o treinamento supervisionado e como referencia para avaliacao.

## Execucao 01 - Baseline TF-IDF

Data da execucao: 2026-06-02.

Objetivo: estabelecer uma linha de base inicial com modelos classicos de aprendizado de maquina usando representacao TF-IDF das letras.

Comandos executados:

```bash
python -m src.data
python -m src.train_baseline --classifier logreg --no-cv
python -m src.train_baseline --classifier linearsvc --no-cv
python -m src.train_baseline --classifier nb --no-cv
python -m src.lexical_analysis
```

Pre-processamento aplicado:

| Etapa | Resultado |
| --- | ---: |
| Linhas lidas | 13.320 |
| Linhas gravadas | 13.320 |
| Letras removidas por valores nulos | 0 |
| Letras removidas por tamanho minimo | 0 |
| Duplicatas removidas nesta etapa | 0 |

Divisao dos dados:

| Conjunto | Quantidade |
| --- | ---: |
| Treino | 10.656 |
| Teste | 2.664 |

Configuracao comum dos baselines:

| Parametro | Valor |
| --- | --- |
| Representacao textual | TF-IDF |
| N-gramas | unigramas e bigramas |
| `min_df` | 2 |
| `max_df` | 0.95 |
| `max_features` | 50.000 |
| `sublinear_tf` | True |
| Separacao treino/teste | estratificada |
| `test_size` | 0.2 |
| `random_state` | 42 |
| Validacao cruzada | nao executada nesta rodada |

Resultados gerais:

| Modelo | Acuracia | Precisao macro | Revocacao macro | F1 macro | F1 weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.2842 | 0.2733 | 0.2842 | 0.2744 | 0.2744 |
| TF-IDF + Naive Bayes | 0.2797 | 0.2712 | 0.2797 | 0.2649 | 0.2649 |
| TF-IDF + LinearSVC | 0.2545 | 0.2483 | 0.2545 | 0.2508 | 0.2508 |

Melhor modelo da rodada:

```text
baseline_tfidf_logreg
```

Metricas por genero do melhor modelo:

| Genero | Precisao | Revocacao | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| forro | 0.2635 | 0.3784 | 0.3107 | 296 |
| gospel/religioso | 0.2491 | 0.2466 | 0.2479 | 296 |
| mpb | 0.2000 | 0.1520 | 0.1727 | 296 |
| pop | 0.2103 | 0.1520 | 0.1765 | 296 |
| rap | 0.3277 | 0.4595 | 0.3826 | 296 |
| rock | 0.2909 | 0.2703 | 0.2802 | 296 |
| romantico | 0.1487 | 0.0980 | 0.1181 | 296 |
| samba | 0.4120 | 0.3716 | 0.3908 | 296 |
| sertanejo | 0.3577 | 0.4291 | 0.3902 | 296 |

Artefatos gerados localmente:

| Tipo | Caminho |
| --- | --- |
| Modelo Logistic Regression | `models/baseline_tfidf_logreg.joblib` |
| Modelo LinearSVC | `models/baseline_tfidf_linearsvc.joblib` |
| Modelo Naive Bayes | `models/baseline_tfidf_nb.joblib` |
| Comparacao de modelos | `results/metrics/comparacao_modelos.csv` |
| Metricas do melhor baseline | `results/metrics/baseline_tfidf_logreg_metrics.json` |
| Matriz de confusao | `results/figures/confusion_matrix_baseline_tfidf_logreg.png` |
| Analise lexical | `results/analysis/top_tfidf_terms_by_genre.csv` |
| Distribuicao de classes | `results/figures/class_distribution.png` |

Observacoes iniciais:

- O melhor resultado inicial foi obtido com Logistic Regression.
- O desempenho ainda e baixo, o que indica que a classificacao por letras e uma tarefa dificil com os recursos atuais.
- As classes `samba`, `sertanejo` e `rap` tiveram os melhores F1-scores na rodada inicial.
- As classes `romantico`, `mpb` e `pop` tiveram os menores F1-scores, possivelmente por sobreposicao tematica e vocabulario menos distintivo.
- Como a base esta balanceada, `accuracy` e `F1 macro` sao metricas especialmente relevantes para comparar os modelos.

## Execucao 02 - Variacoes do TF-IDF

Data da execucao: 2026-06-03.

Objetivo: testar variacoes da representacao TF-IDF antes de avancar para modelos neurais. A avaliacao manteve a mesma base balanceada, separacao treino/teste estratificada, `test_size=0.2` e `random_state=42`.

Alteracao implementada no codigo:

- `src.train_baseline` passou a aceitar parametros de configuracao do TF-IDF pela linha de comando.
- Cada experimento pode receber `--experiment-name`, evitando sobrescrever modelos e metricas anteriores.
- As metricas passaram a registrar a configuracao do vetorizador em `vectorizer_config`.

Comandos principais executados:

```bash
python -m src.train_baseline --classifier logreg --experiment-name word_1_2 --no-cv
python -m src.train_baseline --classifier logreg --experiment-name word_1_1 --ngram-min 1 --ngram-max 1 --no-cv
python -m src.train_baseline --classifier logreg --experiment-name word_1_3 --ngram-min 1 --ngram-max 3 --no-cv
python -m src.train_baseline --classifier logreg --experiment-name word_1_2_100k --ngram-min 1 --ngram-max 2 --max-features 100000 --no-cv
python -m src.train_baseline --classifier logreg --experiment-name word_1_2_min_df_5 --ngram-min 1 --ngram-max 2 --min-df 5 --no-cv
python -m src.train_baseline --classifier logreg --experiment-name char_wb_3_5_100k --analyzer char_wb --ngram-min 3 --ngram-max 5 --max-features 100000 --no-cv
python -m src.train_baseline --classifier linearsvc --experiment-name char_wb_3_5_100k --analyzer char_wb --ngram-min 3 --ngram-max 5 --max-features 100000 --no-cv
```

Resultados da rodada:

| Modelo | Configuracao TF-IDF | Acuracia | Precisao macro | Revocacao macro | F1 macro | F1 weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `char_wb`, 3-5, 100k features | 0.2827 | 0.2750 | 0.2827 | 0.2762 | 0.2762 |
| Logistic Regression | palavras, 1-2, 50k features | 0.2842 | 0.2733 | 0.2842 | 0.2744 | 0.2744 |
| Logistic Regression | palavras, 1-2, `min_df=5` | 0.2823 | 0.2722 | 0.2823 | 0.2723 | 0.2723 |
| Logistic Regression | palavras, 1-1, 50k features | 0.2789 | 0.2713 | 0.2789 | 0.2717 | 0.2717 |
| Logistic Regression | palavras, 1-3, 50k features | 0.2800 | 0.2689 | 0.2800 | 0.2703 | 0.2703 |
| Logistic Regression | palavras, 1-2, 100k features | 0.2812 | 0.2676 | 0.2812 | 0.2690 | 0.2690 |
| LinearSVC | `char_wb`, 3-5, 100k features | 0.2594 | 0.2554 | 0.2594 | 0.2564 | 0.2564 |

Melhor modelo por F1 macro:

```text
baseline_tfidf_logreg_char_wb_3_5_100k
```

Metricas por genero do melhor modelo da rodada:

| Genero | Precisao | Revocacao | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| forro | 0.2695 | 0.3615 | 0.3088 | 296 |
| gospel/religioso | 0.2676 | 0.2703 | 0.2689 | 296 |
| mpb | 0.2035 | 0.1588 | 0.1784 | 296 |
| pop | 0.2041 | 0.1689 | 0.1848 | 296 |
| rap | 0.3187 | 0.3919 | 0.3515 | 296 |
| rock | 0.2756 | 0.2635 | 0.2694 | 296 |
| romantico | 0.1480 | 0.1115 | 0.1272 | 296 |
| samba | 0.4225 | 0.3682 | 0.3935 | 296 |
| sertanejo | 0.3654 | 0.4493 | 0.4030 | 296 |

Observacoes da rodada:

- O uso de n-gramas de caracteres (`char_wb` 3-5) produziu o melhor F1 macro ate o momento, embora com acuracia levemente menor que o baseline de palavras 1-2.
- O aumento de `max_features` para 100.000 em palavras nao melhorou o desempenho.
- Usar apenas unigramas reduziu levemente o resultado, indicando que bigramas contribuem para o baseline.
- A inclusao de trigramas de palavras nao trouxe ganho, possivelmente por aumentar a esparsidade.
- `LinearSVC` com `char_wb` ficou abaixo da regressao logistica nesta configuracao.
- A melhoria ainda e pequena; antes de conclusoes definitivas, a proxima rodada deve usar validacao cruzada nos dois melhores modelos.

## Execucao 03 - Validacao Cruzada dos Melhores Baselines

Data da execucao: 2026-06-08.

Objetivo: verificar se a pequena vantagem observada para os n-gramas de caracteres se mantinha em diferentes divisoes dos dados.

Protocolo:

| Item | Valor |
| --- | --- |
| Metodo | validacao cruzada estratificada |
| Numero de folds | 5 |
| Dados usados nos folds | conjunto de treino, com 10.656 letras |
| Conjunto de teste externo | 2.664 letras |
| Classificador | Logistic Regression |
| `random_state` | 42 |

Comandos executados:

```bash
python -m src.train_baseline --classifier logreg --experiment-name word_1_2_cv5 --ngram-min 1 --ngram-max 2 --folds 5
python -m src.train_baseline --classifier logreg --experiment-name char_wb_3_5_100k_cv5 --analyzer char_wb --ngram-min 3 --ngram-max 5 --max-features 100000 --folds 5
```

Resultados da validacao cruzada:

| Modelo | Acuracia media | Desvio da acuracia | F1 macro medio | Desvio do F1 macro |
| --- | ---: | ---: | ---: | ---: |
| Palavras 1-2, 50k features | 0.2778 | 0.0072 | 0.2668 | 0.0094 |
| `char_wb` 3-5, 100k features | 0.2735 | 0.0062 | 0.2657 | 0.0071 |

Resultados no conjunto de teste externo:

| Modelo | Acuracia | F1 macro |
| --- | ---: | ---: |
| Palavras 1-2, 50k features | 0.2842 | 0.2744 |
| `char_wb` 3-5, 100k features | 0.2827 | 0.2762 |

Conclusoes:

- A representacao por caracteres apresentou o maior F1 macro no conjunto de teste externo, mas essa vantagem nao se repetiu na media dos cinco folds.
- O modelo baseado em palavras obteve a maior acuracia media e o maior F1 macro medio na validacao cruzada.
- As diferencas entre os dois modelos sao pequenas, inferiores a 0.005 nas medias da validacao cruzada.
- O desvio baixo indica que ambos os modelos tiveram comportamento relativamente estavel entre os folds.
- Por apresentar melhor desempenho medio, menor dimensionalidade e interpretacao mais direta dos termos, o TF-IDF de palavras 1-2 permanece como baseline principal.
- O modelo por caracteres continua relevante como fonte complementar de atributos em um experimento combinado.

## Execucao 04 - Combinacao de Palavras e Caracteres

Data da execucao: 2026-06-08.

Objetivo: verificar se a combinacao das duas representacoes TF-IDF complementares melhoraria a classificacao.

Arquitetura do pipeline:

| Componente | Configuracao |
| --- | --- |
| Atributos de palavras | TF-IDF, n-gramas 1-2, 50.000 features |
| Atributos de caracteres | TF-IDF `char_wb`, n-gramas 3-5, 100.000 features |
| Combinacao | `FeatureUnion` |
| Classificador | Logistic Regression |
| Total maximo de atributos | 150.000 |

Alteracao implementada:

- O script `src.train_baseline` passou a aceitar `--feature-set word-char`.
- O modelo combinado e salvo como um pipeline unico, podendo ser usado diretamente por `src.predict`.
- O JSON de metricas registra separadamente as configuracoes de palavras e caracteres.

Comandos executados:

```bash
python -m src.train_baseline --classifier logreg --feature-set word-char --experiment-name word_char --no-cv
python -m src.train_baseline --classifier logreg --feature-set word-char --experiment-name word_char_cv5 --folds 5
```

Resultado no conjunto de teste externo:

| Modelo | Acuracia | F1 macro |
| --- | ---: | ---: |
| Palavras + caracteres | 0.2860 | 0.2805 |
| Palavras 1-2 | 0.2842 | 0.2744 |
| Caracteres 3-5 | 0.2827 | 0.2762 |

Resultado da validacao cruzada:

| Modelo | Acuracia media | Desvio da acuracia | F1 macro medio | Desvio do F1 macro |
| --- | ---: | ---: | ---: | ---: |
| Palavras 1-2 | 0.2778 | 0.0072 | 0.2668 | 0.0094 |
| Caracteres 3-5 | 0.2735 | 0.0062 | 0.2657 | 0.0071 |
| Palavras + caracteres | 0.2705 | 0.0068 | 0.2637 | 0.0075 |

Metricas por genero do modelo combinado no teste externo:

| Genero | Precisao | Revocacao | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| forro | 0.2857 | 0.3716 | 0.3231 | 296 |
| gospel/religioso | 0.2475 | 0.2466 | 0.2470 | 296 |
| mpb | 0.2095 | 0.1791 | 0.1931 | 296 |
| pop | 0.2177 | 0.1824 | 0.1985 | 296 |
| rap | 0.3389 | 0.4088 | 0.3706 | 296 |
| rock | 0.2918 | 0.2770 | 0.2842 | 296 |
| romantico | 0.1593 | 0.1216 | 0.1379 | 296 |
| samba | 0.4051 | 0.3750 | 0.3895 | 296 |
| sertanejo | 0.3536 | 0.4122 | 0.3807 | 296 |

Conclusoes:

- O modelo combinado atingiu a maior acuracia e o maior F1 macro no conjunto de teste externo.
- A melhoria no teste nao se confirmou na validacao cruzada, na qual o modelo combinado obteve as menores medias entre os tres finalistas.
- A uniao direta aumenta a dimensionalidade e pode adicionar atributos redundantes, favorecendo uma divisao especifica sem melhorar a generalizacao.
- O TF-IDF de palavras 1-2 continua sendo o baseline principal por apresentar a melhor media na validacao cruzada.
- O modelo combinado deve ser tratado como experimento complementar, e nao como substituto definitivo do baseline.

## Execucao 05 - Analise dos Pares de Generos Confundidos

Data da execucao: 2026-06-08.

Objetivo: identificar os erros mais frequentes do baseline principal e avaliar quais generos apresentam maior sobreposicao quando apenas a letra e usada como entrada.

Modelo analisado:

```text
baseline_tfidf_logreg_word_1_2_cv5
```

Protocolo:

| Item | Valor |
| --- | ---: |
| Exemplos no conjunto de teste | 2.664 |
| Predicoes corretas | 757 |
| Predicoes incorretas | 1.907 |
| Acuracia | 0.2842 |
| Exemplos por genero | 296 |
| `random_state` | 42 |

Comando executado:

```bash
python -m src.error_analysis --model baseline_tfidf_logreg_word_1_2_cv5
```

Principais pares de generos confundidos:

| Posicao | Par de generos | Erros no par | Percentual de todos os erros |
| ---: | --- | ---: | ---: |
| 1 | rap e rock | 132 | 6.92% |
| 2 | forro e sertanejo | 117 | 6.14% |
| 3 | forro e gospel/religioso | 112 | 5.87% |
| 4 | mpb e rap | 101 | 5.30% |
| 5 | forro e romantico | 86 | 4.51% |
| 6 | pop e rock | 78 | 4.09% |
| 7 | gospel/religioso e pop | 73 | 3.83% |
| 8 | romantico e sertanejo | 67 | 3.51% |
| 9 | mpb e rock | 66 | 3.46% |
| 10 | gospel/religioso e sertanejo | 63 | 3.30% |

Os cinco pares mais frequentes concentraram 548 erros, correspondendo a aproximadamente 28.74% de todas as classificacoes incorretas.

Principais confusoes direcionais:

| Genero real | Genero previsto | Quantidade | Percentual da classe real |
| --- | --- | ---: | ---: |
| gospel/religioso | forro | 75 | 25.34% |
| rock | rap | 74 | 25.00% |
| mpb | rap | 70 | 23.65% |
| forro | sertanejo | 64 | 21.62% |
| romantico | forro | 63 | 21.28% |
| rap | rock | 58 | 19.59% |
| sertanejo | forro | 53 | 17.91% |
| pop | gospel/religioso | 44 | 14.86% |
| mpb | rock | 43 | 14.53% |
| pop | rock | 41 | 13.85% |

Desempenho por classe, ordenado pela revocacao:

| Genero | Revocacao | Erros | Genero mais confundido |
| --- | ---: | ---: | --- |
| romantico | 0.0980 | 267 | forro |
| mpb | 0.1520 | 251 | rap |
| pop | 0.1520 | 251 | gospel/religioso |
| gospel/religioso | 0.2466 | 223 | forro |
| rock | 0.2703 | 216 | rap |
| samba | 0.3716 | 186 | forro |
| forro | 0.3784 | 184 | sertanejo |
| sertanejo | 0.4291 | 169 | forro |
| rap | 0.4595 | 160 | rock |

Artefatos gerados:

| Conteudo | Caminho |
| --- | --- |
| Confusoes direcionais | `results/analysis/confusoes_direcionais_baseline_tfidf_logreg_word_1_2_cv5.csv` |
| Pares consolidados | `results/analysis/pares_confundidos_baseline_tfidf_logreg_word_1_2_cv5.csv` |
| Resumo por genero | `results/analysis/resumo_erros_por_genero_baseline_tfidf_logreg_word_1_2_cv5.csv` |
| Relatorio JSON | `results/analysis/relatorio_erros_baseline_tfidf_logreg_word_1_2_cv5.json` |
| Matriz normalizada | `results/figures/confusion_matrix_normalized_baseline_tfidf_logreg_word_1_2_cv5.png` |

Interpretacao:

- A confusao entre rap e rock ocorre nos dois sentidos e representa o maior bloco de erros do modelo.
- Forro aparece em tres dos cinco pares mais confundidos, sendo frequentemente associado a sertanejo, gospel/religioso e romantico.
- MPB e pop apresentaram baixa revocacao e distribuicao dos erros entre diversas classes, indicando pouca separacao lexical com a representacao atual.
- O genero romantico foi o mais dificil de reconhecer, com apenas 9.80% de revocacao e forte tendencia de classificacao como forro ou sertanejo.
- Como hipotese, parte dessas confusoes decorre do compartilhamento de temas e vocabulario entre as letras, como amor, religiosidade e experiencias cotidianas.
- O sistema nao recebe atributos sonoros, ritmicos ou instrumentais, que poderiam distinguir generos cujas letras possuem conteudo lexical semelhante.
- Esses resultados justificam a comparacao com modelos neurais e, em trabalhos futuros, com abordagens multimodais que combinem letra e audio.

## Execucao 06 - Primeira CNN Textual

Data da execucao: 2026-06-09.

Objetivo: comparar o baseline TF-IDF com um primeiro modelo neural convolucional treinado diretamente sobre sequencias de palavras.

Backend:

- A implementacao inicial em TensorFlow foi substituida por PyTorch porque o ambiente do projeto utiliza Python 3.14, para o qual nao havia distribuicao TensorFlow disponivel no indice do `pip`.
- Foi utilizado PyTorch 2.12.0 em CPU, pois o computador nao possui GPU CUDA.
- A mesma implementacao PyTorch foi preparada para a futura LSTM, evitando manter dois backends neurais diferentes.

Arquitetura:

| Componente | Configuracao |
| --- | --- |
| Vocabulario maximo | 30.000 palavras |
| Tamanho real do vocabulario | 30.000 palavras |
| Comprimento maximo | 300 tokens |
| Embedding | 128 dimensoes, treinavel |
| Convolucoes | tamanhos 3, 4 e 5 |
| Filtros por convolucao | 96 |
| Agregacao | max pooling global |
| Dropout | 0.5 |
| Saida | 9 classes com logits |
| Otimizador | AdamW |
| Taxa de aprendizado | 0.001 |
| Funcao de perda | entropia cruzada |

Protocolo:

| Item | Valor |
| --- | ---: |
| Treino | 9.590 |
| Validacao | 1.066 |
| Teste | 2.664 |
| Batch size | 128 |
| Epocas solicitadas | 8 |
| Epocas executadas | 8 |
| Paciencia do early stopping | 2 |
| Dispositivo | CPU |
| Tempo de treinamento | 399.4 segundos |
| Melhor perda de validacao | 1.9925 |

Comando executado:

```bash
python -m src.train_cnn --epochs 8 --batch-size 128 --max-words 30000 --max-len 300 --embedding-dim 128 --hidden-dim 96 --dropout 0.5 --learning-rate 0.001 --patience 2 --device cpu
```

Resultados:

| Modelo | Acuracia | Precisao macro | Revocacao macro | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| CNN PyTorch | 0.2294 | 0.2364 | 0.2294 | 0.1959 |
| TF-IDF palavras 1-2 | 0.2842 | 0.2733 | 0.2842 | 0.2744 |
| TF-IDF palavras + caracteres | 0.2860 | 0.2788 | 0.2860 | 0.2805 |

Metricas por genero da CNN:

| Genero | Precisao | Revocacao | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| forro | 0.1716 | 0.5439 | 0.2609 | 296 |
| gospel/religioso | 0.2195 | 0.0912 | 0.1289 | 296 |
| mpb | 0.2000 | 0.0304 | 0.0528 | 296 |
| pop | 0.2500 | 0.0946 | 0.1373 | 296 |
| rap | 0.2857 | 0.4459 | 0.3483 | 296 |
| rock | 0.2355 | 0.2196 | 0.2273 | 296 |
| romantico | 0.2273 | 0.0169 | 0.0314 | 296 |
| samba | 0.2548 | 0.3108 | 0.2801 | 296 |
| sertanejo | 0.2831 | 0.3108 | 0.2963 | 296 |

Distribuicao das previsoes:

| Genero previsto | Quantidade | Percentual |
| --- | ---: | ---: |
| forro | 938 | 35.21% |
| rap | 462 | 17.34% |
| samba | 361 | 13.55% |
| sertanejo | 325 | 12.20% |
| rock | 276 | 10.36% |
| gospel/religioso | 123 | 4.62% |
| pop | 112 | 4.20% |
| mpb | 45 | 1.69% |
| romantico | 22 | 0.83% |

Artefatos gerados:

| Conteudo | Caminho |
| --- | --- |
| Pesos e configuracao | `models/cnn_pytorch.pt` |
| Vocabulario | `models/cnn_pytorch_vocabulary.json` |
| Rotulos | `models/cnn_pytorch_labels.json` |
| Metricas e historico | `results/metrics/cnn_pytorch_metrics.json` |
| Curvas de treinamento | `results/figures/training_history_cnn_pytorch.png` |
| Matriz de confusao | `results/figures/confusion_matrix_cnn_pytorch.png` |

Interpretacao:

- A CNN ficou abaixo dos modelos TF-IDF tanto em acuracia quanto em F1 macro.
- A perda de treino continuou diminuindo, enquanto a perda de validacao estabilizou proxima de 2.0, indicando inicio de sobreajuste.
- A acuracia de treino chegou a 39.09%, mas a melhor acuracia de validacao foi 24.95%.
- O modelo concentrou 35.21% das previsoes em forro, apesar de cada genero representar 11.11% do conjunto de teste.
- MPB e romantico foram raramente previstos, o que reduziu significativamente o F1 macro.
- O resultado indica que, nesta base e configuracao, a representacao esparsa TF-IDF generaliza melhor do que embeddings treinados do zero.
- Possiveis melhorias incluem embeddings pre-treinados em portugues, regularizacao adicional e ajuste da arquitetura; contudo, a comparacao inicial deve ser preservada como resultado experimental.

## Execucao 07 - Primeira LSTM Bidirecional

Data da execucao: 2026-06-09.

Objetivo: avaliar se uma arquitetura recorrente bidirecional, capaz de considerar a ordem das palavras nos dois sentidos, melhoraria a classificacao em relacao a CNN textual e aos baselines TF-IDF.

Alteracao implementada:

- A BiLSTM passou a usar `pack_padded_sequence` com o comprimento real de cada letra.
- Os tokens de preenchimento a direita deixaram de influenciar os estados finais usados pelo classificador.
- A interface permaneceu compativel com o modulo de predicao e com a CNN existente.

Arquitetura:

| Componente | Configuracao |
| --- | --- |
| Vocabulario maximo | 30.000 palavras |
| Tamanho real do vocabulario | 30.000 palavras |
| Comprimento maximo | 300 tokens |
| Embedding | 128 dimensoes, treinavel |
| Camada recorrente | LSTM bidirecional |
| Unidades por direcao | 96 |
| Representacao final | concatenacao dos estados finais das duas direcoes |
| Dropout | 0.5 |
| Saida | 9 classes com logits |
| Otimizador | AdamW |
| Taxa de aprendizado | 0.001 |
| Funcao de perda | entropia cruzada |

Protocolo:

| Item | Valor |
| --- | ---: |
| Treino | 9.590 |
| Validacao | 1.066 |
| Teste | 2.664 |
| Batch size | 128 |
| Epocas solicitadas | 8 |
| Epocas executadas | 7 |
| Paciencia do early stopping | 2 |
| Dispositivo | CPU |
| Tempo de treinamento | 1.717,0 segundos |
| Melhor perda de validacao | 2.0395 |

Comando executado:

```bash
python -m src.train_lstm --epochs 8 --batch-size 128 --max-words 30000 --max-len 300 --embedding-dim 128 --hidden-dim 96 --dropout 0.5 --learning-rate 0.001 --patience 2 --device cpu
```

Resultados:

| Modelo | Acuracia | Precisao macro | Revocacao macro | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| LSTM bidirecional PyTorch | 0.2083 | 0.2048 | 0.2083 | 0.2020 |
| CNN PyTorch | 0.2294 | 0.2364 | 0.2294 | 0.1959 |
| TF-IDF palavras 1-2 | 0.2842 | 0.2733 | 0.2842 | 0.2744 |
| TF-IDF palavras + caracteres | 0.2860 | 0.2788 | 0.2860 | 0.2805 |

Metricas por genero da LSTM bidirecional:

| Genero | Precisao | Revocacao | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| forro | 0.1834 | 0.2095 | 0.1956 | 296 |
| gospel/religioso | 0.1888 | 0.1588 | 0.1725 | 296 |
| mpb | 0.2188 | 0.1182 | 0.1535 | 296 |
| pop | 0.1924 | 0.2230 | 0.2066 | 296 |
| rap | 0.2550 | 0.3446 | 0.2931 | 296 |
| rock | 0.2047 | 0.2365 | 0.2194 | 296 |
| romantico | 0.1656 | 0.0845 | 0.1119 | 296 |
| samba | 0.1930 | 0.2230 | 0.2069 | 296 |
| sertanejo | 0.2419 | 0.2770 | 0.2583 | 296 |

Distribuicao das previsoes:

| Genero previsto | Quantidade | Percentual |
| --- | ---: | ---: |
| rap | 400 | 15.02% |
| pop | 343 | 12.88% |
| rock | 342 | 12.84% |
| samba | 342 | 12.84% |
| sertanejo | 339 | 12.73% |
| forro | 338 | 12.69% |
| gospel/religioso | 249 | 9.35% |
| mpb | 160 | 6.01% |
| romantico | 151 | 5.67% |

Artefatos gerados:

| Conteudo | Caminho |
| --- | --- |
| Pesos e configuracao | `models/lstm_pytorch.pt` |
| Vocabulario | `models/lstm_pytorch_vocabulary.json` |
| Rotulos | `models/lstm_pytorch_labels.json` |
| Metricas e historico | `results/metrics/lstm_pytorch_metrics.json` |
| Curvas de treinamento | `results/figures/training_history_lstm_pytorch.png` |
| Matriz de confusao | `results/figures/confusion_matrix_lstm_pytorch.png` |

Interpretacao:

- A LSTM bidirecional ficou abaixo da CNN em acuracia, mas obteve F1 macro ligeiramente maior.
- A distribuicao das previsoes foi consideravelmente mais equilibrada que a da CNN, reduzindo a concentracao excessiva em `forro`.
- `rap` apresentou o melhor F1 da LSTM, seguido por `sertanejo`; `romantico` permaneceu como a classe mais dificil.
- A perda de treino caiu continuamente, enquanto a perda de validacao parou de melhorar apos a quinta epoca. O treinamento foi encerrado na setima epoca e os pesos da quinta epoca foram restaurados.
- A BiLSTM exigiu aproximadamente 4,3 vezes o tempo de treinamento da CNN em CPU e nao superou os modelos TF-IDF.
- Os resultados reforcam que, para esta quantidade de dados, embeddings treinados do zero ainda generalizam pior que a representacao esparsa de palavras e caracteres.

## Execucao 08 - LSTM Bidirecional com GloVe em Portugues

Data da execucao: 2026-06-09.

Objetivo: verificar se embeddings pre-treinados em portugues melhorariam a BiLSTM, mantendo o restante do protocolo comparavel ao da Execucao 07.

Fonte dos embeddings:

| Item | Valor |
| --- | --- |
| Modelo | NILC GloVe 100d |
| Repositorio | `nilc-nlp/glove-100d` |
| Idioma | portugues brasileiro e europeu |
| Dimensoes | 100 |
| Licenca informada no repositorio | LGPL-3.0 |
| Ajuste durante o treino | habilitado |

Alteracoes implementadas:

- `src.train_lstm` passou a aceitar `--pretrained-embeddings nilc-glove-100d`.
- O pipeline baixa os vetores pelo Hugging Face Hub e carrega somente as linhas correspondentes ao vocabulario do experimento.
- As palavras sem vetor pre-treinado mantem a inicializacao aleatoria.
- O token de padding permanece com vetor zerado.
- As metricas registram fonte, cobertura e congelamento dos embeddings.
- `--experiment-name` permite preservar os artefatos da LSTM anterior.

Cobertura:

| Item | Valor |
| --- | ---: |
| Palavras elegiveis no vocabulario | 29.998 |
| Palavras encontradas no GloVe | 20.429 |
| Cobertura | 68,10% |

Protocolo:

| Item | Valor |
| --- | ---: |
| Treino | 9.590 |
| Validacao | 1.066 |
| Teste | 2.664 |
| Comprimento maximo | 300 tokens |
| Unidades da BiLSTM por direcao | 96 |
| Batch size | 128 |
| Dropout | 0.5 |
| Taxa de aprendizado | 0.001 |
| Epocas solicitadas | 8 |
| Epocas executadas | 7 |
| Paciencia do early stopping | 2 |
| Dispositivo | CPU |
| Tempo de treinamento | 1.731,9 segundos |
| Melhor perda de validacao | 1.9968 |

Comando executado:

```bash
python -m src.train_lstm --experiment-name lstm_pytorch_nilc_glove_100d --pretrained-embeddings nilc-glove-100d --epochs 8 --batch-size 128 --max-words 30000 --max-len 300 --embedding-dim 100 --hidden-dim 96 --dropout 0.5 --learning-rate 0.001 --patience 2 --device cpu
```

Resultados:

| Modelo | Acuracia | Precisao macro | Revocacao macro | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| LSTM + NILC GloVe 100d | 0.2297 | 0.2239 | 0.2297 | 0.2066 |
| LSTM com embedding aleatorio | 0.2083 | 0.2048 | 0.2083 | 0.2020 |
| CNN com embedding aleatorio | 0.2294 | 0.2364 | 0.2294 | 0.1959 |
| TF-IDF palavras 1-2 | 0.2842 | 0.2733 | 0.2842 | 0.2744 |
| TF-IDF palavras + caracteres | 0.2860 | 0.2788 | 0.2860 | 0.2805 |

Metricas por genero:

| Genero | Precisao | Revocacao | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| forro | 0.2107 | 0.2534 | 0.2301 | 296 |
| gospel/religioso | 0.2353 | 0.1622 | 0.1920 | 296 |
| mpb | 0.2791 | 0.0405 | 0.0708 | 296 |
| pop | 0.1748 | 0.2534 | 0.2069 | 296 |
| rap | 0.2809 | 0.5068 | 0.3614 | 296 |
| rock | 0.2007 | 0.1993 | 0.2000 | 296 |
| romantico | 0.0941 | 0.0270 | 0.0420 | 296 |
| samba | 0.3018 | 0.2264 | 0.2587 | 296 |
| sertanejo | 0.2374 | 0.3986 | 0.2976 | 296 |

Distribuicao das previsoes:

| Genero previsto | Quantidade | Percentual |
| --- | ---: | ---: |
| rap | 534 | 20,05% |
| sertanejo | 497 | 18,66% |
| pop | 429 | 16,10% |
| forro | 356 | 13,36% |
| rock | 294 | 11,04% |
| samba | 222 | 8,33% |
| gospel/religioso | 204 | 7,66% |
| romantico | 85 | 3,19% |
| mpb | 43 | 1,61% |

Artefatos gerados:

| Conteudo | Caminho |
| --- | --- |
| Pesos e configuracao | `models/lstm_pytorch_nilc_glove_100d.pt` |
| Vocabulario | `models/lstm_pytorch_nilc_glove_100d_vocabulary.json` |
| Rotulos | `models/lstm_pytorch_nilc_glove_100d_labels.json` |
| Metricas e historico | `results/metrics/lstm_pytorch_nilc_glove_100d_metrics.json` |
| Curvas de treinamento | `results/figures/training_history_lstm_pytorch_nilc_glove_100d.png` |
| Matriz de confusao | `results/figures/confusion_matrix_lstm_pytorch_nilc_glove_100d.png` |

Interpretacao:

- Os embeddings pre-treinados aumentaram a acuracia da BiLSTM em 0.0214 e o F1 macro em 0.0046.
- A nova BiLSTM igualou aproximadamente a acuracia da CNN e obteve F1 macro superior.
- `rap`, `sertanejo` e `samba` se beneficiaram da inicializacao pre-treinada.
- A distribuicao voltou a se concentrar em algumas classes; `mpb` e `romantico` foram raramente previstos e tiveram queda de F1.
- O tempo de treinamento permaneceu praticamente igual ao da LSTM com embeddings aleatorios.
- A cobertura de 68,10% foi suficiente para produzir ganho, mas o resultado continuou abaixo dos modelos TF-IDF.
- O experimento confirma que embeddings pre-treinados ajudam, mas nao resolvem a sobreposicao lexical entre os generos nem a baixa separacao das classes mais dificeis.

## Execucao 09 - Ajuste de Regularizacao da LSTM com GloVe

Data da execucao: 2026-06-09.

Objetivo: reduzir o sobreajuste observado nas LSTMs e selecionar os pesos de acordo com a metrica principal do projeto, o F1 macro.

Alteracoes implementadas:

- O historico passou a registrar F1 macro de validacao por epoca.
- O early stopping pode monitorar `valid_loss` ou `valid_f1_macro`.
- Foram adicionados parametros para `weight_decay`, label smoothing e clipping de gradiente.
- A leitura da tabela comparativa passou a aceitar arquivos UTF-8 com ou sem BOM.
- Os valores padrao anteriores foram preservados para manter compatibilidade com as execucoes existentes.

Configuracao ajustada:

| Componente | Execucao 08 | Execucao 09 |
| --- | ---: | ---: |
| Dimensoes do GloVe | 100 | 100 |
| Unidades por direcao | 96 | 64 |
| Dropout | 0.5 | 0.6 |
| `weight_decay` | 0.01 | 0.02 |
| Label smoothing | 0.0 | 0.05 |
| Clipping de gradiente | nao aplicado | 1.0 |
| Monitor do early stopping | perda de validacao | F1 macro de validacao |
| Paciencia | 2 | 3 |

Protocolo:

| Item | Valor |
| --- | ---: |
| Treino | 9.590 |
| Validacao | 1.066 |
| Teste | 2.664 |
| Vocabulario | 30.000 palavras |
| Cobertura do GloVe | 68,10% |
| Comprimento maximo | 300 tokens |
| Batch size | 128 |
| Taxa de aprendizado | 0.001 |
| Epocas solicitadas | 10 |
| Epocas executadas | 9 |
| Melhor epoca | 6 |
| Melhor F1 macro de validacao | 0.2143 |
| Dispositivo | CPU |
| Tempo de treinamento | 1.538,7 segundos |

Comando executado:

```bash
python -m src.train_lstm --experiment-name lstm_pytorch_nilc_glove_100d_tuned --pretrained-embeddings nilc-glove-100d --epochs 10 --batch-size 128 --max-words 30000 --max-len 300 --embedding-dim 100 --hidden-dim 64 --dropout 0.6 --learning-rate 0.001 --weight-decay 0.02 --label-smoothing 0.05 --gradient-clip 1.0 --monitor valid_f1_macro --patience 3 --device cpu
```

Resultados:

| Modelo | Acuracia | Precisao macro | Revocacao macro | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| LSTM + GloVe ajustada | 0.2264 | 0.2135 | 0.2264 | 0.2103 |
| LSTM + GloVe original | 0.2297 | 0.2239 | 0.2297 | 0.2066 |
| LSTM com embedding aleatorio | 0.2083 | 0.2048 | 0.2083 | 0.2020 |
| CNN com embedding aleatorio | 0.2294 | 0.2364 | 0.2294 | 0.1959 |
| TF-IDF palavras 1-2 | 0.2842 | 0.2733 | 0.2842 | 0.2744 |
| TF-IDF palavras + caracteres | 0.2860 | 0.2788 | 0.2860 | 0.2805 |

Metricas por genero:

| Genero | Precisao | Revocacao | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| forro | 0.2283 | 0.2128 | 0.2203 | 296 |
| gospel/religioso | 0.2216 | 0.1453 | 0.1755 | 296 |
| mpb | 0.1867 | 0.1047 | 0.1342 | 296 |
| pop | 0.1825 | 0.1622 | 0.1717 | 296 |
| rap | 0.2982 | 0.3818 | 0.3348 | 296 |
| rock | 0.2185 | 0.2872 | 0.2482 | 296 |
| romantico | 0.1111 | 0.0304 | 0.0477 | 296 |
| samba | 0.2101 | 0.4088 | 0.2775 | 296 |
| sertanejo | 0.2647 | 0.3041 | 0.2830 | 296 |

Distribuicao das previsoes:

| Genero previsto | Quantidade | Percentual |
| --- | ---: | ---: |
| samba | 576 | 21,62% |
| rock | 389 | 14,60% |
| rap | 379 | 14,23% |
| sertanejo | 340 | 12,76% |
| forro | 276 | 10,36% |
| pop | 263 | 9,87% |
| gospel/religioso | 194 | 7,28% |
| mpb | 166 | 6,23% |
| romantico | 81 | 3,04% |

Artefatos gerados:

| Conteudo | Caminho |
| --- | --- |
| Pesos e configuracao | `models/lstm_pytorch_nilc_glove_100d_tuned.pt` |
| Vocabulario | `models/lstm_pytorch_nilc_glove_100d_tuned_vocabulary.json` |
| Rotulos | `models/lstm_pytorch_nilc_glove_100d_tuned_labels.json` |
| Metricas e historico | `results/metrics/lstm_pytorch_nilc_glove_100d_tuned_metrics.json` |
| Curvas de treinamento | `results/figures/training_history_lstm_pytorch_nilc_glove_100d_tuned.png` |
| Matriz de confusao | `results/figures/confusion_matrix_lstm_pytorch_nilc_glove_100d_tuned.png` |

Interpretacao:

- A configuracao ajustada atingiu o melhor F1 macro entre os modelos neurais, com ganho de 0.0037 sobre a LSTM GloVe original.
- A acuracia caiu 0.0034, mostrando que a melhoria ocorreu principalmente pelo maior equilibrio entre as classes.
- O F1 de `mpb` aumentou de 0.0708 para 0.1342 e o de `rock` de 0.2000 para 0.2482.
- `rap` perdeu parte do desempenho, mas permaneceu como a classe de maior F1.
- `romantico` continuou com baixa revocacao, indicando que regularizacao e reducao da arquitetura nao resolvem sua sobreposicao lexical.
- As previsoes ficaram menos concentradas em `rap`, `sertanejo` e `pop`, mas passaram a favorecer `samba`.
- O modelo ajustado ainda ficou abaixo dos baselines TF-IDF; novos ajustes incrementais na mesma arquitetura tem retorno provavelmente limitado.

## Execucao 10 - BERTimbau com Encoder Congelado

Data da execucao: 2026-06-09.

Objetivo: avaliar representacoes contextuais pre-treinadas em portugues usando BERTimbau, com uma configuracao compativel com o ambiente de 16 GB de RAM e sem GPU CUDA.

Modelo:

| Item | Valor |
| --- | --- |
| Checkpoint | `neuralmind/bert-base-portuguese-cased` |
| Arquitetura | BERT Base |
| Camadas do encoder | 12 |
| Parametros totais carregados | 108.930.057 |
| Parametros treinaveis | 6.921 |
| Estrategia | encoder congelado e classificador linear treinavel |
| Biblioteca | Transformers 5.10.2 |
| Backend | PyTorch 2.12.0 CPU |

Alteracoes implementadas:

- `src.train_bertimbau` deixou de ser um placeholder e passou a executar treinamento e avaliacao completos.
- Foram adicionados divisao estratificada, early stopping por F1 macro, acumulacao de gradiente, warmup, clipping e congelamento parcial ou total do encoder.
- Para o encoder totalmente congelado, o script pode calcular os vetores contextuais uma unica vez com `--cache-frozen-features`.
- O classificador e treinado sobre os vetores `[CLS]` armazenados em memoria, evitando repetir o BERT a cada epoca.
- `src.predict` passou a carregar e usar modelos Transformers salvos como diretorios Hugging Face.
- As dependencias opcionais foram registradas em `requirements-transformer.txt`.

Protocolo:

| Item | Valor |
| --- | ---: |
| Treino | 9.590 |
| Validacao | 1.066 |
| Teste | 2.664 |
| Comprimento maximo | 128 subwords |
| Batch do encoder | 64 |
| Batch do classificador | 256 |
| Taxa de aprendizado | 0.001 |
| `weight_decay` | 0.01 |
| Warmup | 10% |
| Clipping de gradiente | 1.0 |
| Epocas solicitadas | 20 |
| Epocas executadas | 14 |
| Melhor epoca | 10 |
| Melhor F1 macro de validacao | 0.2246 |
| Paciencia | 4 |
| Tempo de extracao dos vetores | 1.417,1 segundos |
| Tempo total | 1.419,9 segundos |
| Dispositivo | CPU |

Comando executado:

```bash
python -m src.train_bertimbau --experiment-name bertimbau_base_frozen --epochs 20 --batch-size 64 --classifier-batch-size 256 --max-len 128 --learning-rate 0.001 --weight-decay 0.01 --warmup-ratio 0.1 --gradient-clip 1.0 --trainable-encoder-layers 0 --cache-frozen-features --patience 4 --device cpu
```

Resultados:

| Modelo | Acuracia | Precisao macro | Revocacao macro | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| BERTimbau com encoder congelado | 0.2301 | 0.2189 | 0.2301 | 0.2151 |
| LSTM + GloVe ajustada | 0.2264 | 0.2135 | 0.2264 | 0.2103 |
| LSTM + GloVe original | 0.2297 | 0.2239 | 0.2297 | 0.2066 |
| CNN com embedding aleatorio | 0.2294 | 0.2364 | 0.2294 | 0.1959 |
| TF-IDF palavras 1-2 | 0.2842 | 0.2733 | 0.2842 | 0.2744 |
| TF-IDF palavras + caracteres | 0.2860 | 0.2788 | 0.2860 | 0.2805 |

Metricas por genero:

| Genero | Precisao | Revocacao | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| forro | 0.2220 | 0.3277 | 0.2647 | 296 |
| gospel/religioso | 0.2151 | 0.1926 | 0.2032 | 296 |
| mpb | 0.1458 | 0.0946 | 0.1148 | 296 |
| pop | 0.2029 | 0.1419 | 0.1670 | 296 |
| rap | 0.2807 | 0.4324 | 0.3404 | 296 |
| rock | 0.2257 | 0.2196 | 0.2226 | 296 |
| romantico | 0.1582 | 0.0845 | 0.1101 | 296 |
| samba | 0.2616 | 0.1520 | 0.1923 | 296 |
| sertanejo | 0.2577 | 0.4257 | 0.3210 | 296 |

Distribuicao das previsoes:

| Genero previsto | Quantidade | Percentual |
| --- | ---: | ---: |
| sertanejo | 489 | 18,36% |
| rap | 456 | 17,12% |
| forro | 437 | 16,40% |
| rock | 288 | 10,81% |
| gospel/religioso | 265 | 9,95% |
| pop | 207 | 7,77% |
| mpb | 192 | 7,21% |
| samba | 172 | 6,46% |
| romantico | 158 | 5,93% |

Artefatos gerados:

| Conteudo | Caminho |
| --- | --- |
| Modelo e tokenizer | `models/bertimbau_base_frozen/` |
| Configuracao de inferencia | `models/bertimbau_base_frozen/training_config.json` |
| Metricas e historico | `results/metrics/bertimbau_base_frozen_metrics.json` |
| Curvas de treinamento | `results/figures/training_history_bertimbau_base_frozen.png` |
| Matriz de confusao | `results/figures/confusion_matrix_bertimbau_base_frozen.png` |

Interpretacao:

- O BERTimbau congelado atingiu o melhor F1 macro entre os modelos neurais, superando a LSTM GloVe ajustada em 0.0048.
- O ganho confirma que representacoes contextuais em portugues ajudam mesmo sem atualizar o encoder.
- `rap` e `sertanejo` apresentaram os melhores F1-scores, enquanto `mpb` e `romantico` permaneceram entre as classes mais dificeis.
- O classificador concentrou previsoes em `sertanejo`, `rap` e `forro`, mas manteve distribuicao menos extrema que a primeira CNN.
- Quase todo o custo computacional ocorreu na unica passagem de extracao dos vetores; as 14 epocas do classificador consumiram menos de tres segundos adicionais.
- O modelo ocupa aproximadamente 436 MB, muito mais que as LSTMs, e ainda ficou abaixo dos baselines TF-IDF.
- Um fine-tuning parcial ou completo do encoder pode melhorar o resultado, mas exigiria custo substancialmente maior em CPU.

## Execucao 11 - Consolidacao Final dos Modelos

Data da execucao: 2026-06-09.

Objetivo: comparar as principais familias de modelos em desempenho, tempo de treinamento, tamanho para inferencia e estabilidade experimental.

Alteracoes implementadas:

- O baseline passou a registrar `training_seconds` e `model_size_bytes`.
- Os dois finalistas TF-IDF foram reproduzidos sem validacao cruzada para medir custo e tamanho no ambiente atual.
- A CNN foi reproduzida para restaurar seu artefato e medir o pacote completo de inferencia.
- O modulo `src.final_comparison` passou a gerar tabela, grafico e recomendacao final de forma automatizada.
- O tamanho dos modelos neurais inclui pesos, vocabulario e rotulos; o tamanho do BERTimbau inclui modelo, tokenizer e configuracoes.

Comandos executados:

```bash
python -m src.train_baseline --classifier logreg --experiment-name word_1_2_final --ngram-min 1 --ngram-max 2 --no-cv
python -m src.train_baseline --classifier logreg --feature-set word-char --experiment-name word_char_final --no-cv
python -m src.train_cnn --epochs 8 --batch-size 128 --max-words 30000 --max-len 300 --embedding-dim 128 --hidden-dim 96 --dropout 0.5 --learning-rate 0.001 --patience 2 --device cpu
python -m src.final_comparison
```

Comparacao consolidada:

| Modelo | Acuracia | F1 macro teste | F1 macro CV | Tempo | Tamanho |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF palavras + caracteres | 0.2860 | 0.2805 | 0.2637 | 49,1 s | 12,4 MB |
| TF-IDF palavras 1-2 | 0.2842 | 0.2744 | 0.2668 | 11,7 s | 5,3 MB |
| BERTimbau congelado | 0.2301 | 0.2151 | nao executada | 23,7 min | 416,2 MB |
| BiLSTM + GloVe ajustada | 0.2264 | 0.2103 | nao executada | 25,6 min | 12,4 MB |
| BiLSTM com embedding aleatorio | 0.2083 | 0.2020 | nao executada | 28,6 min | 16,0 MB |
| CNN reproduzida | 0.2305 | 0.1926 | nao executada | 3,7 min | 15,9 MB |

Observacao sobre a reproducao da CNN:

- A Execucao 06 havia obtido acuracia 0.2294, F1 macro 0.1959 e tempo de 399,4 segundos.
- A reproducao obteve acuracia 0.2305, F1 macro 0.1926 e tempo de 223,8 segundos.
- A pequena variacao e compatível com diferencas numericas e de execucao em CPU; a conclusao comparativa nao mudou.

Relacoes de custo:

- O BERTimbau levou aproximadamente 121 vezes o tempo do TF-IDF de palavras e ocupou 78 vezes mais espaco.
- A BiLSTM GloVe ajustada levou aproximadamente 131 vezes o tempo do TF-IDF de palavras.
- O TF-IDF combinado aumentou o F1 macro de teste em 0.0061 sobre o modelo de palavras, mas exigiu 4,2 vezes o tempo e 2,3 vezes o espaco.
- Todos os modelos neurais ficaram abaixo dos dois finalistas TF-IDF no conjunto de teste.

Decisao final:

```text
Modelo principal recomendado: TF-IDF de palavras 1-2 com Logistic Regression.
```

Justificativa:

- apresentou o maior F1 macro medio na validacao cruzada entre os finalistas;
- ficou apenas 0.0061 abaixo do melhor F1 no teste externo;
- foi o modelo mais rapido e o menor entre os comparados;
- oferece interpretabilidade direta dos termos e implantacao simples;
- apresentou melhor equilibrio entre generalizacao, custo e desempenho.

O modelo combinado de palavras e caracteres permanece como o melhor resultado pontual no teste externo. O BERTimbau congelado e o melhor modelo neural, mas seu custo e tamanho nao se justificam diante do desempenho inferior.

Artefatos gerados:

| Conteudo | Caminho |
| --- | --- |
| Tabela final | `results/metrics/comparacao_final_modelos.csv` |
| Grafico final | `results/figures/comparacao_final_modelos.png` |
| Recomendacao estruturada | `results/analysis/recomendacao_modelo_final.json` |
| Modelo principal | `models/baseline_tfidf_logreg_word_1_2_final.joblib` |
| Melhor teste externo | `models/baseline_tfidf_logreg_word_char_final.joblib` |

## Execucao 12 - Operacionalizacao do Modelo Principal

Data da execucao: 2026-06-09.

Objetivo: tornar a recomendacao final o comportamento padrao do sistema, validar o checkpoint definitivo e documentar seu uso e suas limitacoes.

Alteracoes implementadas:

- `src.predict` passou a usar `--model best` por padrao.
- `src.error_analysis` passou a analisar `best` por padrao.
- Ambos consultam `results/analysis/recomendacao_modelo_final.json`, evitando nomes fixos de checkpoints antigos.
- Os exemplos do README passaram a funcionar com os artefatos realmente gerados.
- Foi criado `docs/model_card.md` com dados, metricas, usos pretendidos, limitacoes e consideracoes eticas.

Comandos validados:

```bash
python -m src.predict --text "Eu canto o amor e a saudade no meu coracao"
python -m src.error_analysis
```

Resultado da verificacao de reprodutibilidade:

| Item | Execucao 05 | Checkpoint final |
| --- | ---: | ---: |
| Exemplos de teste | 2.664 | 2.664 |
| Predicoes corretas | 757 | 757 |
| Erros | 1.907 | 1.907 |
| Acuracia | 0.2842 | 0.2842 |
| Principal par confundido | rap e rock | rap e rock |
| Erros no principal par | 132 | 132 |

Os dez principais pares confundidos foram reproduzidos exatamente, confirmando que o checkpoint final corresponde a configuracao escolhida nas etapas anteriores.

Artefatos gerados:

| Conteudo | Caminho |
| --- | --- |
| Model card | `docs/model_card.md` |
| Relatorio final de erros | `results/analysis/relatorio_erros_baseline_tfidf_logreg_word_1_2_final.json` |
| Pares confundidos | `results/analysis/pares_confundidos_baseline_tfidf_logreg_word_1_2_final.csv` |
| Resumo por genero | `results/analysis/resumo_erros_por_genero_baseline_tfidf_logreg_word_1_2_final.csv` |
| Matriz normalizada | `results/figures/confusion_matrix_normalized_baseline_tfidf_logreg_word_1_2_final.png` |

## Proximas Rodadas Planejadas

1. Considerar fine-tuning parcial do BERTimbau somente se houver acesso a GPU.
2. Avaliar abordagens multimodais com audio em trabalhos futuros.
3. Usar o TF-IDF de palavras 1-2 como modelo principal do sistema atual.
