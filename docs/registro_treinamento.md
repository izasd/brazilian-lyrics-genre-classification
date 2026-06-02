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

## Proximas Rodadas Planejadas

1. Testar variacoes do baseline TF-IDF, como remocao de stopwords, apenas unigramas, ou ajuste de `max_features`.
2. Executar validacao cruzada estratificada para reduzir dependencia de uma unica divisao treino/teste.
3. Comparar os resultados com modelos neurais, inicialmente CNN e LSTM.
4. Avaliar a matriz de confusao para identificar pares de generos frequentemente confundidos.
5. Registrar todos os novos experimentos neste documento.
