# Model Card - Classificador de Generos por Letras

## Modelo Principal

| Item | Valor |
| --- | --- |
| Nome | `baseline_tfidf_logreg_word_1_2_final` |
| Arquitetura | TF-IDF de palavras + Logistic Regression |
| N-gramas | unigramas e bigramas |
| Atributos maximos | 50.000 |
| Idioma | portugues |
| Entrada | letra de musica |
| Saida | um de 9 generos |
| Artefato | `models/baseline_tfidf_logreg_word_1_2_final.joblib` |

## Uso Pretendido

O modelo foi desenvolvido para experimentos academicos de classificacao de generos musicais brasileiros usando somente o texto das letras.

Usos adequados:

- reproducao dos experimentos do projeto;
- comparacao entre representacoes textuais;
- demonstracoes e prototipos academicos;
- apoio a analises agregadas do dataset.

O modelo nao deve ser usado como fonte definitiva para catalogacao comercial, atribuicao de autoria, recomendacao musical ou decisoes sobre artistas.

## Dados

O conjunto principal contem 13.320 letras, balanceadas em 1.480 exemplos para cada genero:

- forro;
- gospel/religioso;
- mpb;
- pop;
- rap;
- rock;
- romantico;
- samba;
- sertanejo.

Somente as colunas `lyrics` e `genre` participam do treinamento. Artista, titulo, emocoes e demais metadados nao sao usados como entrada.

Divisao:

| Conjunto | Exemplos |
| --- | ---: |
| Treino | 10.656 |
| Teste | 2.664 |

A divisao e estratificada e usa `random_state=42`.

## Desempenho

| Metrica | Valor |
| --- | ---: |
| Acuracia no teste | 0.2842 |
| Precisao macro | 0.2733 |
| Revocacao macro | 0.2842 |
| F1 macro no teste | 0.2744 |
| F1 macro medio em 5 folds | 0.2668 |
| Desvio do F1 macro em 5 folds | 0.0094 |

O modelo combinado de palavras e caracteres obteve F1 macro 0.2805 no teste externo, mas media inferior na validacao cruzada. Por isso, o modelo de palavras 1-2 permanece como recomendacao principal.

## Desempenho por Genero

| Genero | F1-score |
| --- | ---: |
| forro | 0.3107 |
| gospel/religioso | 0.2479 |
| mpb | 0.1727 |
| pop | 0.1765 |
| rap | 0.3826 |
| rock | 0.2802 |
| romantico | 0.1181 |
| samba | 0.3908 |
| sertanejo | 0.3902 |

## Limitacoes

- A acuracia e o F1 macro sao baixos para uso como classificador definitivo.
- Letras de generos distintos compartilham temas e vocabulario, especialmente amor, religiosidade e experiencias cotidianas.
- `romantico`, `mpb` e `pop` apresentam os menores resultados.
- O modelo nao recebe audio, ritmo, instrumentacao, epoca, regiao ou contexto do artista.
- A taxonomia de generos vem do dataset e pode conter rotulos subjetivos ou sobrepostos.
- O sistema trunca ou abstrai informacoes estruturais que nao sejam capturadas pelos n-gramas.
- Resultados nao devem ser generalizados automaticamente para letras fora da distribuicao estudada.

## Principais Confusoes

Os pares com mais erros no teste sao:

| Par | Erros |
| --- | ---: |
| rap e rock | 132 |
| forro e sertanejo | 117 |
| forro e gospel/religioso | 112 |
| mpb e rap | 101 |
| forro e romantico | 86 |

O relatorio completo fica em:

```text
results/analysis/relatorio_erros_baseline_tfidf_logreg_word_1_2_final.json
```

## Uso

Gerar os dados processados e o modelo final:

```bash
python -m src.data
python -m src.train_baseline --classifier logreg --experiment-name word_1_2_final --ngram-min 1 --ngram-max 2 --no-cv
python -m src.final_comparison
```

Predizer:

```bash
python -m src.predict --text "texto da letra aqui"
```

Analisar erros:

```bash
python -m src.error_analysis
```

## Consideracoes Eticas

Generos musicais sao categorias culturais, historicas e frequentemente hibridas. Uma previsao incorreta nao deve ser interpretada como invalidacao da identidade declarada por artistas ou comunidades. O modelo reflete as escolhas, distribuicoes e possiveis vieses do dataset.
