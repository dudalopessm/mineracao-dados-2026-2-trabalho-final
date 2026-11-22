# Exploração dos dados

## Conhecendo o dataset

Dados de uma companhia de telefonia iraniana em um período de 12 meses - https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset?utm_source=chatgpt.com

São 3150 linhas, e cada instância é um cliente com os seguintes atributos:
```
1. número de ligacoes falhadas (int)
2. reclamacoes (binario) (0: No complaint, 1: complaint)
3. meses totais de assinatura (int) - varia de 3 a 47 meses
4. faixa de cobrança (int) - atributo ordinal (0: lowest amount, 10: highest amount)
5. total de segundos em ligacoes (int)
6. total de ligacoes feitas (int)
7. total de mensagens de texto sms (int)
8. números distintos ligados (int)
9. grupo de idade (int) - atributo ordinal (1: younger age, 5: older age)
10. plano de tarifa (binario) (1: Pay as you go, 2: contractual)
11. status de assinatura (binario) (1: active, 2: non-active)
12. idade (int) - possui apenas 5 valores (15, 25, 30, 45, 55)
13. valor de cliente calculado (contínuo)

classe: churn (binário) (1: churn, 0: non-churn)
```

São 13 atributos preditores. Como idade e grupo de idade carregam a mesma informação, na prática há 12 preditores distintos.

No CSV, alguns nomes de colunas têm espaços duplos ("Call  Failure", "Subscription  Length", "Charge  Amount"), o que deve ser considerado no código.

Os dados estão anonimizados: não há PII (dados que identifiquem o cliente).

Não há valores ausentes, portanto nenhuma técnica de imputação será usada. Há, porém, 300 linhas exatamente duplicadas (restam 2850 linhas únicas).

Inconsistências observadas:
- `Charge Amount` vai de 0 a 10 nos dados, mas a documentação da UCI indica 0 a 9.
- Idade e grupo de idade são redundantes (correspondência 1:1).
- `Status` pode causar vazamento de informação (leakage), pois é quase uma prévia do churn.

Os dados de uso (ligações, segundos, SMS, números distintos, falhas) referem-se aos primeiros 9 meses. Os meses de assinatura representam o tempo total de casa do cliente e não se limitam a essa janela. O rótulo de churn é o estado do cliente ao final de 12 meses; o intervalo de 3 meses entre os dados e o rótulo é o intervalo de planejamento.

Churn é a evasão de clientes de uma empresa. Um cliente churn rompeu o vínculo com a empresa; um não churn permanece ativo.

### Tarefa

**Classificação**: predizer se o cliente continuará na empresa ao final de 1 ano. A classe é o churn: 1 indica que o cliente cancelou e 0, que permaneceu.

### Hipóteses

As hipóteses abaixo guiam a análise exploratória e ajudam a explicar o desempenho dos classificadores.

- **H1**: há algum atributo isolado decisivo para o churn?
- **H2**: quanto ter tido um problema anterior com a empresa (complaint = 1) influencia o churn?
- **H3**: há relação entre o número de ligações falhadas e a reclamação?
- **H4**: se H3 se confirmar, como a tupla (falhas, reclamação) se relaciona com o churn?
- **H5**: o plano de tarifa é decisivo para o churn?
- **H6**: clientes que cancelam usam o serviço com frequência? (Isso pode indicar problemas no pós-venda ou no suporte ao cliente.)

Essas perguntas interessam à administração da empresa, tanto para a expansão do negócio quanto para a qualidade do serviço. Como reter um cliente costuma custar menos do que conquistar um novo, a previsão de churn apoia ações de retenção.

## Análise exploratória

A análise completa (tabelas, gráficos e respostas às hipóteses) está em `src/analise-exploratoria.ipynb`.

### Distribuição das classes

2655 não churn (84,3%) e 495 churn (15,7%), razão de 5,36 : 1. A base é desbalanceada: um modelo que sempre responde "não churn" teria ~84% de acurácia sem acertar nenhum churn.

### Respostas às hipóteses

Números da base sem duplicatas (2850 clientes, 446 churn).

- **H1 (atributo isolado decisivo)**: não há. `Complains` (r = 0,53) e `Status` (r = 0,50) são os atributos mais fortes, mas nenhum resolve sozinho. Essas regras servem de referência mínima para os modelos:
  - "reclamou → churn": recall de 0,426 e especificidade de 0,983. Quase todo cliente que reclama sai, mas 57% dos que saem nunca reclamaram.
  - "inativo → churn": recall de 0,729 e especificidade de 0,851, com suspeita de vazamento.
- **H2 (reclamação)**: confirmada e forte. Quem reclamou sai em 82,6% dos casos, contra 9,8% de quem não reclamou (≈ 8 vezes mais). O efeito independe do status: entre os ativos, 70,4% de quem reclamou saiu, contra 3,1% de quem não reclamou.
- **H3 (falhas × reclamação)**: a relação existe, mas é fraca (r = 0,15; com a taxa de falha, 0,16). Quem reclamou tem em média 11,5 falhas contra 7,5, e taxa de falha de 0,23 contra 0,13. As falhas explicam só parte das reclamações.
- **H4 (falhas, reclamação) × churn**:
  - O número absoluto de falhas não separa o churn (14,5% a 18,1% em todas as faixas). Dentro de cada grupo de reclamação, o churn chega a cair com mais falhas, porque quem usa mais acumula mais falhas e sai menos (H6).
  - A taxa de falha (falhas ÷ ligações) corrige o efeito do volume: sem reclamação, o churn vai de 2,7% (taxa ≤ 0,1) a 21,6% (taxa > 0,2); com reclamação, chega a 89,1%.
  - A reclamação domina, e a taxa de falha alta agrava o risco.
- **H5 (plano de tarifa)**: não é decisivo sozinho. O contratual tem 2,6% de churn contra 16,8% no pré-pago, mas reúne só 8% dos clientes; quase todos os churns estão no pré-pago.
- **H6 (uso)**: quem cancela usa muito menos o serviço. As medianas dos clientes churn ficam entre 35% e 46% das de quem fica (29 contra 64 ligações, 1.256 contra 3.596 segundos, 11 contra 26 SMS), com o mesmo tempo de assinatura (35 meses). Entre os 107 clientes sem nenhum uso, 52,3% saíram, contra 14,2% dos que usam. A queda de uso parece anteceder o cancelamento, o que, junto de H2, aponta para problemas de pós-venda e suporte.

### O que fica para os algoritmos

A análise exploratória olha um ou dois atributos por vez. Os classificadores devem responder:

- **Combinação de atributos**: se reclamação, status, uso e cobrança juntos superam as regras de um atributo só (recall de 0,73 da regra do `Status`).
- **Peso do `Status`**: comparar o desempenho com e sem o atributo para medir quanto ele infla os resultados.
- **Relações não lineares**: a idade (churn quase zero nos grupos extremos), a faixa de cobrança e a interação falhas × reclamação não são lineares. Espera-se que XGBoost e MLP as capturem melhor que a regressão logística e o Naive Bayes.
- **Redundância entre atributos** (segundos × ligações, r = 0,95; SMS × Customer Value, r = 0,92): avaliar o impacto no Naive Bayes, que supõe independência, e na estabilidade dos coeficientes da regressão logística.
- **Importância dos atributos**: usar os coeficientes da regressão logística e a importância do XGBoost para confirmar ou não H1, H2 e H6 com todos os atributos juntos.
- **Efeito do desbalanceamento**: comparar recall e especificidade com e sem pesos de classe (ou priors uniformes).
- **Limites**: o ruído de rótulo (14 combinações conflitantes) e o ruído de medição (números distintos e falhas maiores que o total de ligações) impõem um teto ao desempenho.

# Pré-processamento

## Decisões

- **Duplicatas**: remover as 300 linhas duplicadas (restam 2850). A proporção de churn quase não muda (15,71% → 15,65%), e a remoção evita que a mesma linha caia no treino e no teste.
- **Rótulos conflitantes**: 14 combinações de atributos aparecem com churn 0 e 1. Ficam na base como ruído de rótulo.
- **Age Group**: remover, por ser redundante com Age (correspondência 1:1, r = 0,96).
- **Charge Amount = 10**: tratar como valor válido (7 linhas; é a faixa mais alta, sem churn, como as faixas 5 a 9).
- **Clientes sem nenhum uso** (154 na base original, 107 sem duplicatas): manter. Um cliente pode nunca ter usado o serviço e continuar ou sair, e isso é informação relevante para o churn.
- **Números distintos > total de ligações (120 linhas) e falhas > total de ligações (3 linhas)**: manter. Não são dados inválidos, mas indicam possível ruído de medição, o que deve constar no relatório.
- **Outliers**: não remover. São clientes reais (uso muito alto ou nulo), e o `log1p` já reduz a influência das caudas.
- **Status**: rodar os experimentos com e sem o atributo, para avaliar o vazamento.
- **Nenhum atributo novo**: a taxa de falha (falhas ÷ ligações) foi útil na análise exploratória (H4), mas não entra como atributo. Os modelos recebem só os atributos originais e cabe a eles combinar falhas e ligações. Pelo mesmo motivo não há indicador de clientes sem uso (ver "Efeito do `log1p` e os clientes sem uso").
- **Renomeação das colunas** para `snake_case` em português, sem os espaços duplos:

  | original | novo |
  |---|---|
  | Call  Failure | `falhas_ligacao` |
  | Complains | `reclamacao` |
  | Subscription  Length | `meses_assinatura` |
  | Charge  Amount | `faixa_cobranca` |
  | Seconds of Use | `segundos_uso` |
  | Frequency of use | `qtd_ligacoes` |
  | Frequency of SMS | `qtd_sms` |
  | Distinct Called Numbers | `numeros_distintos` |
  | Tariff Plan | `plano_contratual` |
  | Status | `inativo` |
  | Age | `idade` (vira one-hot) |
  | Customer Value | `valor_cliente` |
  | Churn | `churn` |

- **Recodificação de `Tariff Plan` e `Status` para 0/1**: os dois usam 1/2 na base e passam a ser `plano_contratual` (0: pré-pago, 1: contratual) e `inativo` (0: ativo, 1: inativo). Assim todos os binários (`reclamacao`, `plano_contratual`, `inativo`, `churn`) seguem a convenção 1 = presença da característica, e cada coeficiente da regressão logística é diretamente o efeito de ter a característica.
- **One-hot na idade**: `idade` vira `idade_15`, `idade_25`, `idade_30`, `idade_45` e `idade_55`. A relação com o churn não é monotônica (churn quase zero nos grupos extremos). Como número, a regressão logística só aprenderia "quanto mais velho, mais (ou menos) churn"; com uma coluna por faixa, cada grupo tem seu coeficiente. As 5 colunas são mantidas, sem categoria de referência: a regularização L2 lida com a colinearidade, e cada coeficiente fica comparável à média, e não a um grupo arbitrário. Para XGBoost e MLP a mudança é neutra.
- **`log1p` nos atributos de uso** (`falhas_ligacao`, `segundos_uso`, `qtd_ligacoes`, `qtd_sms`, `numeros_distintos`, `valor_cliente`), antes da padronização:
  - Esses atributos têm forte assimetria à direita (cauda longa e muitos zeros), vista nos boxplots da análise exploratória. O `log1p` aceita zero (`log1p(0) = 0`), por isso substitui o `log`.
  - Na MLP e na regressão logística, a cauda deixa de dominar a padronização e o gradiente. Na regressão logística, o efeito passa a ser relativo ("usar o dobro"), o que combina com H6 (quem sai usa de 35% a 46% do que usa quem fica).
  - O Naive Bayes gaussiano é o mais beneficiado, pois supõe normalidade de cada atributo dentro de cada classe.
  - No XGBoost não tem efeito: o log preserva a ordem dos valores, e as árvores só usam a ordem nos cortes.
  - `meses_assinatura` (3 a 47, sem cauda longa) e `faixa_cobranca` (ordinal, 0 a 10) ficam sem log.
- **Padronização (z-score)**: `StandardScaler` (média 0, desvio 1) para MLP e regressão logística, só nos 8 atributos numéricos (os 6 com log, `meses_assinatura` e `faixa_cobranca`). Binários e one-hot ficam em 0/1. O min-max foi descartado porque, com caudas longas, um valor extremo comprime os demais perto de 0. XGBoost e Naive Bayes gaussiano dispensam escala: as árvores usam só a ordem dos valores, e o NB estima média e variância por atributo, de modo que a escala não altera as decisões.
- **Naive Bayes gaussiano para todos os atributos**: o `GaussianNB` trata binários, one-hot, ordinais e contínuos como normais. Para os binários isso é uma aproximação grosseira, mas um NB misto (`CategoricalNB` nos discretos + `GaussianNB` nos contínuos, somando os log-likelihoods) não existe pronto no scikit-learn e acrescenta complexidade fora do foco do trabalho. A limitação deve ser comentada no relatório como possível explicação para o desempenho do NB.
- **Atributos redundantes**: todos ficam na base principal. MLP, XGBoost e regressão logística com L2 lidam bem com atributos correlacionados, mas o Naive Bayes conta a mesma evidência duas vezes. Por isso haverá uma versão adicional do NB sem os redundantes.
  - Critério: um par é redundante se |r| > 0,9 nos dados que o modelo recebe (já com `log1p`); dentro do par, sai o atributo menos correlacionado com o churn.
  - Correlações na base sem duplicatas:

    | | sem log | com log |
    |---|---|---|
    | r(segundos, ligações) | 0,95 | 0,925 |
    | r(SMS, valor_cliente) | 0,92 | 0,714 |
    | r com churn: segundos / ligações | −0,296 / −0,299 | −0,340 / −0,358 |
    | r com churn: SMS / valor_cliente | −0,219 / −0,287 | −0,204 / −0,356 |

  - Só o par segundos × ligações continua redundante depois do log, e sai `segundos_uso` (−0,340 com o churn, contra −0,358 de `qtd_ligacoes`). A diferença é pequena e a escolha, pouco crítica, mas segue uma regra objetiva.
  - O par SMS × `valor_cliente` cai para 0,71 com o log e os dois ficam. Vale comentar no relatório que o `log1p` desfez boa parte de uma redundância que parecia forte na análise exploratória. A UCI descreve `valor_cliente` apenas como "the calculated value of customer", sem fórmula, então não há base para tratá-lo como derivado dos outros atributos.
  - Essa versão é montada na modelagem, selecionando colunas, sem arquivo separado.
- **Desbalanceamento**: pesos de classe no critério "balanced" (peso inversamente proporcional à frequência da classe):
  - Regressão logística: `class_weight='balanced'`.
  - XGBoost: `scale_pos_weight` = nº de não churn / nº de churn no treino (~5,4).
  - MLP: o `MLPClassifier` não tem `class_weight`; os pesos vão exemplo a exemplo em `sample_weight` no `fit` (`compute_sample_weight('balanced', y_treino)`, calculado em cada fold só com o `y` do treino). O ambiente do projeto (`src/.venv`) usa o scikit-learn 1.9.1, que aceita `sample_weight` no `MLPClassifier`.
  - Naive Bayes: ver a seção sobre priors 0,5 / 0,5.

## Onde cada transformação é feita

Transformações que tratam cada linha isoladamente são feitas uma vez na base inteira, antes da validação cruzada; não há vazamento, porque o resultado de uma linha não depende das outras. As que aprendem parâmetros com os dados são feitas dentro de cada fold, ajustadas só no treino.

| etapa | onde | por quê |
|---|---|---|
| remover duplicatas | base inteira | evita que a mesma linha caia no treino e no teste |
| remover `Age Group` | base inteira | não aprende nada com os dados |
| renomear colunas | base inteira | não aprende nada com os dados |
| recodificar `Tariff Plan` e `Status` para 0/1 | base inteira | mapeamento fixo |
| one-hot da idade | base inteira | as 5 categorias são conhecidas de antemão |
| `log1p` | base inteira | função fixa, aplicada a cada valor |
| padronização (`StandardScaler`) | **dentro do fold** | aprende média e desvio. A transformação é a mesma para MLP e regressão logística (um único `ColumnTransformer` nos dois pipelines), mas os parâmetros mudam a cada fold. Padronizar a base inteira usaria a média e o desvio do fold de teste para transformar o treino |
| pesos de classe / priors | **dentro do fold** | dependem da proporção de churn no treino |

## Arquivos gerados

O notebook `src/pre-processamento.ipynb` lê `data/raw/raw-customer-churn.csv` e salva em `data/processed/`:
- `churn-com-status.csv`: base pré-processada completa (2850 linhas).
- `churn-sem-status.csv`: a mesma base sem a coluna `inativo`, para medir o vazamento.

Os dois arquivos já têm o `log1p` e o one-hot aplicados e estão sem padronização, que fica no pipeline da modelagem.

## Resultado do pré-processamento

O notebook foi executado do início ao fim no ambiente `src/.venv`, com `assert`s em cada etapa, e todos passaram. A figura dos histogramas antes e depois do log está em `data/results/pre-processamento/log1p-antes-depois.png`.

Forma da base em cada etapa:

| etapa | linhas | colunas |
|---|---|---|
| base bruta | 3150 | 14 |
| sem duplicatas | 2850 | 14 |
| sem `Age Group` | 2850 | 13 |
| one-hot da idade | 2850 | 17 |
| base final (com `inativo`) | 2850 | 17 |
| base final sem `inativo` | 2850 | 16 |

Conferências:
- Churn de 15,71% → 15,65% após remover as duplicatas (446 churns).
- `Age Group` ↔ `Age`: correspondência 1:1 confirmada nos dois sentidos antes da remoção.
- Recodificação: 229 clientes no plano contratual e 684 inativos, com contagens iguais às de antes da troca de 1/2 para 0/1.
- One-hot: 112 / 921 / 1296 / 367 / 154 clientes nas faixas de 15, 25, 30, 45 e 55 anos; cada cliente está em exatamente uma faixa.
- Os CSVs salvos foram relidos e são iguais à base em memória.

Colunas da base final, como referência para o `ColumnTransformer` da modelagem:
- **Padronizar** (8 numéricas): `falhas_ligacao`, `meses_assinatura`, `faixa_cobranca`, `segundos_uso`, `qtd_ligacoes`, `qtd_sms`, `numeros_distintos`, `valor_cliente`.
- **Passar direto** (0/1): `reclamacao`, `plano_contratual`, `inativo`, `idade_15`, `idade_25`, `idade_30`, `idade_45`, `idade_55`.
- **Classe**: `churn`.
- **NB sem redundantes**: todas as colunas acima, exceto `segundos_uso`.

# Algoritmos

- Vistos em aula: **MLP** e **Naive Bayes**.
- Não vistos em aula: **XGBoost** e **regressão logística**, que exigem descrição detalhada com pseudocódigo.

## Métricas

- **Acurácia**: não será usada, por causa do desbalanceamento.
- **Recall (classe churn)**: métrica principal. Mede quantos churns o modelo deixou de encontrar (falsos negativos). Falsos alarmes não são o foco, por isso a precisão não será analisada.
- **Especificidade**: mede se o modelo acerta os clientes que ficam.
- **F1 macro**: resume o desempenho geral. O peso igual entre as classes é proposital, para que a classe churn não seja abafada pela majoritária.
- **F1 weighted**: não será usado, pois favorece a classe majoritária.
- **Acurácia balanceada**: não será usada; é a média entre recall e especificidade, já analisados separadamente.
- **Entropia cruzada (log loss)**: mede se os modelos erram com alta confiança.
- **ROC-AUC**: não será usada. Com muito mais não churn do que churn, a taxa de falsos positivos fica baixa mesmo com muitos falsos alarmes, o que deixa a curva otimista.
- **PR-AUC**: será usada, por focar na classe positiva e ser mais informativa em bases desbalanceadas.
- A matriz de confusão de cada classificador e as fórmulas das métricas devem constar no relatório (exigência do enunciado).

## Estratégia de avaliação

Validação cruzada estratificada com 10 folds, mantendo em cada fold a proporção de churn da base (~45 churns por fold de teste). Todas as configurações (pelo menos 2 por algoritmo) e todos os algoritmos usam os mesmos folds (mesma semente), o que torna a comparação justa. Transformações que aprendem com os dados, como a padronização para MLP e regressão logística, são ajustadas só no treino de cada fold (pipeline).

## Efeito do `log1p` e os clientes sem uso

O log comprimiu a cauda direita, mas a assimetria trocou de sinal em vez de ficar perto de zero:

| atributo | assimetria antes | depois |
|---|---|---|
| `falhas_ligacao` | 1,07 | −0,48 |
| `segundos_uso` | 1,31 | −2,79 |
| `qtd_ligacoes` | 1,15 | −1,43 |
| `qtd_sms` | 1,97 | −0,22 |
| `numeros_distintos` | 1,04 | −1,21 |
| `valor_cliente` | 1,42 | −1,34 |

A maior parte dos clientes fica com distribuição próxima da normal depois do log, mas os 107 clientes sem uso continuam em `log1p(0) = 0`, muito abaixo dos demais (em `segundos_uso`, quase todos ficam entre 6 e 10). Esses zeros viram valores extremos à esquerda e produzem a assimetria negativa.

Esses 107 clientes têm zero ao mesmo tempo em `segundos_uso`, `qtd_ligacoes`, `numeros_distintos` e `falhas_ligacao` (79% deles também em `qtd_sms` e `valor_cliente`), e 52,3% saíram.

Decidimos não criar um indicador `sem_uso`, pois ele repetiria o que já está em 4 colunas:
- **XGBoost**: sem ganho. Um único corte (`segundos_uso ≤ 0`) já isola o grupo, e o indicador só dividiria a importância com as colunas de uso.
- **Regressão logística**: único caso com algum ganho. O indicador daria ao grupo um deslocamento próprio no intercepto, e o coeficiente de uso mediria só quem usa. Como o grupo já está no extremo da escala, uma reta "mais uso, menos churn" já o marca como alto risco, então o ganho esperado é pequeno.
- **MLP**: pouco ou nenhum ganho; as ativações não lineares aprendem o limiar "uso = 0" sozinhas.
- **Naive Bayes**: pioraria. O NB já conta a evidência "sem uso" pelo menos 4 vezes, pois trata como independentes colunas que são zero juntas. O indicador seria mais uma contagem e não corrigiria as gaussianas deformadas pelos zeros.

No relatório, mostrar a tabela e a figura e explicar que o grupo sem uso fica isolado depois do log, já está representado em 4 colunas e é tratado de forma diferente por cada modelo. Isso pode explicar diferenças de desempenho, por exemplo se a regressão logística ou o NB ficarem atrás da MLP e do XGBoost. Depois da padronização, esses clientes ficam em torno de −4 desvios nas colunas de uso, valor alto mas aceitável para a MLP e a regressão logística.

## Naive Bayes com priors 0,5 / 0,5

O Naive Bayes não tem pesos de classe; o equivalente é a probabilidade a priori de cada classe. Por padrão, ele usa as proporções do treino (≈ 0,84 não churn / 0,16 churn). Serão testadas duas parametrizações: priors da base e priors uniformes (`priors=[0.5, 0.5]`).

**Por que usar**: com os priors da base, o modelo parte da crença de que um cliente tem ~5,4 vezes mais chance de ficar do que de sair. Para classificar alguém como churn, a evidência dos atributos precisa superar essa vantagem, o que tende a reduzir o recall, a métrica que mais nos interessa. Os priors uniformes fazem a decisão depender só da evidência, o mesmo princípio dos pesos "balanced" dos outros algoritmos.

**Como influencia o resultado**: pelo teorema de Bayes, a chance a posteriori de churn é a razão de verossimilhança multiplicada pela razão dos priors. Trocar 0,16/0,84 por 0,5/0,5 multiplica essa chance por ~5,4 para todos os clientes, o que equivale a baixar o limiar de decisão:
- O **recall** da classe churn sobe, porque mais clientes são classificados como churn.
- A **especificidade** cai, porque parte dos clientes que ficam passa a ser apontada como churn (falsos alarmes).
- O **F1 macro** pode subir ou cair, conforme o balanço entre os dois efeitos.
- O **log loss** tende a piorar: as probabilidades deixam de refletir a proporção real de churn e superestimam o churn.
- A **PR-AUC não muda**: multiplicar a chance de todos pelo mesmo fator não altera a ordem dos clientes por probabilidade, só o ponto de corte. Isso deve ser comentado no relatório, pois mostra que as duas parametrizações separam as classes igualmente e diferem só no ponto de operação.

**Efeito na base**: nenhum. Os priors não alteram os dados nem a distribuição das classes; os 10 folds continuam estratificados com ~15,6% de churn, e a avaliação é feita na distribuição real. Muda apenas a regra de decisão do modelo.

## Próximos passos: aplicação dos 4 algoritmos

### Preparação

- Instalar o XGBoost no `src/.venv` (`pip install xgboost`), que ainda não está no ambiente, e gerar um `requirements.txt` com as versões usadas.
- Criar um notebook (por exemplo `src/modelagem.ipynb`) que lê `data/processed/churn-com-status.csv` e `churn-sem-status.csv`. Nenhuma transformação de base inteira é refeita.
- Folds: um único `StratifiedKFold(n_splits=10, shuffle=True, random_state=<semente fixa>)`, gerado uma vez e reaproveitado por todos os algoritmos, configurações e pelas duas bases.
- Pré-processamento dentro do fold: um `ColumnTransformer` com `StandardScaler` nas 8 colunas numéricas e `passthrough` nas binárias e one-hot, usado nos pipelines da MLP e da regressão logística. Na base sem `inativo`, essa coluna sai da lista de passagem direta.
- Função de avaliação única: para cada configuração, percorre os 10 folds, ajusta o modelo só no treino e guarda, por fold, recall (churn), especificidade, F1 macro, log loss, PR-AUC e matriz de confusão. O resultado é a média e o desvio padrão dos 10 folds, mais a matriz de confusão somada.

### Configurações por algoritmo (mínimo de 2 cada)

Todas usam o critério "balanced" contra o desbalanceamento, salvo indicação contrária.

- **Regressão logística** (com padronização):
  - `class_weight='balanced'`, variando a força da regularização (`C`), por exemplo 1,0 e 0,1.
  - Opcional: sem `class_weight`, para medir o efeito do desbalanceamento.
- **Naive Bayes gaussiano** (sem padronização):
  - Priors da base (padrão).
  - `priors=[0.5, 0.5]`.
  - Sem os redundantes (sem `segundos_uso`), com os priors que se saírem melhor.
- **XGBoost** (sem padronização):
  - `scale_pos_weight` = nº de não churn / nº de churn no treino de cada fold (~5,4).
  - Duas complexidades, por exemplo árvores rasas (`max_depth=3`) e mais profundas (`max_depth=6`), com `n_estimators` e `learning_rate` fixos.
- **MLP** (com padronização):
  - `sample_weight = compute_sample_weight('balanced', y_treino)` no `fit`, calculado em cada fold.
  - Duas arquiteturas, por exemplo uma camada oculta pequena (`(16,)`) e duas camadas (`(32, 16)`), com `early_stopping` ou `max_iter` suficiente para convergir.

### Experimentos e comparações

1. **Com × sem `inativo`**: rodar todas as configurações nas duas bases para medir quanto o `Status` infla os resultados.
2. **Referência mínima**: comparar com as regras de um atributo da análise exploratória ("reclamou → churn": recall 0,426; "inativo → churn": recall 0,729).
3. **Efeito do desbalanceamento**: comparar recall e especificidade com e sem pesos de classe / priors uniformes.
4. **NB com × sem redundantes**: mostrar o efeito da redundância na suposição de independência.
5. **Importância dos atributos**: coeficientes da regressão logística (sobre os dados padronizados) e importância do XGBoost, para confirmar ou não H1, H2 e H6 com todos os atributos juntos.

### Saídas para o relatório

- Tabela com média ± desvio de cada métrica por algoritmo, configuração e base.
- Matriz de confusão de cada classificador (exigência do enunciado).
- Curvas precisão-recall das melhores configurações.
- Figuras em `data/results/modelagem/` e tabelas em CSV, no padrão dos notebooks anteriores.
- Descrição detalhada com pseudocódigo do XGBoost e da regressão logística.
