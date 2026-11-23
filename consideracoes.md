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
- **One-hot na idade**: `idade` vira `idade_15`, `idade_25`, `idade_30`, `idade_45` e `idade_55`. A relação com o churn não é monotônica (churn quase zero nos grupos extremos). Como número, a regressão logística só aprenderia "quanto mais velho, mais (ou menos) churn"; com uma coluna por faixa, cada grupo tem seu coeficiente. As 5 colunas são mantidas, sem categoria de referência: a regularização L2 lida com a colinearidade, e cada coeficiente fica comparável à média, e não a um grupo arbitrário. O one-hot só é necessário na regressão logística; para XGBoost e MLP a mudança é neutra, porque eles aprendem a relação não monotônica sozinhos. O **Naive Bayes e o XGBoost não usam o one-hot**: os notebooks reconstroem a idade numérica a partir das 5 colunas com `idade_numerica` (em `src/comum.py`), linha a linha e sem vazamento. No XGBoost, poucos cortes separam as faixas, e a importância da idade fica numa só coluna em vez de dividida em 5. Com média e variância próprias por classe, duas gaussianas já representam o churn concentrado nas idades do meio, enquanto as 5 colunas one-hot seriam dependentes entre si (exatamente uma vale 1) e violariam a suposição de independência.
- **`log1p` nos atributos de uso** (`falhas_ligacao`, `segundos_uso`, `qtd_ligacoes`, `qtd_sms`, `numeros_distintos`, `valor_cliente`), antes da padronização:
  - Esses atributos têm forte assimetria à direita (cauda longa e muitos zeros), vista nos boxplots da análise exploratória. O `log1p` aceita zero (`log1p(0) = 0`), por isso substitui o `log`.
  - Na MLP e na regressão logística, a cauda deixa de dominar a padronização e o gradiente. Na regressão logística, o efeito passa a ser relativo ("usar o dobro"), o que combina com H6 (quem sai usa de 35% a 46% do que usa quem fica).
  - O Naive Bayes gaussiano é o mais beneficiado, pois supõe normalidade de cada atributo dentro de cada classe.
  - No XGBoost não tem efeito: o log preserva a ordem dos valores, e as árvores só usam a ordem nos cortes.
  - `meses_assinatura` (3 a 47, sem cauda longa) e `faixa_cobranca` (ordinal, 0 a 10) ficam sem log.
- **Padronização (z-score)**: `StandardScaler` (média 0, desvio 1) para MLP e regressão logística, só nos 8 atributos numéricos (os 6 com log, `meses_assinatura` e `faixa_cobranca`). Binários e one-hot ficam em 0/1. O min-max foi descartado porque, com caudas longas, um valor extremo comprime os demais perto de 0. XGBoost e Naive Bayes gaussiano dispensam escala: as árvores usam só a ordem dos valores, e o NB estima média e variância por atributo, de modo que a escala não altera as decisões.
- **Naive Bayes gaussiano para todos os atributos**: o `GaussianNB` trata binários, ordinais e contínuos como normais (a idade entra numérica, sem one-hot). Para os binários isso é uma aproximação grosseira, mas um NB misto (`CategoricalNB` nos discretos + `GaussianNB` nos contínuos, somando os log-likelihoods) não existe pronto no scikit-learn e acrescenta complexidade fora do foco do trabalho. A limitação deve ser comentada no relatório como possível explicação para o desempenho do NB.
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

O Naive Bayes não tem pesos de classe; o equivalente é a probabilidade a priori de cada classe. Por padrão, ele usa as proporções do treino (≈ 0,84 não churn / 0,16 churn). **As configurações principais do NB usam priors uniformes (`priors=[0.5, 0.5]`)**. A configuração `priors-base` (priors da base) serve de contraponto experimental para medir o efeito do desbalanceamento; a análise abaixo diz o que se espera dessa comparação.

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

- Um notebook por algoritmo, mais um de comparação. Todos leem `data/processed/churn-com-status.csv` e `churn-sem-status.csv`, sem refazer nenhuma transformação de base inteira.
- **Módulo comum (`src/comum.py`)**, importado por todos os notebooks de modelagem. Concentra o que precisa ser igual em todos: listas de colunas, leitura das bases (`carregar_base`), folds (`gerar_folds`, `carregar_folds`), pré-processador (`criar_preprocessador`), avaliação (`avaliar`) e gravação dos resultados (`salvar_resultados`).
- **Folds (`data/processed/folds.csv`)**: gerados uma única vez por `src/folds.ipynb` com `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` e reaproveitados por todos os algoritmos, configurações e pelas duas bases.
  - O arquivo tem uma coluna `fold` (0 a 9) com o fold de teste de cada cliente, na ordem das linhas dos CSVs.
  - Os mesmos folds valem para as duas bases: o `StratifiedKFold` estratifica só pelo `y`, e as duas têm as mesmas 2850 linhas, na mesma ordem e com o mesmo `churn`.
  - Cada fold de teste tem 285 clientes e 44 ou 45 churns (15,4% a 15,8%).
  - `carregar_folds` relê o arquivo e confere o tamanho, a estratificação e a igualdade com os folds gerados pela semente. Se uma versão futura do scikit-learn mudar o embaralhamento, o erro aparece na hora.
- Pré-processamento dentro do fold: `criar_preprocessador` monta um `ColumnTransformer` com `StandardScaler` nas 8 colunas numéricas e `passthrough` nas binárias e one-hot, usado nos pipelines da MLP e da regressão logística. Na base sem `inativo`, essa coluna sai da lista de passagem direta.
- Função de avaliação única (`avaliar`): para cada configuração, percorre os 10 folds, cria um modelo novo, ajusta só no treino e guarda, por fold, recall (churn), especificidade, F1 macro, log loss, PR-AUC e matriz de confusão. Devolve a média e o desvio dos 10 folds, a matriz de confusão somada, a probabilidade de churn de cada cliente prevista fora do fold (para as curvas PR) e os modelos de cada fold (para coeficientes e importâncias).
  - O modelo é criado por uma função que recebe o `y` do treino, o que permite calcular o `scale_pos_weight` do XGBoost em cada fold.
  - Uma segunda função opcional devolve os argumentos do `fit`, como o `sample_weight` da MLP (dentro do pipeline, `mlp__sample_weight`).
- `salvar_resultados` grava em `data/results/modelagem/`, para cada algoritmo, configuração e base, as métricas por fold, a matriz somada e as probabilidades fora do fold. O notebook de comparação junta esses arquivos.

### Configurações por algoritmo (mínimo de 2 cada)

Todas usam o critério "balanced" contra o desbalanceamento (pesos de classe, `scale_pos_weight` ou priors uniformes). Para medir o efeito do desbalanceamento, cada algoritmo tem também **uma configuração sem esse tratamento**, comparada fold a fold à versão com tratamento (função `tabela_desbalanceamento` em `src/comum.py`):
- MLP: `relu_adam_32` sem `sample_weight`;
- XGBoost: a configuração escolhida pela parcimônia, sem `scale_pos_weight` (valor 1);
- regressão logística: a configuração escolhida pela parcimônia, sem `class_weight`;
- Naive Bayes: a configuração `priors-base` (a `completo` com os priors estimados no treino).

Espera-se que, sem o tratamento, o recall caia, a especificidade suba e o log loss melhore, com pouca ou nenhuma mudança na PR-AUC (no NB, a PR-AUC deve ser idêntica). Ou seja, o desbalanceamento deve afetar o ponto de operação (o limiar 0,5) mais do que a separação das classes. Essas versões são salvas com o sufixo `_sem-pesos` (ou `priors-base`, no NB).

- **Regressão logística** (com padronização e one-hot da idade), em `src/modelagem-rl.ipynb`:
  - Fixos: `penalty='l2'`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=1000`. A penalidade é indispensável, porque as 5 colunas one-hot somam 1, como o intercepto.
  - **Opção 1**: `C` 1 × 0,1.
  - **Opção 2**: grade `C` ∈ {0,001; 0,01; 0,1; 1; 10; 100}, com 6 configurações × 2 bases. Ela contém a opção 1.
  - Opção 1 × 2: a melhor da grade (PR-AUC na base sem `inativo`) é comparada fold a fold com as duas da opção 1. Pela parcimônia, entre as configurações a menos de um desvio da melhor, fica a de menor `C`.
  - Coeficientes (média ± desvio nos 10 folds e odds ratio) para conferir H1, H2 e H6, e a estabilidade do par `segundos_uso` × `qtd_ligacoes` em função de `C`.
- **Naive Bayes gaussiano** (sem padronização), em `src/modelagem-nb.ipynb`. Todas as configurações usam `var_smoothing=1e-9` (padrão) e idade numérica; as três principais usam `priors=[0.5, 0.5]`:
  - `completo`: todos os atributos, com `log1p`.
  - `sem-redundantes`: sem `segundos_uso`, para medir o efeito da redundância na suposição de independência.
  - `sem-log1p`: atributos de uso de volta à escala original (`expm1`), para medir o efeito do log na suposição de normalidade.
  - `priors-base`: a `completo` com os priors estimados no treino (padrão do `GaussianNB`). É a variação de parâmetro do modelo e a medida do efeito do desbalanceamento.
  - As quatro rodam nas duas bases (8 rodadas). O `var_smoothing` não é variado: ele é proporcional à maior variância entre os atributos e, sem padronização, suavizaria mais os atributos de variância pequena.
- **XGBoost** (sem padronização, idade numérica), em `src/modelagem-xgb.ipynb`:
  - `scale_pos_weight` = nº de não churn / nº de churn no treino de cada fold (~5,4).
  - Fixos: `n_estimators=300`, sem early stopping, `tree_method='hist'`, `random_state=42`. Os demais ficam nos padrões (`reg_lambda=1`, `gamma=0`, `min_child_weight=1`, sem subamostragem).
  - **Opção A**: `max_depth` 3 × 6, com `learning_rate=0,1`.
  - **Opção B**: grade `max_depth` ∈ {2, 3, 4, 6} × `learning_rate` ∈ {0,05; 0,1; 0,3}, com 12 configurações × 2 bases. Ela contém a opção A.
  - A × B: a melhor da grade (PR-AUC na base sem `inativo`) é comparada fold a fold com as duas da opção A. Pela parcimônia, entre as configurações a menos de um desvio da melhor, fica a mais simples.
  - Importância dos atributos por `gain`, normalizada e com média ± desvio nos 10 folds, para conferir H1, H2 e H6.
- **MLP** (com padronização), em `src/modelagem-mlp.ipynb`. Pergunta central: uma camada oculta basta ou a rede precisa de mais de uma?
  - `sample_weight = compute_sample_weight('balanced', y_treino)` no `fit`, calculado em cada fold.
  - 7 configurações × 2 bases = 14 rodadas, variando um fator por vez a partir da base ReLU + Adam com 32 neurônios por camada. A grade inicial, de 13 arquiteturas × 2 ativações × 2 otimizadores (104 rodadas), foi reduzida: as redes afuniladas e expandidas responderiam a uma pergunta (o formato importa?) que não era o foco.
    - Profundidade: `relu_adam` com `(32,)`, `(32,32)` e `(32,32,32)`.
    - Ativação: `tanh_adam` com `(32,)` e `(32,32)`, comparadas às `relu_adam` de mesma arquitetura.
    - Otimizador: `relu_sgd` com `(32,)` e `(32,32)`, comparadas às `relu_adam` de mesma arquitetura.
    - Ativação: a mesma em todas as camadas ocultas. O `MLPClassifier` não permite uma ativação por camada, por isso a ideia de ReLU na 1ª camada e tanh nas demais foi descartada.
    - Otimizador: `adam` com os padrões (lr 0,001, β1 0,9, β2 0,999) ou `sgd` puro (lr 0,001 constante, sem momento).
  - Fixos: 200 épocas (`max_iter=200`), **sem `early_stopping`**, `alpha=1e-4`, `batch_size=200`, `random_state=42`. O `tol` padrão só encerra o treino antes de 200 épocas se a perda de treino parar de cair. Os folds que chegam a 200 épocas sem convergir são registrados.
  - Escolha: a melhor de 1 camada e a melhor de várias camadas pela PR-AUC na base sem `inativo`, avaliadas também na base com `inativo`. Se o ganho da rede mais profunda for menor que o desvio da sua PR-AUC, fica a de 1 camada (parcimônia). A comparação 1 camada × várias é pareada fold a fold, já que todas as configurações usam os mesmos folds.

### Experimentos e comparações

1. **Com × sem `inativo`**: rodar todas as configurações nas duas bases para medir quanto o `Status` infla os resultados.
2. **Referência mínima**: comparar com as regras de um atributo da análise exploratória ("reclamou → churn": recall 0,426; "inativo → churn": recall 0,729).
3. **Efeito do desbalanceamento**: em cada algoritmo, comparar fold a fold uma configuração com e sem o tratamento do desbalanceamento (recall, especificidade, log loss, PR-AUC, FN e FP).
4. **NB com × sem redundantes**: mostrar o efeito da redundância na suposição de independência.
5. **Importância dos atributos**: coeficientes da regressão logística (sobre os dados padronizados) e importância do XGBoost, para confirmar ou não H1, H2 e H6 com todos os atributos juntos.

### Saídas para o relatório

- Tabela com média ± desvio de cada métrica por algoritmo, configuração e base.
- Matriz de confusão de cada classificador (exigência do enunciado).
- Curvas precisão-recall das melhores configurações.
- Figuras em `data/results/modelagem/` e tabelas em CSV, no padrão dos notebooks anteriores.
- Descrição detalhada com pseudocódigo do XGBoost e da regressão logística.

### Escolha das configurações e notebook de comparação

- **Escolha em cada algoritmo**: a configuração que representa o algoritmo é a de maior PR-AUC média na base sem `inativo`. A PR-AUC independe do limiar e não premia quem chama todos de churn; a base sem `inativo` não tem a suspeita de vazamento. Pela parcimônia, entre as configurações a menos de um desvio da melhor, fica a mais simples:
  - MLP: 1 camada, salvo se a mais profunda ganhar mais que o desvio;
  - XGBoost: menor `max_depth` e, empatando, maior `learning_rate`;
  - regressão logística: menor `C`;
  - NB: menos atributos, entre `completo`, `sem-redundantes` e `sem-log1p`.

  Cada notebook grava a sua escolha com `registrar_escolhida` (`<alg>-escolhida.txt`).
- **Limitação (viés otimista, mantido)**: a escolha e a avaliação usam os mesmos folds, sem validação cruzada aninhada. Os números das configurações escolhidas são levemente otimistas, mais nas grades maiores (XGBoost e regressão logística). Isso deve ser declarado no relatório.
- **`src/comparacao.ipynb`**: lê os resultados gravados pelos quatro notebooks, sem treinar nada, e deve ser rodado depois deles. Contém:
  - estratégia de avaliação e fórmulas das medidas;
  - tabela comparativa (com as regras de referência e o classificador trivial);
  - ranking por medida e comparação pareada entre os algoritmos;
  - matrizes de confusão e curvas PR;
  - sobreposição dos erros (Jaccard) e perfil dos clientes que todos erram;
  - efeito consolidado do desbalanceamento e do `inativo`;
  - roteiro da discussão com as perguntas do enunciado.
