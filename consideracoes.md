# Exploração dos dados

## Conhecendo o dataset

Dados de 12 meses de uma companhia de telefonia iraniana (UCI, [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)).

São 3150 linhas, uma por cliente, com os atributos:
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

Idade e grupo de idade carregam a mesma informação, então os 13 atributos preditores se reduzem a 12 na prática.

Os dados de uso (ligações, segundos, SMS, números distintos e falhas) cobrem os primeiros 9 meses. Os meses de assinatura contam todo o tempo de casa do cliente, sem se limitar a essa janela. O churn é o estado do cliente ao fim dos 12 meses, e os 3 meses entre os dados e o rótulo são o intervalo de planejamento. Um cliente churn rompeu o vínculo com a empresa; um não churn continua ativo.

Os dados são anônimos (sem PII) e não têm valores ausentes, então nenhuma imputação é necessária. Há 300 linhas exatamente duplicadas (restam 2850 únicas). No CSV, alguns nomes de colunas têm espaços duplos ("Call  Failure", "Subscription  Length", "Charge  Amount").

Inconsistências:
- `Charge Amount` vai de 0 a 10 nos dados, e a documentação da UCI diz 0 a 9.
- Idade e grupo de idade têm correspondência 1:1.
- `Status` pode vazar informação do rótulo, porque o cliente inativo é quase um churn anunciado.

### Tarefa

Classificação: prever se o cliente continuará na empresa ao fim de 1 ano. A classe é o churn (1: cancelou; 0: permaneceu).

### Hipóteses

As hipóteses guiam a análise exploratória e ajudam a explicar o desempenho dos classificadores.

- **H1**: há algum atributo isolado decisivo para o churn?
- **H2**: quanto ter tido um problema anterior com a empresa (complaint = 1) influencia o churn?
- **H3**: há relação entre o número de ligações falhadas e a reclamação?
- **H4**: se H3 se confirmar, como a tupla (falhas, reclamação) se relaciona com o churn?
- **H5**: o plano de tarifa é decisivo para o churn?
- **H6**: clientes que cancelam usam o serviço com frequência? (Pouco uso pode indicar problemas no pós-venda ou no suporte ao cliente.)

As perguntas interessam à administração, tanto para a expansão do negócio quanto para a qualidade do serviço. Reter um cliente costuma custar menos que conquistar um novo, e a previsão de churn diz onde agir.

## Análise exploratória

A análise completa, com tabelas e gráficos, está em `src/analise-exploratoria.ipynb`.

### Distribuição das classes

São 2655 não churn (84,3%) e 495 churn (15,7%), uma razão de 5,36 para 1. Um modelo que sempre respondesse "não churn" teria cerca de 84% de acurácia sem acertar nenhum churn.

### Respostas às hipóteses

Números da base sem duplicatas (2850 clientes, 446 churn).

- **H1 (atributo isolado decisivo)**: não há. `Complains` (r = 0,53) e `Status` (r = 0,50) são os mais fortes, mas nenhum resolve sozinho. As regras de um atributo servem de referência mínima para os modelos:
  - "reclamou → churn": recall de 0,426 e especificidade de 0,983. Quase todo cliente que reclama sai, mas 57% dos que saem nunca reclamaram.
  - "inativo → churn": recall de 0,729 e especificidade de 0,851, com suspeita de vazamento.
- **H2 (reclamação)**: confirmada e forte. Quem reclamou sai em 82,6% dos casos, contra 9,8% de quem não reclamou (cerca de 8 vezes mais). O efeito independe do status: entre os ativos, 70,4% de quem reclamou saiu, contra 3,1% de quem não reclamou.
- **H3 (falhas × reclamação)**: a relação existe, mas é fraca (r = 0,15; com a taxa de falha, 0,16). Quem reclamou tem em média 11,5 falhas, contra 7,5, e taxa de falha de 0,23, contra 0,13. As falhas explicam só parte das reclamações.
- **H4 (falhas, reclamação) × churn**:
  - O número absoluto de falhas não separa o churn (de 14,5% a 18,1% em todas as faixas). Dentro de cada grupo de reclamação, o churn chega a cair com mais falhas, porque quem usa mais acumula mais falhas e sai menos (H6).
  - A taxa de falha (falhas ÷ ligações) corrige o efeito do volume: sem reclamação, o churn vai de 2,7% (taxa ≤ 0,1) a 21,6% (taxa > 0,2); com reclamação, chega a 89,1%.
  - A reclamação domina, e a taxa de falha alta agrava o risco.
- **H5 (plano de tarifa)**: não é decisivo sozinho. O contratual tem 2,6% de churn, contra 16,8% no pré-pago, mas reúne só 8% dos clientes, e quase todos os churns estão no pré-pago.
- **H6 (uso)**: quem cancela usa bem menos o serviço. As medianas dos clientes churn ficam entre 35% e 46% das de quem fica (29 contra 64 ligações, 1.256 contra 3.596 segundos, 11 contra 26 SMS), com o mesmo tempo de assinatura (35 meses). Entre os 107 clientes sem nenhum uso, 52,3% saíram, contra 14,2% dos que usam. A queda de uso parece anteceder o cancelamento, o que, junto de H2, aponta para problemas de pós-venda e suporte.

### O que fica para os algoritmos

A análise exploratória olha um ou dois atributos por vez. Ficaram para os classificadores:

- se reclamação, status, uso e cobrança juntos superam as regras de um atributo (recall de 0,73 da regra do `Status`);
- quanto o `Status` infla os resultados, comparando o desempenho com e sem ele;
- as relações não lineares: idade (churn quase zero nos grupos extremos), faixa de cobrança e a interação falhas × reclamação, que XGBoost e MLP devem captar melhor que a regressão logística e o Naive Bayes;
- o efeito dos pares redundantes (segundos × ligações, r = 0,95; SMS × `Customer Value`, r = 0,92) no Naive Bayes, que supõe independência, e na estabilidade dos coeficientes da regressão logística;
- a importância dos atributos (coeficientes da regressão logística e ganho do XGBoost), para conferir H1, H2 e H6 com todos os atributos juntos;
- o teto imposto pelo ruído de rótulo (14 combinações de atributos com rótulos conflitantes) e pelo ruído de medição (números distintos e falhas maiores que o total de ligações).

# Pré-processamento

## Decisões

- **Duplicatas**: removidas as 300 linhas duplicadas (restam 2850). A proporção de churn quase não muda (de 15,71% para 15,65%), e a mesma linha não pode cair no treino e no teste.
- **Rótulos conflitantes**: as 14 combinações de atributos com churn 0 e 1 ficam na base, como ruído de rótulo.
- **Age Group**: removido, por ser redundante com Age (correspondência 1:1, r = 0,96).
- **Charge Amount = 10**: tratado como valor válido. São 7 linhas da faixa mais alta, sem churn, como nas faixas 5 a 9.
- **Clientes sem nenhum uso** (154 na base original, 107 sem duplicatas): mantidos. Um cliente pode nunca ter usado o serviço e ficar ou sair, e isso informa sobre o churn.
- **Números distintos > total de ligações (120 linhas) e falhas > total de ligações (3 linhas)**: mantidos. Não são dados inválidos, mas indicam possível ruído de medição, a ser citado no relatório.
- **Outliers**: mantidos. São clientes reais (uso muito alto ou nulo), e o `log1p` reduz a influência das caudas.
- **Status**: os experimentos rodam com e sem o atributo, para medir o vazamento.
- **Nenhum atributo novo**: a taxa de falha ajudou na análise exploratória (H4), mas os modelos recebem só os atributos originais e cabe a eles combinar falhas e ligações. Pelo mesmo motivo não há indicador de clientes sem uso (ver "Efeito do `log1p` e os clientes sem uso").
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

- **`Tariff Plan` e `Status` em 0/1**: na base eles usam 1/2 e passam a ser `plano_contratual` (0: pré-pago, 1: contratual) e `inativo` (0: ativo, 1: inativo). Todos os binários (`reclamacao`, `plano_contratual`, `inativo`, `churn`) seguem a convenção 1 = presença da característica, e cada coeficiente da regressão logística é o efeito de ter a característica.
- **One-hot na idade**: `idade` vira `idade_15`, `idade_25`, `idade_30`, `idade_45` e `idade_55`. A relação com o churn não é monotônica (churn quase zero nos grupos extremos). Como número, a idade daria à regressão logística um único coeficiente ("quanto mais velho, mais ou menos churn"); com uma coluna por faixa, cada grupo tem o seu. As 5 colunas ficam, sem categoria de referência: a regularização L2 resolve a colinearidade com o intercepto, e cada coeficiente fica comparado à média, e não a um grupo arbitrário.
  - Só a regressão logística precisa do one-hot. A MLP o recebe porque usa a mesma base padronizada, e para ela a mudança é neutra.
  - Naive Bayes e XGBoost recebem a idade numérica, reconstruída das 5 colunas por `idade_numerica` (em `src/comum.py`), linha a linha e sem vazamento. No XGBoost, poucos cortes separam as faixas, e a importância da idade fica numa só coluna. No NB, as 5 colunas one-hot seriam dependentes entre si (exatamente uma vale 1) e violariam a independência, enquanto duas gaussianas com média e variância próprias já podem representar o churn concentrado nas idades do meio.
- **`log1p` nos atributos de uso** (`falhas_ligacao`, `segundos_uso`, `qtd_ligacoes`, `qtd_sms`, `numeros_distintos`, `valor_cliente`), antes da padronização:
  - Esses atributos têm forte assimetria à direita (cauda longa e muitos zeros), vista nos boxplots da análise exploratória. O `log1p` aceita zero (`log1p(0) = 0`), por isso foi usado no lugar do `log`.
  - Na MLP e na regressão logística, a cauda deixa de dominar a padronização e o gradiente. Na regressão logística o efeito passa a ser relativo ("usar o dobro"), o que combina com H6.
  - O Naive Bayes gaussiano é o mais beneficiado, porque supõe normalidade de cada atributo dentro de cada classe.
  - No XGBoost não tem efeito: o log preserva a ordem dos valores, e as árvores só usam a ordem nos cortes.
  - `meses_assinatura` (3 a 47, sem cauda longa) e `faixa_cobranca` (ordinal, 0 a 10) ficam sem log.
- **Padronização (z-score)**: `StandardScaler` (média 0, desvio 1) só para MLP e regressão logística, nos 8 atributos numéricos (os 6 com log, `meses_assinatura` e `faixa_cobranca`). Binários e one-hot ficam em 0/1. O min-max foi descartado porque, com caudas longas, um valor extremo comprime os demais perto de 0. XGBoost e Naive Bayes gaussiano dispensam escala: as árvores usam só a ordem dos valores, e o NB estima média e variância de cada atributo, então a escala não altera as decisões.
- **Naive Bayes gaussiano em todos os atributos**: o `GaussianNB` trata binários, ordinais e contínuos como normais. Para os binários a aproximação é grosseira, mas um NB misto (`CategoricalNB` nos discretos e `GaussianNB` nos contínuos, somando os log-likelihoods) não existe pronto no scikit-learn e foge do foco do trabalho. A limitação entra no relatório como possível explicação para o desempenho do NB.
- **Atributos redundantes**: todos ficam na base principal. MLP, XGBoost e regressão logística com L2 lidam bem com atributos correlacionados, mas o Naive Bayes conta a mesma evidência duas vezes, por isso ganhou uma versão sem os redundantes.
  - Critério: um par é redundante se |r| > 0,9 nos dados que o modelo recebe (já com `log1p`). Dentro do par, sai o atributo menos correlacionado com o churn.
  - Correlações na base sem duplicatas:

    | | sem log | com log |
    |---|---|---|
    | r(segundos, ligações) | 0,95 | 0,925 |
    | r(SMS, valor_cliente) | 0,92 | 0,714 |
    | r com churn: segundos / ligações | −0,296 / −0,299 | −0,340 / −0,358 |
    | r com churn: SMS / valor_cliente | −0,219 / −0,287 | −0,204 / −0,356 |

  - Só o par segundos × ligações continua redundante depois do log, e sai `segundos_uso` (−0,340 com o churn, contra −0,358 de `qtd_ligacoes`). A diferença é pequena, mas a regra é objetiva.
  - O par SMS × `valor_cliente` cai para 0,71 com o log, e os dois ficam: o `log1p` desfez boa parte de uma redundância que parecia forte na análise exploratória. A UCI descreve `valor_cliente` só como "the calculated value of customer", sem fórmula, então não há base para tratá-lo como derivado dos outros atributos.
  - A versão sem redundantes é montada na modelagem, selecionando colunas, sem arquivo separado.
- **Desbalanceamento**: tratado na modelagem, com pesos de classe "balanced" (ver "Tratamento do desbalanceamento").

## Onde cada transformação é feita

Transformações que tratam cada linha isoladamente são feitas uma vez na base inteira, antes da validação cruzada. Não há vazamento, porque o resultado de uma linha não depende das outras. As que aprendem parâmetros com os dados são ajustadas dentro de cada fold, só no treino.

| etapa | onde | por quê |
|---|---|---|
| remover duplicatas | base inteira | evita que a mesma linha caia no treino e no teste |
| remover `Age Group` | base inteira | não aprende nada com os dados |
| renomear colunas | base inteira | não aprende nada com os dados |
| recodificar `Tariff Plan` e `Status` para 0/1 | base inteira | mapeamento fixo |
| one-hot da idade | base inteira | as 5 categorias são conhecidas de antemão |
| `log1p` | base inteira | função fixa, aplicada a cada valor |
| padronização (`StandardScaler`) | **dentro do fold** | aprende média e desvio. O mesmo `ColumnTransformer` serve à MLP e à regressão logística, mas os parâmetros mudam a cada fold. Padronizar a base inteira usaria a média e o desvio do fold de teste para transformar o treino |
| pesos de classe / priors | **dentro do fold** | dependem da proporção de churn no treino |

## Arquivos gerados

O notebook `src/pre-processamento.ipynb` lê `data/raw/raw-customer-churn.csv` e grava em `data/processed/`:
- `churn-com-status.csv`: base pré-processada completa (2850 linhas);
- `churn-sem-status.csv`: a mesma base sem a coluna `inativo`, para medir o vazamento.

Os dois arquivos já têm o `log1p` e o one-hot e estão sem padronização, que fica no pipeline da modelagem.

## Resultado do pré-processamento

O notebook roda do início ao fim no ambiente `src/.venv`, com `assert`s em cada etapa, e todos passaram. Os histogramas antes e depois do log estão em `data/results/pre-processamento/log1p-antes-depois.png`.

| etapa | linhas | colunas |
|---|---|---|
| base bruta | 3150 | 14 |
| sem duplicatas | 2850 | 14 |
| sem `Age Group` | 2850 | 13 |
| one-hot da idade | 2850 | 17 |
| base final (com `inativo`) | 2850 | 17 |
| base final sem `inativo` | 2850 | 16 |

Conferências:
- Churn de 15,71% para 15,65% após remover as duplicatas (446 churns).
- `Age Group` e `Age`: correspondência 1:1 confirmada nos dois sentidos antes da remoção.
- Recodificação: 229 clientes no plano contratual e 684 inativos, as mesmas contagens de antes da troca de 1/2 para 0/1.
- One-hot: 112 / 921 / 1296 / 367 / 154 clientes nas faixas de 15, 25, 30, 45 e 55 anos, cada cliente em exatamente uma faixa.
- Os CSVs gravados foram relidos e são iguais à base em memória.

Colunas da base final, como referência para o `ColumnTransformer`:
- padronizar (8 numéricas): `falhas_ligacao`, `meses_assinatura`, `faixa_cobranca`, `segundos_uso`, `qtd_ligacoes`, `qtd_sms`, `numeros_distintos`, `valor_cliente`;
- passar direto (0/1): `reclamacao`, `plano_contratual`, `inativo`, `idade_15`, `idade_25`, `idade_30`, `idade_45`, `idade_55`;
- classe: `churn`;
- NB sem redundantes: todas as colunas acima, exceto `segundos_uso`.

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

A maior parte dos clientes fica perto da normal depois do log, mas os 107 clientes sem uso continuam em `log1p(0) = 0`, muito abaixo dos demais (em `segundos_uso`, quase todos ficam entre 6 e 10). Esses zeros viram valores extremos à esquerda e produzem a assimetria negativa. Os 107 têm zero ao mesmo tempo em `segundos_uso`, `qtd_ligacoes`, `numeros_distintos` e `falhas_ligacao` (79% deles também em `qtd_sms` e `valor_cliente`), e 52,3% saíram. Depois da padronização, ficam em torno de −4 desvios nas colunas de uso, valor alto mas aceitável para a MLP e a regressão logística.

Um indicador `sem_uso` repetiria o que já está em 4 colunas, por isso não foi criado:
- XGBoost: sem ganho. Um único corte (`segundos_uso ≤ 0`) já isola o grupo, e o indicador só dividiria a importância com as colunas de uso.
- Regressão logística: o único caso com algum ganho. O indicador daria ao grupo um deslocamento próprio no intercepto, e o coeficiente de uso mediria só quem usa. Como o grupo já está no extremo da escala, a reta "mais uso, menos churn" já o marca como alto risco, e o ganho esperado é pequeno.
- MLP: pouco ou nenhum ganho, porque as ativações não lineares aprendem o limiar "uso = 0".
- Naive Bayes: pioraria. O NB já conta a evidência "sem uso" pelo menos 4 vezes, ao tratar como independentes colunas que são zero juntas, e o indicador não corrigiria as gaussianas deformadas pelos zeros.

No relatório, a tabela e a figura mostram que o grupo sem uso fica isolado depois do log, já está representado em 4 colunas e é tratado de forma diferente por cada modelo.

# Modelagem

## Algoritmos

- Vistos em aula: MLP e Naive Bayes.
- Não vistos em aula: XGBoost e regressão logística, descritos em detalhe e com pseudocódigo nos seus notebooks.

## Métricas

A classe churn é a positiva. As fórmulas e a matriz de confusão de cada classificador estão em `src/comparacao.ipynb` e vão para o relatório (exigência do enunciado).

- **Recall (classe churn)**: métrica principal. Cada falso negativo é um cliente que sai sem ação de retenção.
- **Especificidade**: fração dos clientes que ficam reconhecidos como tal.
- **F1 macro**: resumo do desempenho geral, com peso igual para as duas classes, para que a classe churn não seja abafada pela majoritária.
- **Entropia cruzada (log loss)**: mede se os modelos erram com alta confiança.
- **PR-AUC**: capacidade de pôr os churns no topo da lista, independentemente do limiar. Foca na classe positiva e é informativa em bases desbalanceadas; a linha de base é a taxa de churn (0,156).

Descartadas por causa do desbalanceamento: acurácia; F1 weighted, que favorece a classe majoritária; acurácia balanceada, que é a média de recall e especificidade, já analisados separadamente; e ROC-AUC, porque, com muito mais não churn que churn, a taxa de falsos positivos fica baixa mesmo com muitos falsos alarmes e a curva fica otimista. A precisão não é analisada sozinha (entra no F1 e na PR-AUC).

## Estratégia de avaliação

Validação cruzada estratificada com 10 folds, gerada uma única vez por `src/folds.ipynb` com `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` e gravada em `data/processed/folds.csv` (coluna `fold`, de 0 a 9, com o fold de teste de cada cliente, na ordem das linhas dos CSVs).

- Cada fold de teste tem 285 clientes e 44 ou 45 churns (15,4% a 15,8%).
- Todos os algoritmos, configurações e as duas bases usam os mesmos folds. O `StratifiedKFold` estratifica só pelo `y`, e as duas bases têm as mesmas 2850 linhas, na mesma ordem e com o mesmo `churn`. Por isso as comparações são pareadas fold a fold.
- `carregar_folds` relê o arquivo e confere o tamanho, a estratificação e a igualdade com os folds gerados pela semente, para acusar na hora se uma versão futura do scikit-learn mudar o embaralhamento.
- O cliente é classificado como churn quando a probabilidade prevista é ≥ 0,5.

## Código comum (`src/comum.py`)

Um notebook por algoritmo (`src/modelagem-mlp.ipynb`, `-nb`, `-rl`, `-xgb`) e um de comparação, todos lendo os CSVs de `data/processed/` sem refazer nenhuma transformação de base inteira. O módulo `src/comum.py` concentra o que precisa ser igual em todos:
- listas de colunas e leitura das bases (`carregar_base`), folds (`gerar_folds`, `carregar_folds`) e idade numérica (`idade_numerica`);
- `criar_preprocessador`: `ColumnTransformer` com `StandardScaler` nas 8 numéricas e `passthrough` nas binárias e one-hot, usado nos pipelines da MLP e da regressão logística (na base sem `inativo`, a coluna sai da lista);
- `avaliar`: percorre os 10 folds, cria um modelo novo, ajusta só no treino e guarda, por fold, recall, especificidade, F1 macro, log loss, PR-AUC e matriz de confusão. Devolve a média e o desvio, a matriz somada, a probabilidade de churn de cada cliente prevista fora do fold (para as curvas PR) e os modelos de cada fold (para coeficientes e importâncias). O modelo é criado por uma função que recebe o `y` do treino (para o `scale_pos_weight` do XGBoost), e uma segunda função opcional devolve os argumentos do `fit` (como o `mlp__sample_weight` da MLP);
- `salvar_resultados`, `registrar_escolhida` e `ler_resultados`: gravam e leem, em `data/results/modelagem/`, as métricas por fold, a matriz somada, as probabilidades fora do fold e a configuração escolhida de cada algoritmo;
- `tabela_desbalanceamento` e `diferenca_pareada`: comparações fold a fold;
- `salvar_tabela`: grava uma tabela exibida no notebook como PNG (`<alg>-tab-*.png`), para o relatório.

## Tratamento do desbalanceamento

Todos os algoritmos usam o critério "balanced", com peso inversamente proporcional à frequência da classe, calculado em cada fold só com o `y` do treino:
- regressão logística: `class_weight='balanced'`;
- XGBoost: `scale_pos_weight` = nº de não churn ÷ nº de churn no treino (cerca de 5,4);
- MLP: o `MLPClassifier` não tem `class_weight`, então os pesos vão exemplo a exemplo em `sample_weight` no `fit` (`compute_sample_weight('balanced', y_treino)`). O scikit-learn 1.9.1 do ambiente aceita `sample_weight` no `MLPClassifier`;
- Naive Bayes: priors uniformes (`priors=[0.5, 0.5]`) nas configurações principais.

O NB não tem pesos de classe, e o equivalente são as probabilidades a priori. Com os priors da base (0,84 / 0,16), o modelo parte da crença de que um cliente tem cerca de 5,4 vezes mais chance de ficar que de sair, e a evidência dos atributos precisa superar essa vantagem para apontar churn. Pelo teorema de Bayes, a chance a posteriori de churn é a razão de verossimilhança multiplicada pela razão dos priors. Trocar 0,16/0,84 por 0,5/0,5 multiplica a chance de todos os clientes por cerca de 5,4, o que equivale a baixar o limiar de decisão. Os priors não mudam os dados: os folds continuam com cerca de 15,6% de churn e a avaliação é feita na distribuição real.

Para medir o efeito, cada algoritmo tem uma configuração sem tratamento, comparada fold a fold à versão com tratamento (sufixo `_sem-pesos`, ou `priors-base` no NB):
- MLP: `relu_adam_32` sem `sample_weight`;
- XGBoost: a configuração escolhida, com `scale_pos_weight` = 1;
- regressão logística: a configuração escolhida, sem `class_weight`;
- Naive Bayes: `priors-base` (a `completo` com os priors estimados no treino).

O esperado era que, sem o tratamento, o recall caísse, a especificidade subisse e o log loss melhorasse, com pouca mudança na PR-AUC. No NB a PR-AUC deveria ser idêntica, porque multiplicar a chance de todos pelo mesmo fator não muda a ordem dos clientes. O resultado está na conclusão.

## Configurações por algoritmo

- **Regressão logística** (padronização e one-hot da idade), em `src/modelagem-rl.ipynb`:
  - fixos: `penalty='l2'`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=1000`. A penalidade é indispensável, porque as 5 colunas one-hot somam 1, como o intercepto;
  - opção 1: `C` 1 × 0,1;
  - opção 2: grade `C` ∈ {0,001; 0,01; 0,1; 1; 10; 100}, 6 configurações × 2 bases, que contém a opção 1;
  - coeficientes (média ± desvio nos 10 folds e odds ratio) para conferir H1, H2 e H6, e a estabilidade do par `segundos_uso` × `qtd_ligacoes` em função de `C`.
- **Naive Bayes gaussiano** (sem padronização, idade numérica, `var_smoothing=1e-9`), em `src/modelagem-nb.ipynb`, com quatro configurações nas duas bases (8 rodadas):
  - `completo`: todos os atributos, com `log1p`;
  - `sem-redundantes`: sem `segundos_uso`, para medir o efeito da redundância na suposição de independência;
  - `sem-log1p`: atributos de uso de volta à escala original (`expm1`), para medir o efeito do log na suposição de normalidade;
  - `priors-base`: a `completo` com os priors estimados no treino (padrão do `GaussianNB`), que é a variação de parâmetro do modelo e a medida do efeito do desbalanceamento.
  - O `var_smoothing` não é variado: ele é proporcional à maior variância entre os atributos e, sem padronização, suavizaria mais os atributos de variância pequena.
- **XGBoost** (sem padronização, idade numérica), em `src/modelagem-xgb.ipynb`:
  - fixos: `n_estimators=300`, sem early stopping, `tree_method='hist'`, `random_state=42`; os demais nos padrões (`reg_lambda=1`, `gamma=0`, `min_child_weight=1`, sem subamostragem);
  - opção A: `max_depth` 3 × 6, com `learning_rate=0,1`;
  - opção B: grade `max_depth` ∈ {2, 3, 4, 6} × `learning_rate` ∈ {0,05; 0,1; 0,3}, 12 configurações × 2 bases, que contém a opção A;
  - importância dos atributos por `gain`, normalizada, com média ± desvio nos 10 folds, para conferir H1, H2 e H6.
- **MLP** (padronização), em `src/modelagem-mlp.ipynb`. A pergunta central é se uma camada oculta basta.
  - 7 configurações × 2 bases, variando um fator por vez a partir de ReLU + Adam com 32 neurônios por camada: profundidade (`relu_adam` com `(32,)`, `(32,32)` e `(32,32,32)`), ativação (`tanh_adam` com `(32,)` e `(32,32)`) e otimizador (`relu_sgd` com `(32,)` e `(32,32)`).
  - A grade inicial, de 13 arquiteturas × 2 ativações × 2 otimizadores (104 rodadas), foi reduzida: as redes afuniladas e expandidas responderiam se o formato importa, pergunta fora do foco.
  - A ativação é a mesma em todas as camadas ocultas. O `MLPClassifier` não permite uma ativação por camada, e a ideia de ReLU na 1ª camada e tanh nas demais foi descartada.
  - Otimizadores: `adam` com os padrões (lr 0,001, β1 0,9, β2 0,999) ou `sgd` puro (lr 0,001 constante, sem momento).
  - Fixos: 200 épocas (`max_iter=200`), sem `early_stopping`, `alpha=1e-4`, `batch_size=200`, `random_state=42`. O `tol` padrão só encerra o treino antes de 200 épocas se a perda de treino parar de cair, e os folds que chegam a 200 épocas sem convergir são registrados.

## Escolha das configurações

A configuração que representa cada algoritmo é a de maior PR-AUC média na base sem `inativo`: a PR-AUC independe do limiar e não premia quem chama todos de churn, e essa base não tem a suspeita de vazamento. Pela parcimônia, entre as configurações a menos de um desvio da melhor, fica a mais simples:
- MLP: 1 camada, salvo se a rede mais profunda ganhar mais que o desvio da sua PR-AUC (comparação pareada fold a fold);
- XGBoost: menor `max_depth` e, empatando, maior `learning_rate`;
- regressão logística: menor `C`;
- NB: menos atributos, entre `completo`, `sem-redundantes` e `sem-log1p`.

Nas grades da regressão logística e do XGBoost, a melhor configuração também é comparada fold a fold com as da opção 1 e da opção A. Cada notebook grava a sua escolha com `registrar_escolhida` (`<alg>-escolhida.txt`).

A escolha e a avaliação usam os mesmos folds, sem validação cruzada aninhada. Os números das configurações escolhidas são levemente otimistas, mais nas grades maiores (XGBoost e regressão logística), e o relatório deve dizer isso.

## Comparação e saídas

`src/comparacao.ipynb` lê os resultados gravados pelos quatro notebooks, sem treinar nada, e roda depois deles. Ele reúne:
- a estratégia de avaliação e as fórmulas das medidas;
- a tabela comparativa, com as regras de referência e o classificador trivial;
- a posição de cada algoritmo em cada medida e a comparação pareada entre eles;
- matrizes de confusão e curvas precisão-recall;
- a sobreposição dos erros (Jaccard) e o perfil dos clientes que todos erram;
- os efeitos consolidados do desbalanceamento e do `inativo`;
- a discussão com as perguntas do enunciado.

Figuras, CSVs e as tabelas em PNG (`*-tab-*.png`) ficam em `data/results/modelagem/`.

# Conclusão

Valores na base sem `inativo` (a referência), em média ± desvio nos 10 folds, com FN e FP somados nos folds. A discussão completa está em `src/comparacao.ipynb` e as tabelas em `data/results/modelagem/` (`comparacao-tab-*.png`).

| | configuração | recall | especificidade | F1 macro | log loss | PR-AUC | FN | FP |
|---|---|---|---|---|---|---|---|---|
| XGBoost | `d2_lr0.3` | 0,914 ± 0,059 | 0,958 ± 0,012 | 0,913 ± 0,026 | 0,129 ± 0,037 | 0,932 ± 0,037 | 38 | 100 |
| MLP | `relu_adam_32-32-32` | 0,914 ± 0,054 | 0,956 ± 0,009 | 0,910 ± 0,020 | 0,159 ± 0,054 | 0,932 ± 0,035 | 38 | 105 |
| Regressão logística | `C0.1` | 0,840 ± 0,082 | 0,855 ± 0,023 | 0,774 ± 0,033 | 0,351 ± 0,032 | 0,748 ± 0,082 | 71 | 348 |
| Naive Bayes | `sem-redundantes` | 0,791 ± 0,069 | 0,802 ± 0,016 | 0,712 ± 0,027 | 0,959 ± 0,248 | 0,710 ± 0,096 | 93 | 476 |
| regra "inativo → churn" | | 0,729 | 0,851 | | | | | |
| regra "reclamou → churn" | | 0,426 | 0,983 | | | | | |
| trivial "sempre não churn" | | 0 | 1 | | | 0,156 | 446 | 0 |

## Qual algoritmo apresentou o melhor desempenho? A resposta depende da medida?

XGBoost e MLP empatam no topo. Eles têm o mesmo recall (0,914) e a mesma PR-AUC (0,932), e na comparação pareada a diferença MLP − XGBoost é de −0,001 ± 0,022 na PR-AUC (o XGBoost vence em só 4 dos 10 folds), −0,003 ± 0,020 no F1 macro e +0,030 ± 0,039 no log loss, tudo dentro da variação entre folds. Na base com `inativo`, o quadro se repete (PR-AUC 0,940 contra 0,936). A regressão logística e o Naive Bayes perdem para o XGBoost nos 10 folds, por −0,185 ± 0,063 e −0,222 ± 0,077 de PR-AUC.

Pelas medidas que avaliam o modelo inteiro (PR-AUC, F1 macro e log loss), a ordem é a mesma nas duas bases: XGBoost ≈ MLP > regressão logística > Naive Bayes. Com uma medida isolada no limiar 0,5, a resposta muda:
- XGBoost e MLP trocam de posição conforme a medida, sempre por diferenças menores que o desvio;
- no recall, a regressão logística passa o NB sem `inativo` (0,840 contra 0,791), e o NB passa a regressão logística com `inativo` (0,892 contra 0,865);
- o maior recall entre as configurações salvas é da MLP de 1 camada (0,944, com especificidade de 0,907), e o NB sem `log1p` chega a 0,932 de recall com especificidade de 0,602. A maior especificidade é a do classificador trivial (1,000).

Recall e especificidade sozinhos descrevem o ponto de operação, que o limiar e os pesos de classe deslocam. Por isso a escolha usou a PR-AUC.

## A alteração dos parâmetros modificou os resultados?

Sim, sempre que a mudança levava o modelo ao subajuste. Fora dessa faixa, os resultados formam um patamar.

Mudanças com efeito grande:
- otimizador da MLP: com SGD puro, a PR-AUC cai para 0,562 (1 camada) e 0,486 (2 camadas), contra 0,893 e 0,929 com Adam. O SGD sem momento não termina de treinar em 200 épocas (a perda de treino vai só de 0,66 a 0,50);
- profundidade da MLP: de 1 para 2 camadas, a PR-AUC sobe de 0,893 para 0,929, e com 3 camadas os falsos alarmes caem de 224 para 105;
- regularização da regressão logística: com `C` = 0,01 e 0,001, a PR-AUC cai para 0,614 e 0,501;
- taxa × profundidade no XGBoost: a PR-AUC vai de 0,856 (`d2_lr0.05`) a 0,940 (`d4_lr0.1`);
- `log1p` no NB: sem ele, a especificidade cai de 0,836 para 0,602 e o F1 macro perde 0,135.

Mudanças dentro da variação entre folds: ReLU × tanh (no máximo 0,019 de PR-AUC), `C` de 0,1 a 100 (PR-AUC de 0,748 a 0,765), o XGBoost a partir de `d3_lr0.1` (9 das 12 configurações entre 0,927 e 0,940) e o NB sem `segundos_uso` (+0,014 ± 0,033).

Os pesos de classe e os priors mexem pouco na separação e muito no ponto de operação (ver o desbalanceamento, abaixo).

## Os algoritmos apresentaram comportamentos diferentes em relação aos erros de classificação?

Sim, na quantidade de erros e nos clientes em que cada um erra. Com o tratamento do desbalanceamento, todos dão mais falsos alarmes que churns perdidos. No limiar 0,5, a precisão é de 0,803 no XGBoost (408 acertos em 508 alertas), 0,795 na MLP, 0,519 na regressão logística e 0,426 no NB.

Na sobreposição dos erros (índice de Jaccard, base sem `inativo`), NB e regressão logística erram nos mesmos clientes com mais frequência (0,44 nos FN e 0,48 nos FP). MLP e XGBoost, com métricas quase iguais, dividem só cerca de um terço dos erros (0,29 nos FN e 0,34 nos FP), e os demais pares ficam entre 0,13 e 0,28. Como os dois melhores erram em clientes diferentes, combiná-los poderia reduzir os erros; isso não foi testado.

Os erros comuns aos quatro algoritmos são poucos e têm perfil claro:
- 43 dos 2404 clientes que ficam são falsos alarmes de todos. Desses, 51,2% não têm nenhum uso (contra 1,2% dos demais não churns) e 72,1% são inativos (contra 13,9%), com mediana de 0 ligações. Entre os clientes sem uso, 52,3% cancelam, e os atributos não distinguem a metade que fica.
- 9 dos 446 churns são perdidos por todos. Eles têm mediana de 82 ligações (contra 27 dos demais churns), só 11,1% reclamaram (contra 43,2%) e 22,2% estão inativos (contra 73,9%): é o perfil de quem fica.

Esses erros vêm dos dados, por ruído de rótulo ou falta de informação nos atributos, e nenhum dos algoritmos os resolve.

## O desbalanceamento das classes influenciou a avaliação?

Sim. A base tem 15,6% de churn (446 de 2850), e isso pesou em dois pontos.

Na escolha das medidas: o classificador que sempre responde "não churn" teria 84,4% de acurácia com recall zero. Por isso a acurácia e a ROC-AUC ficaram de fora, e a PR-AUC foi lida contra a sua linha de base, 0,156.

No treino, a comparação com e sem o tratamento (diferença sem − com, base sem `inativo`) mostra:

| algoritmo | recall | especificidade | PR-AUC | log loss | FN (com → sem) |
|---|---|---|---|---|---|
| Regressão logística | −0,327 | +0,118 | +0,010 | −0,114 | 71 → 217 |
| MLP (`relu_adam_32`) | −0,264 | +0,072 | −0,019 | −0,042 | 25 → 143 |
| XGBoost | −0,119 | +0,020 | −0,001 | −0,015 | 38 → 91 |
| Naive Bayes (`completo`) | −0,074 | +0,064 | 0,000 | −0,153 | 101 → 134 |

Sem tratamento, os modelos apontam menos clientes como churn: o recall cai, a especificidade sobe, e a PR-AUC muda no máximo 0,019. No NB a PR-AUC ficou idêntica nos 10 folds, como previsto. O desbalanceamento desloca o ponto de operação e quase não altera a capacidade de separar as classes. O log loss melhora sem os pesos porque as probabilidades voltam a refletir a proporção real de churn; com os pesos, elas superestimam o churn, o que não atrapalha a ordenação dos clientes. O F1 macro quase não percebe a troca (de −0,012 a +0,036), embora ela custe de 33 a 146 churns perdidos a mais, e por isso o recall é a métrica principal.

## Quais características da base de dados podem explicar as diferenças observadas?

- **Reclamação forte, mas incompleta**: 82,6% de quem reclamou sai, e a reclamação é o atributo mais importante em todos os modelos (odds ratio de 15,2 na regressão logística e 41,5% do ganho no XGBoost). Como 57% dos churns nunca reclamaram, os modelos precisam combinar a reclamação com o uso, e todos encontram muito mais churns que a regra "reclamou → churn".
- **Não linearidades e interações**: churn quase nulo nos extremos de idade, faixa de cobrança, falhas que pesam em relação ao volume de ligações e o limiar de uso zero. O modelo linear perde 0,185 de PR-AUC para o XGBoost. Árvores de profundidade 2, que combinam pares de atributos, já bastam, e a MLP melhora ao passar de 1 para 2 camadas. Mesmo linear, a regressão logística recupera parte desses efeitos: as faixas de idade extremas têm coeficientes negativos, e o par falhas (+1,19) e ligações (−1,54), em log, aproxima a taxa de falha de H4.
- **Atributos que violam as suposições do NB**: os atributos de uso são correlacionados entre si, há binários e uma idade de 5 valores tratados como gaussianos, e há muitos zeros. O NB conta a mesma evidência várias vezes, e o seu log loss (0,959) fica acima do de um modelo que prevê 15,6% para todos (0,433). Tirar `segundos_uso` não resolve, porque os outros quatro atributos de uso continuam correlacionados; o `log1p` é a transformação que mais ajuda o NB.
- **Clientes sem uso**: os 107 clientes sem nenhum uso se dividem quase ao meio entre churn e não churn e são metade dos 43 falsos alarmes comuns aos quatro algoritmos.
- **`inativo`**: efeito pequeno, exceto no NB. Na comparação pareada com − sem, a PR-AUC muda +0,004 na MLP, +0,008 no XGBoost, −0,000 na regressão logística e +0,018 no NB, que também ganha 0,101 de recall. No XGBoost, `inativo` fica com 47,6% do ganho e tira importância da reclamação (de 41,5% para 23,2%) e de `valor_cliente` (de 18,0% para 6,6%). O atributo repete informação já presente nas outras colunas, e a suspeita de vazamento não se traduziu em resultados inflados.
- **Base pequena e ruidosa**: são 446 churns e 14 combinações de atributos com rótulos conflitantes. O desvio da PR-AUC entre folds vai de 0,035 a 0,1, então várias diferenças (MLP × XGBoost, valores de `C`, ativações) ficam dentro da variação.

## Qual modelo você recomendaria para o problema investigado?

O XGBoost `d2_lr0.3` (300 árvores de profundidade 2, taxa 0,3, `scale_pos_weight`), treinado sem `inativo`.

- **Desempenho**: empata com a MLP em PR-AUC e recall (0,932 e 0,914), com menos falsos alarmes (100) e o menor log loss (0,129). No limiar 0,5, encontra 408 dos 446 churns, e 80% dos alertas estão corretos.
- **Estabilidade e custo**: são árvores rasas, sem padronização, treinadas em segundos, e 9 das 12 configurações da grade dão PR-AUC entre 0,927 e 0,940. A MLP equivalente precisa de 3 camadas, não converge em 200 épocas, despenca com outro otimizador (PR-AUC 0,56 com SGD) e perde mais recall sem pesos de classe (−0,264, contra −0,119 do XGBoost).
- **Interpretação**: a importância por `gain` confirma as hipóteses (reclamação, valor do cliente e uso), o que ajuda a explicar à administração quem será abordado e por quê. Os coeficientes da regressão logística podem complementar a explicação (reclamar multiplica a chance de churn por 15), mas como classificador ela daria 3,5 vezes mais falsos alarmes (348 contra 100) e perderia 33 churns a mais.
- **Sem `inativo`**: o status pode não estar disponível com 3 meses de antecedência e pode antecipar o próprio rótulo. Deixá-lo de fora custa só 0,008 de PR-AUC.
- **Ponto de operação**: o limiar 0,5 com pesos de classe dá recall de 0,91. Pela curva precisão-recall, a precisão fica perto de 0,8 até um recall de cerca de 0,9 e cai rápido depois disso. Se a ação de retenção for barata, o limiar pode descer um pouco; se for cara, subir.

Em desempenho, XGBoost e MLP são equivalentes, e a recomendação se apoia no custo, na estabilidade e na interpretação. Os números das configurações escolhidas são levemente otimistas, porque a escolha e a avaliação usaram os mesmos folds.
