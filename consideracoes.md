# exploracao dos dados

## conhecendo o dataset

dados de uma companhia de telefonia Iraniana em um período de 12 meses - https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset?utm_source=chatgpt.com

3150 linhas sendo cada instancia um cliente que possui:
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

temos 13 features que sao atributos preditores. porém, como idade e grupo de idade carregam exatamente a mesma informação, na prática temos 12 preditores com informação distinta.

obs: no CSV, alguns nomes de colunas possuem espaços duplos ("Call  Failure", "Subscription  Length", "Charge  Amount"), o que deve ser considerado no código.

dados de clientes estao anonimizados, ou seja, nao há PII (dados que possam identificar a identidade do cliente) no dataset.

o dataset nao possui missing values, logo, nenhuma técnica nesse sentido será necessária ou utilizada. no entanto, há 300 linhas exatamente duplicadas (restando 2850 linhas únicas).

inconsistências observadas:
- charge amount vai de 0 a 10 nos dados, mas a documentação da UCI indica 0 a 9.
- idade e grupo de idade são redundantes (correspondência 1:1).
- status pode causar vazamento de informação (leakage), pois é quase uma prévia do churn.

os dados de uso (ligações, segundos, sms, números distintos, falhas etc.) são referentes aos primeiros 9 meses. o atributo de meses de assinatura, por sua vez, representa o tempo total de casa do cliente e não está limitado a essa janela. o rótulo de churn, que é o que queremos predizer, é o estado do cliente no final de um ano (12 meses).

o gap de 3 meses entre os dados do dataset e a previsao do churn é o intervalo de planejamento.

o que é churn: métrica que simboliza a evasão de clientes de uma empresa. um cliente churn é aquele que rompeu vínculo com a empresa, e um não churn é aquele que permanece ativo.

### tarefa

**classificação**: no escopo do nosso trabalho, é interessante predizer se um cliente continuará sendo um cliente no final de 1 ano ou não. isso é a classe churn, sendo 1 representando que o cliente cancelou e 0 que o cliente ficou.

### hipóteses

as hipóteses abaixo guiam a análise exploratória e ajudam a explicar o desempenho dos classificadores.

- **H1**: há alguma feature singular decisiva para o churn? 
- **H2**: o quanto ter tido um problema anterior com a empresa (complaint = 1) influencia o churn? 
- **H3**: há alguma relação entre o número de ligações falhadas e a reclamação? 
- **H4**: se H3 se confirmar, como a tupla (falhas, reclamação) se relaciona com o churn? 
- **H5**: o plano de tarifa é decisivo para o churn? 
- **H6**: clientes que cancelam usam o serviço com frequência ou não? (isso pode ser útil para identificar se a empresa possui problemas em seu pós venda ou no suporte do cliente) 

todas essas perguntas podem ser relevantes para a administração da empresa no escopo de expansão do negócio e quality assurance do serviço prestado ao cliente. além disso, reter um cliente costuma custar menos do que conquistar um novo, o que torna a previsão de churn valiosa para ações de retenção.

## análise exploratória 

vamos analisar a distribuição das classes, tentando observar principalmente se há desbalanceamento.

a análise completa (tabelas, gráficos e respostas às hipóteses) está em `src/analise-exploratoria.ipynb`.

### distribuição das classes

2655 não churn (84,3%) e 495 churn (15,7%), razão de 5,36 : 1. a base é desbalanceada: um modelo que sempre responde "não churn" já teria ~84% de acurácia sem acertar nenhum churn.

### respostas às hipóteses

números da base sem duplicatas (2850 clientes, 446 churn).

- **H1 (feature singular decisiva)**: não há. `Complains` (r = 0,53) e `Status` (r = 0,50) são os atributos mais fortes, mas nenhum resolve sozinho:
  - a regra "reclamou → churn" tem recall de 0,426 e especificidade de 0,983: quase todo cliente que reclama sai, mas 57% dos que saem nunca reclamaram.
  - a regra "inativo → churn" tem recall de 0,729 e especificidade de 0,851, e ainda é suspeita de vazamento.
  - essas regras servem de referência mínima para os modelos.
- **H2 (reclamação)**: confirmada e forte. quem reclamou sai em 82,6% dos casos, contra 9,8% de quem não reclamou (≈ 8 vezes mais). o efeito não depende do status: entre os ativos, 70,4% de quem reclamou saiu, contra 3,1% de quem não reclamou.
- **H3 (falhas × reclamação)**: a relação existe, mas é fraca (r = 0,15; com a taxa de falha, 0,16). quem reclamou tem em média 11,5 falhas contra 7,5, e taxa de falha de 0,23 contra 0,13. as falhas explicam só uma parte das reclamações.
- **H4 (falhas, reclamação) × churn**:
  - o número absoluto de falhas não separa o churn (14,5% a 18,1% em todas as faixas). dentro de cada grupo de reclamação, o churn até cai conforme as falhas aumentam, porque quem usa mais acumula mais falhas e sai menos (H6).
  - a **taxa de falha** (falhas ÷ ligações) corrige o efeito do volume: sem reclamação, o churn vai de 2,7% (taxa ≤ 0,1) a 21,6% (taxa > 0,2); com reclamação, chega a 89,1%.
  - a reclamação domina, e a taxa de falha alta agrava o risco.
- **H5 (plano de tarifa)**: não é decisivo sozinho. o contratual tem 2,6% de churn contra 16,8% no pré-pago, mas reúne só 8% dos clientes; quase todos os churns estão no pré-pago.
- **H6 (uso)**: quem cancela usa muito menos o serviço. as medianas dos clientes churn ficam entre 35% e 46% das de quem fica (29 contra 64 ligações, 1.256 contra 3.596 segundos, 11 contra 26 SMS), com o mesmo tempo de assinatura (35 meses). entre os 107 clientes sem nenhum uso, 52,3% saíram, contra 14,2% dos que usam. a queda de uso parece anteceder o cancelamento, o que, junto de H2, aponta para problemas de pós-venda e suporte.

### o que fica para os algoritmos

a análise exploratória mostra relações de um ou dois atributos por vez. os classificadores devem responder:

- **combinação de atributos**: se reclamação, status, uso e cobrança juntos superam as regras de um atributo só (recall de 0,73 da regra do `Status`).
- **peso do `Status`**: comparar o desempenho com e sem o atributo para medir o quanto ele infla os resultados (vazamento).
- **relações não lineares**: a idade (churn quase zero nos grupos extremos), a faixa de cobrança e a interação falhas × reclamação não são lineares. espera-se que XGBoost e MLP capturem isso melhor que a regressão logística e o Naive Bayes.
- **redundância entre atributos** (segundos × ligações, r = 0,95; SMS × Customer Value, r = 0,92): avaliar o impacto no Naive Bayes, que supõe independência, e na estabilidade dos coeficientes da regressão logística.
- **importância dos atributos**: usar os coeficientes da regressão logística e a importância do XGBoost para confirmar (ou não) H1, H2 e H6 com todos os atributos ao mesmo tempo.
- **efeito do desbalanceamento**: comparar o recall e a especificidade com e sem os pesos de classe (ou priors uniformes).
- **limites**: o ruído de rótulo (14 combinações conflitantes) e o ruído de medição (números distintos e falhas maiores que o total de ligações) impõem um teto ao desempenho.

### decisões de pré-processamento

- **duplicatas**: remover as 300 linhas duplicadas (restam 2850). a proporção de churn quase não muda (15,71% → 15,65%), e a remoção evita que a mesma linha caia no treino e no teste.
- **rótulos conflitantes**: 14 combinações de atributos aparecem com churn 0 e 1. ficam na base como ruído de rótulo.
- **Age Group**: remover, por ser redundante com Age (correspondência 1:1, r = 0,96).
- **Charge Amount = 10**: tratar como valor válido (7 linhas; é a faixa mais alta, sem churn, como as faixas 5-9).
- **clientes sem nenhum uso** (154 na base original, 107 sem duplicatas): manter. um cliente pode nunca ter usado o serviço e continuar (ou sair), e isso é informação relevante para o churn.
- **números distintos > total de ligações (120 linhas) e falhas > total de ligações (3 linhas)**: manter. não denotam dados inválidos, mas indicam possível ruído na medição desses atributos, o que deve ser deixado claro no relatório.
- **Status**: rodar os experimentos com e sem o atributo, para avaliar o possível vazamento de informação (leakage).
- **desbalanceamento**: tratar com pesos de classe, todos no critério "balanced" (peso inversamente proporcional à frequência da classe):
  - regressão logística: `class_weight='balanced'`.
  - XGBoost: `scale_pos_weight` = nº de não churn / nº de churn no treino (~5,4).
  - MLP: o `MLPClassifier` não tem `class_weight`, então os pesos são passados exemplo a exemplo com `sample_weight` no `fit` (`compute_sample_weight('balanced', y_treino)`, calculado dentro de cada fold só com o `y` do treino). se a versão do scikit-learn não aceitar, a alternativa é sobreamostrar a classe churn só no treino de cada fold.
  - Naive Bayes: ver a seção abaixo.

#### Naive Bayes com priors 0,5 / 0,5

o Naive Bayes não tem pesos de classe; o equivalente é a **probabilidade a priori** de cada classe. por padrão ele usa as proporções da base de treino (≈ 0,84 não churn / 0,16 churn). vamos testar duas parametrizações: priors da base e priors uniformes (`priors=[0.5, 0.5]`).

**por que usar**: com os priors da base, o modelo parte da crença de que um cliente tem ~5,4 vezes mais chance de ficar do que de sair. para classificar alguém como churn, a evidência dos atributos precisa vencer essa vantagem, o que tende a gerar um recall baixo, justamente a métrica que mais nos interessa. os priors uniformes fazem a decisão depender só da evidência dos atributos, o que é o mesmo princípio dos pesos "balanced" usados nos outros algoritmos.

**como influencia o resultado**: pelo teorema de Bayes, a chance a posteriori de churn é a razão de verossimilhança multiplicada pela razão dos priors. trocar 0,16/0,84 por 0,5/0,5 multiplica essa chance por ~5,4 para **todos** os clientes. na prática, é o mesmo que baixar o limiar de decisão:
- **recall** da classe churn sobe, porque mais clientes passam a ser classificados como churn.
- **especificidade** cai, porque parte dos clientes que ficam passam a ser apontados como churn (falsos alarmes).
- **F1 macro** pode subir ou cair, conforme o balanço entre os dois efeitos.
- **log loss** tende a piorar: as probabilidades deixam de refletir a proporção real de churn (ficam descalibradas, superestimando o churn).
- **PR-AUC não muda**: multiplicar a chance de todos pelo mesmo fator não altera a **ordem** dos clientes por probabilidade, só o ponto de corte. isso deve ser comentado no relatório, porque mostra que as duas parametrizações têm a mesma capacidade de separar as classes e diferem só no ponto de operação.

**efeito na base**: nenhum. os priors não alteram os dados nem a distribuição das classes. os 10 folds continuam estratificados com ~15,6% de churn, e a avaliação é feita na distribuição real. o que muda é só a regra de decisão do modelo.

### algoritmos

- vistos em aula: **MLP** e **Naive Bayes**.
- não vistos em aula: **XGBoost** e **regressão logística** (exigem descrição detalhada com pseudocódigo).

### métricas

- **acurácia**: não será usada, por causa do desbalanceamento.
- **recall (classe churn)**: métrica principal. o interesse é saber se o modelo deixou de encontrar churns que deveriam ser encontrados (falsos negativos). falsos alarmes não são o foco, por isso a precisão não será analisada.
- **especificidade**: avalia se o modelo acerta os clientes que ficam na medida em que deveria.
- **F1 macro**: resume o desempenho geral. o peso igual entre as classes é proposital, para que a classe churn não seja abafada pela majoritária.
- **F1 weighted**: não será usado, pois favorece a classe majoritária.
- **acurácia balanceada**: não será usada. ela é a média entre recall e especificidade, que já são analisados separadamente.
- **entropia cruzada (log loss)**: avalia se os modelos estão errando o óbvio, ou seja, errando com alta confiança.
- **ROC-AUC**: não será usada. com muitos mais não churn do que churn, a taxa de falsos positivos fica baixa mesmo com muitos falsos alarmes, o que deixa a curva otimista.
- **PR-AUC**: será usada, por focar na classe positiva (churn) e ser mais informativa em bases desbalanceadas.
- a matriz de confusão de cada classificador e as fórmulas das métricas devem constar no relatório (exigência do enunciado).

### estratégia de avaliação

**validação cruzada estratificada com 10 folds**, mantendo em cada fold a proporção de churn da base (~45 churns por fold de teste). todas as configurações de parâmetros (pelo menos 2 por algoritmo) e todos os algoritmos são avaliados nos mesmos folds (mesma semente), o que torna a comparação justa. transformações que aprendem com os dados, como a padronização para MLP e regressão logística, devem ser ajustadas só no treino de cada fold (pipeline).

