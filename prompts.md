# Exploração dos Dados

## Gemini - Auto
Usuário
```
o que é churn? o que significa cliente churn e cliente não churn?
```
IA
```
**Churn** (ou *taxa de cancelamento*) é uma métrica de negócios que representa a evasão de clientes, ou seja, a quantidade de consumidores que deixam de usar um serviço, cancelam uma assinatura ou param de comprar de uma empresa dentro de um determinado período.

No contexto de análise de dados e negócios, a base de clientes costuma ser dividida nesses dois grupos:

* **Cliente Churn:** É o cliente que rompeu o vínculo com a empresa. Ele cancelou a assinatura (como na Netflix ou Spotify), encerrou a conta em um banco ou simplesmente parou de fazer pedidos em um aplicativo durante a janela de tempo analisada. Na modelagem preditiva (Machine Learning), essa costuma ser a classe positiva (`1` ou `True`), pois é o comportamento que a empresa tenta prever e evitar.
* **Cliente Não Churn (Retido):** É o cliente que permanece ativo. Ele renovou o plano, continuou utilizando a plataforma ou manteve sua frequência de compra regular dentro do período avaliado. Na modelagem, representa a classe negativa (`0` ou `False`).

### Tipos mais comuns de Churn

* **Churn Voluntário:** O próprio cliente decide sair (por insatisfação, preço alto ou troca por um concorrente).
* **Churn Involuntário:** O cancelamento ocorre sem a intenção direta do cliente, geralmente por falhas técnicas, como cartão de crédito expirado ou falta de saldo na hora da cobrança automática.
```

## Claude Desktop - Opus 5.5 low
Usuário
```
@C:\7-periodo\mineracao-dados baseado nos slides da disciplina existentes na pasta slides/ e em técnicas de ciência de dados, me relembre como realizar uma análise de distribuição de classes de um dataset, visando identificar desbalanceamento. o que você não retirar diretamente dos slides, me avise.
```
IA
```
Li os 14 PDFs em `slides/` (na pasta do Windows). Quase tudo sobre desbalanceamento está em **Aula9-MD-Avaliacao-Classificacao.pdf**. Os slides tratam o tema do ponto de vista da *avaliação*: nenhum slide ensina a fazer a análise exploratória da variável alvo em si. Por isso a parte "como medir o desbalanceamento" vem quase toda de técnicas gerais, e eu marco isso abaixo.

## 📘 O que vem dos slides

**1. Diagnosticar a distribuição** (Aula9, slide 27, passo 1): "veja a distribuição das classes (ex.: 99% × 1%)". É só isso que os slides dizem sobre o diagnóstico.

**2. Por que isso importa**
- Aula9, slide 18: 9.990 exemplos da classe 0 contra 10 da classe 1. Prever sempre 0 dá 99,9% de acurácia e nunca acerta a classe rara.
- Aula9, slide 17: a acurácia só serve quando as classes têm distribuição parecida e os custos dos erros são semelhantes.
- Aula5, slides 12 e 14: o KNN com k grande puxa a predição para a classe majoritária e é sensível a dados desbalanceados.
- Aula7, slide 16: o Gini mede a impureza de um nó a partir da distribuição das classes (a mesma ideia de proporção aplicada localmente).

**3. Consequências para a divisão dos dados**
- Aula9, slides 9 e 12: com desbalanceamento, use divisão/CV **estratificada**, com cada fold mantendo a proporção original (ex.: 70/30).
- Aula3, slide 22: amostragem estratificada garante a representação das classes.
- Aula9, slide 29: "verifique se a divisão preserva as classes".

**4. Métricas** (Aula9, slides 16 a 26): no caso binário, positivo = classe minoritária. Use precisão, recall, especificidade/FPR, F1 e ROC/AUC. A escolha depende do custo de FP e FN (slide 20). Em multiclasse, macro, ponderada e micro mudam a interpretação (slide 26).

**5. Tratamento** (Aula9, slides 27 e 28): oversampling, undersampling e SMOTE, **só no treino**. Fazer oversampling antes da CV é data leakage.

## 🔧 O que NÃO está nos slides (técnicas gerais)

- **Quantificar** em vez de só olhar: contagens e proporções, *imbalance ratio* (IR = n_maj / n_min), entropia normalizada da distribuição (1 = balanceado) e, em multiclasse, a contagem absoluta da menor classe. Ter 50 exemplos já limita a estimativa de recall, qualquer que seja a proporção.
- Não existe um limiar universal. Um IR de 1:4 pode ser tranquilo e um de 1:100 é severo. Importa também a separabilidade: overlap entre classes e pequenos disjuntos costumam pesar mais que a razão em si.
- **Conferir a distribuição em cada split/fold** depois de estratificar, e comparar treino com teste para detectar *prior shift*.
- Olhar a distribuição **condicionada** a outras variáveis (tempo, grupo ou fonte) para pegar desbalanceamento escondido ou drift.
- **Baselines**: o classificador majoritário (`DummyClassifier`) como piso. Em desbalanceamento forte, a **PR-AUC** costuma ser mais informativa que a ROC-AUC. Os slides citam só a ROC.
- Alternativas ao reamostragem: `class_weight`, ajuste do limiar de decisão.

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold


def class_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts()
    return pd.DataFrame({"count": counts, "proportion": counts / counts.sum()})


def imbalance_summary(y: pd.Series) -> dict:
    counts = y.value_counts()
    p = counts / counts.sum()
    return {
        "n_classes": len(counts),
        "minority_class": counts.idxmin(),
        "minority_count": int(counts.min()),
        "imbalance_ratio": counts.max() / counts.min(),
        "normalized_entropy": entropy(p) / np.log(len(counts)),
    }


def fold_distributions(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = {
        f"fold_{i}": y.iloc[test_idx].value_counts(normalize=True)
        for i, (_, test_idx) in enumerate(skf.split(X, y))
    }
    return pd.DataFrame(rows).T

Se quiser, aplico isso em algum dataset específico da disciplina.
```

## Claude Code - Opus 5.5 low - Sessão 1
Usuário
```
em vista do dataset em @data/raw/raw-customer-churn.csv  , leia a minha análise em @consideracoes.md e diagnostique: há algum erro? ao final da seção "conhecendo o dataset", as hipóteses de análise do dataset fazem sentido? há algo a melhorar? 

não faça modificações diretamente no arquivo sem minha autorização.
```
IA
```
Revisei o [consideracoes.md](consideracoes.md) contra o CSV (3150 linhas, 14 colunas). Não alterei nada no arquivo.

## Erros e imprecisões

1. **Charge Amount vai de 0 a 10, não de 0 a 9.** O CSV tem 11 valores distintos. A descrição da UCI diz 0–9, mas os dados não batem com ela. Vale citar essa divergência. Uma tradução possível é "faixa de gasto/cobrança".
2. **Subscription Length vai de 3 a 47 meses.** Isso contradiz a ideia de que os dados cobrem só 9 meses. A janela de 9 meses vale para os dados de uso (ligações, SMS, segundos). O tempo de assinatura é o tempo total de casa do cliente. Convém deixar isso claro na linha 32.
3. **Há 300 linhas exatamente duplicadas** (só 2850 são únicas). Você escreveu que não há missing values, e isso está certo, mas falta registrar as duplicatas e decidir o que fazer com elas. Elas inflam métricas se uma cópia cair no treino e a outra no teste.
4. **Idade não é contínua.** Ela tem só 5 valores (15, 25, 30, 45, 55) e corresponde um a um ao Age Group (1→15, 2→25, 3→30, 4→45, 5→55). É uma feature redundante. Na prática, sobram 12 preditores com informação distinta, não 13.
5. **Tariff Plan e Status não são binários 0/1.** Eles vêm codificados como 1/2. Chamar de binário está certo, mas mencione a codificação, porque isso importa para modelos como regressão logística.
6. **Os nomes das colunas no CSV têm espaços duplos** ("Call  Failure", "Subscription  Length", "Charge  Amount"). É um detalhe prático para o código.

## Sobre as hipóteses (linha 40)

Elas fazem sentido e são relevantes para o negócio. Pontos a melhorar:

- **Risco de vazamento com Status.** Entre os inativos (Status=2), 370 de 782 deram churn (47%). Entre os ativos, foram 125 de 2368 (5%). Status é quase uma prévia do churn. Discuta se ele deve entrar como preditor ou se é praticamente o próprio alvo.
- **Customer Value na regressão.** Esse valor provavelmente é calculado a partir de uso e cobrança (segundos, SMS, frequência). Predizê-lo pode ser trivial, então vale checar a correlação antes. Também há uma ambiguidade no texto: "dado um cliente com churn" restringe a regressão só aos churners? Se a ideia é essa, explicite e justifique. Se não é, retire o "com churn".
- **Torne as perguntas testáveis.** Hoje elas estão soltas num parágrafo. Sugiro listá-las como H1, H2… e indicar o método de cada uma:
  - complaint × churn: tabela de contingência + qui-quadrado
  - call failure × complaint: comparar a distribuição de falhas entre quem reclamou e quem não reclamou (Mann-Whitney)
  - falhas + reclamação → churn: interação, ou uma árvore de decisão rasa
  - "feature decisiva": importância de features e informação mútua
  - "clientes que cancelam usam pouco o serviço?": comparar Seconds/Frequency/SMS entre as classes
- **"Padrão subjetivo"** é vago. Se a ideia é encontrar perfis de clientes, chame de "perfis/segmentos" e cite clusterização ou regras de associação, que combinam com o escopo de mineração de dados.
- **Desbalanceamento.** Já adianto para a próxima seção: são 495 churn contra 2655 não-churn, ou seja, cerca de 15,7% de churn. Isso indica usar F1, recall e PR-AUC em vez de acurácia, e estratificar a divisão treino/teste.

Se quiser, aplico essas correções no arquivo depois que você autorizar.
```

Usuário
```
quero que você corrija o arquivo. mas antes, me pergunte o que você tem dúvida e após isso faça o esboço do que você pensou em corrigir.
```
IA
```
Com base nas suas respostas, este é o esboço das mudanças. Ainda não mexi no arquivo.

## Seção "conhecendo o dataset"

**Lista de atributos (linhas 9–21)**
- Item 4 (charge amount): trocar "0 a 9" por "0 a 10, embora a documentação da UCI diga 0–9". Sugerir a tradução "faixa de cobrança".
- Item 3 (meses de assinatura): anotar que vai de 3 a 47 meses e é o tempo total de casa do cliente, não uma medida da janela de 9 meses.
- Itens 10 e 11 (tariff plan, status): deixar explícito que são binários codificados como 1/2, não 0/1.
- Item 12 (idade): avisar que tem só 5 valores (15, 25, 30, 45, 55), que correspondem um a um ao grupo de idade, então é redundante.

**Linha 26 (13 features):** manter, mas acrescentar que, por causa da redundância idade/grupo de idade, são 12 preditores com informação distinta.

**Linha 30 (missing values):** manter e acrescentar que há 300 linhas exatamente duplicadas (2850 únicas). Só vou registrar, a decisão fica para o pré-processamento, com o aviso sobre o risco de a mesma linha aparecer no treino e no teste.

**Nova nota:** os nomes das colunas no CSV têm espaços duplos ("Call  Failure", "Subscription  Length", "Charge  Amount").

**Linha 32 (janela de 9 meses):** deixar claro que os 9 meses valem para os dados de uso (ligações, SMS, segundos). Subscription Length não entra nessa janela.

**Linha 40 (parágrafo de perguntas):** quebrar em duas partes.
- Tarefas:
  - classificação do churn;
  - regressão do Customer Value para todos os clientes, sem o "dado um cliente com churn". Vou incluir um aviso de que o CV provavelmente é derivado das variáveis de uso, então a regressão pode ser trivial e é preciso checar a correlação antes.
- Hipóteses, cada uma com o método sugerido:
  - **H1:** existe uma feature decisiva para o churn? → importância de features, informação mútua
  - **H2:** ter reclamado aumenta o churn? → tabela de contingência + qui-quadrado
  - **H3:** falhas de ligação se relacionam com reclamação? → comparar distribuições, Mann-Whitney
  - **H4:** como o par falhas + reclamação se relaciona com o churn? → análise de interação, árvore rasa
  - **H5:** o plano de tarifa influencia o churn? → contingência + qui-quadrado
  - **H6:** quem cancela usa pouco o serviço? → comparar segundos, frequência e SMS entre as classes. Mantenho sua observação sobre pós-venda e suporte.
  - **H7:** existem perfis de clientes que cancelam? → clusterização ou regras de associação. Esta substitui o "padrão subjetivo".

**Linha 42:** mantenho como está.

## Seção "análise exploratória"

Depois da frase que já existe, vou acrescentar:
- **Distribuição das classes:** 2655 não-churn (84,3%) e 495 churn (15,7%). É um desbalanceamento moderado, então usar F1, recall e PR-AUC em vez de acurácia, e estratificar a divisão treino/teste.
- **Status × Churn:** 47% de churn entre os inativos contra 5% entre os ativos, com a tabela 2×2. Vou marcar como possível vazamento e dizer que é preciso decidir se Status entra como preditor.

## Estilo

Mantenho seu tom (minúsculas, primeira pessoa do plural, texto direto). Só a parte de hipóteses vira lista.

Posso aplicar? Se quiser cortar ou mudar algum item, me diga antes.
```

Usuário
```
pode aplicar
```
IA
```
Apliquei o esboço no [consideracoes.md](consideracoes.md), sem desvios.

As porcentagens (84,3%/15,7% para as classes e 5,3%/47,3% para Status) estão arredondadas a partir das contagens que eu já tinha. Revise o texto: é seu trabalho e reescrevi alguns trechos.

Os números das duas tabelas novas usam as 3150 linhas, com as 300 duplicadas incluídas. Se você decidir remover as duplicatas no pré-processamento, essas contagens vão mudar e precisam ser recalculadas.
```

Usuário
```
agora, analise se essas hipóteses fazem sentido com o enunciado do trabalho final em
@enunciado-trab-final.pdf
```
IA
```
No geral, sim. O enunciado tem tema único, **classificação**, e pede só exploração dos dados, pré-processamento, 4 algoritmos com 2 configurações cada, e avaliação e comparação. As hipóteses servem bem como roteiro da análise exploratória (item 2a). Encontrei alguns desalinhamentos.

## O que está fora do escopo

1. **A tarefa de regressão do Customer Value ([consideracoes.md:43](consideracoes.md#L43)).** O trabalho é só de classificação. Eu tiraria ou deixaria como uma nota de "outras tarefas possíveis com a base, fora do escopo". Do jeito que está, parece que vocês vão fazer.
2. **A H7 (perfis/segmentos).** Responder com clusterização é aprendizado não supervisionado, que o enunciado não pede e não avalia. Ou vocês a respondem só com tabelas e gráficos na análise exploratória (por exemplo, a taxa de churn por plano × faixa etária), ou tiram.

## O que encaixa bem

- **H1** (feature decisiva) liga direto ao item 2d, que pergunta "quais características da base de dados podem explicar as diferenças observadas?". Dá para responder também com a importância de features de uma árvore ou random forest, que talvez já estejam entre os 4 algoritmos.
- **H2, H3, H5 e H6** viram os "gráficos ou tabelas que contribuam para a compreensão dos dados" do item 2a. São crosstabs e boxplots por classe, sem custo extra.
- **H4** pode justificar a escolha de um algoritmo baseado em árvore (item 2c-II, "justificar sua escolha"), porque árvores capturam interações.

Vale calibrar o esforço. O enunciado não pede testes estatísticos. Um qui-quadrado ou Mann-Whitney enriquece, mas o que pesa na nota são os algoritmos, as parametrizações, as matrizes de confusão e a tabela comparativa. Não deixem as hipóteses tomarem o tempo disso.

## O que o enunciado pede e ainda falta no arquivo

- **Relevância do problema** (2a, primeiro tópico). Existe na [linha 54](consideracoes.md#L54), mas é fraca. Um argumento comum é que reter um cliente costuma custar menos do que conquistar um novo.
- **Inconsistências** (2a). Vi que você enxugou o arquivo e saíram três pontos: a divergência 0–9 vs. 0–10 do Charge Amount, a correspondência 1:1 entre idade e grupo de idade, e Status × Churn. Os dois primeiros são exatamente as "possíveis inconsistências" que o enunciado pede. O caso do Status também é relevante para o 2b, na hora de decidir se ele entra como preditor. Se a remoção foi intencional, pense em trazer de volta pelo menos as inconsistências.
- **Distribuição das classes** (2a). A seção está aberta mas vazia. Lembrete: o 2d pergunta diretamente se o desbalanceamento influenciou a avaliação, então essa análise vai ser retomada depois.

## Observação extra

O item 4 do enunciado exige declarar o uso de LLMs com os **prompts usados** e como as respostas foram verificadas. Se não declararem, a nota é zero. Vi que você abriu um [prompts.md](prompts.md), então parece que já está registrando.

Quer que eu aplique alguma dessas mudanças no arquivo?
```

Usuário
```
sim, quero que você aplique as mudanças. me pergunte o que for necessário e aplique em seguida considerando minhas respostas. modifique apenas o @consideracoes.md . a parte da análise exploratória vou preencher em outra sessão. o que eu tirei foi porque era excesso de texto para algo que parecia óbvio.
```
IA
```
Atualizei o [consideracoes.md](consideracoes.md) com as suas respostas e não mexi na seção de análise exploratória:

- **Inconsistências:** um bloco curto de três itens logo depois da linha das duplicatas (charge amount 0–10 vs. 0–9, idade = grupo de idade, Status como possível vazamento).
- **Tarefa:** tirei a regressão e a seção passou a se chamar "tarefa", só com a classificação.
- **Hipóteses:** tirei a H7 e coloquei uma frase de abertura dizendo que elas guiam a análise exploratória e ajudam a explicar o desempenho dos classificadores.
- **Relevância:** acrescentei ao parágrafo final a frase sobre reter um cliente custar menos do que conquistar um novo.

Ficou uma linha em branco a mais entre a H6 e o parágrafo final, no lugar onde estava a H7. Não muda nada no texto.
```

## Claude Code - Opus 5.5 low - Sessão 2
Usuário
```
antes de seguir minha instrução, leia a análise do dataset em @consideracoes.md para obter contexto da tarefa. 
vamos realizar uma análise exploratória do dataset em @data\raw\raw-customer-churn.csv 
me ajude a identificar a distribuição das classes para que possamos identificar quais algoritmos interessantes para a tarefa de classificação podemos usar. com essa análise também traçaremos quais métricas iremos utilizar para analisar os resultados dos algoritmos. 
identifique se há valores ausentes, duplicatas, possíveis inconsistências e outras características relevantes que precisem ser adereçadas antes da aplicação dos algoritmos.
não tome decisões por mim sobre quais algoritmos usaremos, apenas me fale o que você pensou baseado no enunciado em @enunciado-trab-final.pdf e nos slides de aula em @C:\7-periodo\mineracao-dados\slides
```
IA
```
Coloquei a análise exploratória no fim de [src/analise-exploratoria.ipynb](src/analise-exploratoria.ipynb) e removi a célula markdown vazia. Não consegui executar o notebook: nenhum Python do sistema tem pandas instalado. Por isso conferi os números abaixo com um script separado direto no CSV. Rode as células no seu kernel para gerar gráficos e tabelas. O resultado da célula de correlação eu ainda não verifiquei; o texto ali descreve o que se espera encontrar, e você deve ajustá-lo depois de rodar.

## O que os dados mostram

**Distribuição das classes:** 2655 não churn (84,3%) contra 495 churn (15,7%), cerca de 5,4 para 1. A base é desbalanceada: um modelo que sempre responde "não churn" já acerta 84%.

**Valores ausentes:** nenhum.

**Duplicatas:**
- 300 linhas são cópias exatas, sobrando 2850 únicas. A proporção de churn quase não muda (15,6%).
- 14 combinações de atributos aparecem com rótulo 0 e também 1. É ruído que nenhum modelo consegue acertar.
- Se as duplicatas ficarem, a mesma linha pode cair no treino e no teste, e o resultado fica otimista demais.

**Inconsistências:**
- `Charge Amount = 10` em 7 linhas, mas a documentação diz que vai de 0 a 9.
- `Age` e `Age Group` têm correspondência 1:1, ou seja, a mesma informação duas vezes.
- Em 120 linhas há mais números distintos chamados do que ligações feitas, o que é impossível. Talvez a contagem inclua ligações recebidas, mas a documentação não diz. Vale citar no relatório.
- Em 3 linhas há mais ligações falhadas do que ligações feitas.
- 154 clientes não têm nenhum uso. Isso é coerente: ligações e segundos são zero nos dois casos.

**Atributos ligados ao churn:**

| Atributo | Taxa de churn |
|---|---|
| `Status` | 47% entre inativos, 5% entre ativos |
| `Complains` | 83% entre quem reclamou, 10% entre quem não reclamou |
| `Tariff Plan` | 2,4% no contratual (6 de 245) |
| `Charge Amount` ≥ 5 | 0% |
| `Age Group` 5 | 1% (2 de 170) |

- **`Status`:** é forte candidato a vazamento de informação (*leakage*), porque é medido na mesma janela dos dados de uso.
- **Escalas e caudas:** as escalas variam muito (segundos chegam a ~17 mil, falhas a 36). SMS e `Customer Value` têm cauda longa à direita.

## Algoritmos possíveis (não estou decidindo por você)

Você precisa de 4 algoritmos, sendo pelo menos 2 que não foram vistos em aula.

**Vistos em aula (slides 5 a 8):**
- **Árvore de decisão:** os dados têm limiares quase determinísticos (reclamação, plano, faixa de cobrança). A árvore aproveita isso, dispensa normalização e é interpretável, o que ajuda a responder H1, H2 e H4.
- **kNN:** exige normalização e sofre com atributos redundantes e com a classe minoritária. Isso não é motivo para descartar: os problemas dão uma boa discussão no relatório.
- **Naive Bayes:** é rápido e serve de referência (*baseline*). Mas os atributos de uso são correlacionados entre si e `Age`/`Age Group` repetem a mesma informação, o que viola a hipótese de independência. Isso também rende discussão.
- **Rede neural (MLP):** exige normalização e lida razoavelmente com a assimetria das variáveis.

**Fora da aula, adequados a dados tabulares desbalanceados:**
- **Random Forest:** extensão natural da árvore e mais robusta.
- **Gradient Boosting (XGBoost ou LightGBM):** costuma ser o mais forte nesse tipo de base.
- **SVM:** aceita `class_weight`.
- **Regressão logística:** linear e interpretável.

Para o pseudocódigo exigido no enunciado, Random Forest e regressão logística são os mais simples de descrever.

## Métricas

- Acurácia, se usada, só deve aparecer junto com as outras, nunca sozinha.
- **Precisão, revocação (*recall*) e F1 da classe churn.** Num caso de retenção, deixar de detectar quem vai sair costuma custar mais do que um falso alarme, por isso a revocação pesa.
- **AUC-PR** (área sob a curva precisão-revocação) é mais informativa que **ROC-AUC** quando há desbalanceamento.
- **Acurácia balanceada** ou **F1 macro**, para comparar os modelos de forma geral.
- O enunciado exige a matriz de confusão e as fórmulas das métricas.
- Avaliação: **validação cruzada estratificada**, feita depois de tratar as duplicatas.

## Decisões que ficam com você

1. Remover as duplicatas ou mantê-las.
2. Remover `Status`, por causa do vazamento, ou rodar com e sem ela para comparar.
3. Remover `Age` ou `Age Group`.
4. Tratar o valor 10 de `Charge Amount` como válido ou não.
5. Lidar com o desbalanceamento por `class_weight`, SMOTE ou ajuste do limiar, ou não fazer nada e analisar o efeito. Qualquer reamostragem deve ser aplicada só no treino.
```

Usuário
```
você não leu a análise em @consideracoes.md
```
IA
```
Eu li o arquivo, mas não usei o que estava nele, e o efeito é o mesmo. Errei em três coisas:

1. **Repeti o que você já tinha anotado como se fosse novidade:** as 300 duplicatas, o `Charge Amount` com valor 10, a redundância entre `Age` e `Age Group` e o vazamento pelo `Status`. Tudo isso já estava em [consideracoes.md](consideracoes.md).
2. **Não liguei os resultados às suas hipóteses H1 a H6.** Os números servem para elas:
   - **H2:** 83% de churn entre quem reclamou, contra 10% entre quem não reclamou.
   - **H5:** só 2,4% de churn no plano contratual (6 de 245).
   - **H3, H4 e H6:** o notebook ainda não tem células para elas. Faltam o cruzamento entre falhas e reclamação, o cruzamento (falhas, reclamação) × churn e uma comparação de uso entre quem sai e quem fica.
3. **Ignorei a seção "análise exploratória"**, que você deixou começada no fim do arquivo para continuar.

O que de fato é novo em relação às suas anotações:
- 14 combinações de atributos aparecem com churn 0 e 1 (ruído de rótulo).
- Depois de remover as duplicatas, sobram 2404 não churn e 446 churn (15,6% de churn).
- Em 120 linhas há mais números distintos chamados do que ligações feitas, e em 3 linhas há mais falhas do que ligações.
- Nenhum cliente com `Charge Amount` ≥ 5 deu churn.

Posso fazer as duas coisas:
- Escrever esses achados na seção "análise exploratória" de [consideracoes.md](consideracoes.md), no seu estilo, sem repetir o que já está lá.
- Adicionar ao notebook as células para H3, H4 e H6.

Quer as duas ou só uma?
```

Usuário
```
tá, rodei o código todo em @src/analise-exploratoria.ipynb . olhe os resultados e ajuste as conclusões em cada md após as células de código. 

vamos remover as duplicatas, pois como você disse, não influencia no churn. vamos rodar com esem Status para comparação. podemos remover Age Group. trate o valor 10 de Charge Amount como válido. 

os dados de clientes sem nenhum uso devem ser levados em consideração, pois o cliente pode nunca ter usado mas continuado na empresa, que é o churn. números distintos > total de ligações e falhas > total de ligações podem ficar também porque não denotam dados inválidos, apenas é um indicativo de que a medição desses dados pode ter tido algum tipo de ruído. mas isso deve ser deixado claro no relatório.

vamos usar MLP e Naive Bayes como os vistos em aula. os nao vistos em aula vamos usar XGBoost e regressão logística. 

para métricas, não vamos usar acurácia porque há desbalanceamento nas classes. vamos usar recall porque não é interessante analisar falsos alarmes de churn (precisao), mas é interessante analisar se o modelo não encontrou churns que deveriam ser enocntrados. também podemos analisar especificidade para analisar se o modelo acerta os clientes que ficam na medida em que deveria. vamos puxar o F1 macro para analisar o desempenho dos modelos nesse dataset desbalanceado, e nao usaremos acuracia balanceada justamente porque classes grandes e pequenas recebem o mesmo peso, e isso nao é interessante para nós. nao vamos usar f1 weighted pra nao favorecer a classe maojoritaria. podemos usar também entropia cruzada para analisar se os modelos estao errando o óbvio. nao vamos usar roc-auc porque já sabemos que nao-churn sao mt maiores que os churn. podemos usar a PR-AUC justamente pelo motivo anterior.

vamos usar validacao cruzada estratificada para divisao do dataset porque há alto desbalanceamento nas classes.

me pergunte o que voce ainda tem dúvida. depois, adicione ao notebook as células para responder as hipóteses que ainda nao foram respondidas (com gráficos) e registre na análise exploratória as minhas decisões.
```
IA
```
Revisei as saídas. Antes de registrar as decisões, tenho algumas dúvidas que mudam o que vai para o relatório.

1. O F1 macro também dá o mesmo peso às duas classes (é a média simples do F1 de cada uma), e esse foi o motivo para descartar a acurácia balanceada. Como justificamos o F1 macro no relatório?
   - Manter o F1 macro (Recomendado): o peso igual é proposital para a classe churn não ser abafada; descartamos a acurácia balanceada por ser redundante com recall + especificidade (ela é a média dos dois).
   - Trocar por F1 da classe churn: usar só o F1 da classe 1, sem média entre classes.
   - Rever o critério: vou repensar quais métricas usar.
2. Quantos folds na validação cruzada estratificada, e como testar as parametrizações (o enunciado pede pelo menos 2 por algoritmo)?
   - 10 folds, mesma CV (Recomendado): todas as configurações avaliadas nos mesmos 10 folds (mesma semente), comparação justa entre algoritmos e parâmetros.
   - 5 folds, mesma CV: mais rápido, com ~89 churns por fold de teste.
   - CV aninhada: CV interna escolhe parâmetros e CV externa estima desempenho. Mais rigoroso e mais complexo.
3. Como tratar o desbalanceamento no treino?
   - Pesos de classe (Recomendado): class_weight na regressão logística, scale_pos_weight no XGBoost. O MLP do sklearn não aceita pesos, e o Naive Bayes usa as priors.
   - SMOTE só no treino: oversampling sintético dentro de cada fold (imblearn pipeline). Vale para os 4 algoritmos.
   - Não tratar: treinar na distribuição original e discutir o efeito no recall.
4. Na hipótese H4 sobre falhas: considero o número absoluto de falhas ou a taxa de falhas (falhas ÷ ligações)?
   - Os dois (Recomendado): absoluto e taxa; a taxa corrige o fato de quem liga mais falhar mais.
   - Só absoluto: usa Call Failure como está.
```
Usuário
```
1. Manter o F1 macro (Recomendado)
2. 10 folds, mesma CV (Recomendado)
3. Pesos de classe (Recomendado)
4. Os dois (Recomendado)
```
IA
```
Atualizei as conclusões do notebook, adicionei a seção de hipóteses e registrei suas decisões em [consideracoes.md](consideracoes.md). As células novas ainda não foram rodadas, porque nenhum Python do sistema tem pandas instalado. Escrevi as conclusões com números que calculei direto no CSV (base sem duplicatas), então elas devem bater com a saída quando você rodar.

**Conclusões revisadas no notebook**
- Cada markdown depois das células de código agora traz os números reais e a sua decisão: duplicatas, `Age Group`, `Charge Amount = 10`, clientes sem uso, ruído de medição e `Status`.
- Saíram as menções a kNN. As implicações agora são para MLP, regressão logística, XGBoost e Naive Bayes.
- O markdown de correlação traz os pares reais. Um deles eu não tinha previsto: `Frequency of SMS` × `Customer Value` (0,92).

**Hipóteses (nova seção, cada uma com gráfico)**
- **H1:** nenhum atributo decide o churn sozinho. A maioria dos clientes que saem não reclamou (295 de 495). A conclusão está marcada com "conferir com a saída acima", porque o recall e a especificidade das regras com um só atributo saem só quando a célula rodar.
- **H2:** reclamar leva o churn de cerca de 10% para cerca de 83%. Mesmo entre clientes ativos, quem reclamou sai em 70,4% dos casos, contra 3,1% de quem não reclamou.
- **H3:** a relação entre falhas e reclamação existe, mas é fraca (r ≈ 0,15).
- **H4:** o número absoluto de falhas engana: o churn cai conforme as falhas aumentam, porque quem liga mais falha mais e sai menos. A taxa de falha corrige isso. Sem reclamação, o churn vai de 2,7% para 21,6%; com reclamação, chega a 89%.
- **H5:** o plano contratual tem pouco churn (2,4%), mas quase todos os churns estão no pré-pago. O plano não é decisivo sozinho.
- **H6:** quem cancela usa entre 35% e 45% do que usa quem fica, com o mesmo tempo de assinatura. Entre os clientes sem nenhum uso, 52% saíram.

**Pontos de atenção**
- Na justificativa do F1 macro, registrei a acurácia balanceada como descartada por ser a média entre recall e especificidade, que vocês já analisam separadamente. O argumento original, de que ela dá o mesmo peso às classes, também valeria contra o F1 macro.
- O XGBoost não vem com o scikit-learn: é preciso rodar `pip install xgboost` no kernel do notebook.
```

Usuário
```
uma dúvida: como a MLP vai lidar com o desbalanceamento das classes se nao aceita pesos?
```
IA
````
Do jeito que está, não lida. Foi um furo meu ter deixado isso só como "a discutir". Treinada na distribuição original, a MLP minimiza a entropia cruzada média, e 84% dos exemplos são não churn. Ela tende a empurrar as probabilidades para essa classe, e o recall de churn cai.

Há três formas de corrigir. Recomendo a primeira, se a sua versão do scikit-learn permitir.

1. **`sample_weight` no `fit`.** É o mesmo princípio do `class_weight`, peso maior para os exemplos de churn, só que aplicado exemplo a exemplo. Pelo que lembro, o `MLPClassifier` passou a aceitar esse argumento a partir do scikit-learn 1.7, mas não tenho certeza. Confira a versão com `sklearn.__version__`, ou chame o `fit` com o argumento e veja se dá erro:
   ```python
   from sklearn.utils.class_weight import compute_sample_weight
   w = compute_sample_weight('balanced', y_treino)
   mlp.fit(X_treino, y_treino, sample_weight=w)
   ```
   Vantagem: os três modelos com pesos usam o mesmo critério ("balanced"), o que deixa a comparação justa e fácil de justificar no relatório.
   Cuidado: se a MLP estiver dentro de um `Pipeline` com o padronizador, o peso é passado como `pipe.fit(X, y, mlp__sample_weight=w)`. Calcule `w` dentro de cada fold, só com o `y` de treino.

2. **Sobreamostragem só no treino de cada fold.** Pode ser a simples (`RandomOverSampler`) ou o SMOTE, usando o `Pipeline` do `imblearn` para não vazar para o teste. Funciona em qualquer versão. A desvantagem é que a MLP passa a usar um tratamento diferente dos outros modelos, e isso precisa ser explicado no relatório.

3. **Ajustar o limiar de decisão** de 0,5 para um valor menor. Isso aumenta o recall sem mexer no treino. Mas o limiar precisa ser escolhido numa validação interna, senão vaza informação do teste. É mais trabalhoso, então eu deixaria só como possível melhoria.

O **Naive Bayes** tem uma questão parecida. Por padrão ele usa as probabilidades a priori da base (84/16). Se quiser deixá-lo comparável aos outros, dá para passar `priors=[0.5, 0.5]`. Isso pode até ser uma das duas parametrizações exigidas para ele: priors da base contra priors uniformes.

Quer que eu atualize a decisão sobre desbalanceamento em [consideracoes.md](consideracoes.md) assim que você confirmar a versão do scikit-learn?
````

Usuário
```
confere as saídas em @src/analise-exploratoria.ipynb
```
IA
```
Só deu para conferir a H1. As células de H2 a H6 (31, 34, 37, 40 e 43) estão sem saída no arquivo salvo. Ou não terminaram de rodar, ou o notebook foi salvo antes. Rode de novo a partir da célula 26, salve, e eu confiro as demais.

**H1 (célula 28):** as saídas confirmam a conclusão, e atualizei o texto com os números:
- **"Reclamou → churn":** recall de 0,426 e especificidade de 0,983. Quase todo cliente que reclama sai, mas 57% dos clientes que saem nunca reclamaram.
- **"Inativo → churn":** recall de 0,729 e especificidade de 0,851.
- Acrescentei que essas regras de um atributo só servem de referência mínima. Um modelo com recall abaixo de 0,73 não supera a regra do `Status`, que além disso é suspeita de vazamento.

**H6:** corrigi uma faixa no texto. As medianas dos clientes churn ficam entre 35% e 46% das de quem fica, e não 45% como eu tinha escrito (números distintos chamados: 11 contra 24 = 46%).
```

Usuário
```
rodei de novo, confere as saídas agora e atualiza o @consideracoes.md  com as conclusoes das hipoteses resolvidas e o que vamos resolver com os algoritmos. sobre a MLP: acho que consigo usar sim, nao consegui rodar pq y_treino nao tava definido ainda, mas vamos tentar em um primeiro momento. sobre naive-bayes, pode usar 0.5 e 0.5, mas justifique por que está usando no @consideracoes.md , como influenci o resultado e qual o efeito disso na base.
```
IA
```
Conferi as saídas de H1 a H6 e todas confirmam as conclusões escritas no notebook. Só precisei corrigir H2 e H5, que ainda citavam números da base original e agora usam os da base sem duplicatas (2850 clientes):
- **H2:** 82,6% de churn entre quem reclamou, contra 9,8% de quem não reclamou.
- **H5:** 2,6% de churn no plano contratual (229 clientes), contra 16,8% no pré-pago.

Em [consideracoes.md](consideracoes.md), adicionei três seções:

1. **Respostas às hipóteses:** um resumo de H1 a H6 com os números da base sem duplicatas.
2. **O que fica para os algoritmos:**
   - se a combinação de atributos supera a regra de um atributo só, cujo recall de referência é 0,73 (`Status`);
   - o quanto o `Status` infla o desempenho;
   - as relações não lineares (idade, faixa de cobrança, falhas × reclamação);
   - o efeito dos atributos redundantes no Naive Bayes e na regressão logística;
   - a importância dos atributos nos modelos, para confirmar as hipóteses;
   - o teto de desempenho imposto pelo ruído de rótulo e de medição.
3. **Desbalanceamento:** os pesos ficaram no mesmo critério ("balanced") para todos. A MLP usa `sample_weight` no `fit`, calculado dentro de cada fold só com o `y` de treino. Se a versão do scikit-learn não aceitar, a alternativa registrada é sobreamostrar a classe churn só no treino.

**Naive Bayes com priors 0,5 / 0,5:** a justificativa está numa subseção própria.
- **Por que usar:** com os priors da base, a evidência dos atributos precisa vencer uma vantagem de cerca de 5,4 vezes a favor de "fica", o que tende a derrubar o recall. Com priors uniformes, a decisão depende só dos atributos, como nos pesos "balanced" dos outros modelos.
- **Como influencia o resultado:** a chance de churn de todos os clientes é multiplicada por cerca de 5,4. O recall sobe e a especificidade cai. O log loss tende a piorar, porque as probabilidades deixam de refletir a proporção real de churn.
- **A PR-AUC não muda:** a ordem dos clientes por probabilidade continua a mesma; só muda o ponto de corte. É um bom ponto para discutir no relatório.
- **Efeito na base:** nenhum. Os dados e a proporção de churn nos folds continuam iguais; muda só a regra de decisão do modelo.

Em "o que fica para os algoritmos", incluí comparar recall e especificidade com e sem pesos de classe. Vocês não decidiram isso; se não quiserem rodar sem pesos, é só remover essa linha.
```

Usuário
```
salva os resultados de cada gráfico gerado no notebook em @src/analise-exploratoria.ipynb  na pasta @data/results/hypotesis-conclusions com o nome da hipótese que aquele gráfico fecha. salva o dataset sem as duplicatas em @data/processed também em .csv
```
IA
```
Salvei os 9 gráficos e o CSV.

**Gráficos**, em [data/results/hypotesis-conclusions/](data/results/hypotesis-conclusions/):
- **Hipóteses:** `H1-correlacao-com-churn`, `H2-churn-por-reclamacao`, `H3-falhas-x-reclamacao`, `H4-churn-por-falhas-e-reclamacao`, `H5-churn-por-plano` e `H6-uso-churn-x-nao-churn`.
- **Gráficos da análise geral:** três deles não fecham nenhuma hipótese, então usei o prefixo `eda-` em vez de um nome de hipótese: `eda-distribuicao-classes`, `eda-boxplots-atributos-numericos` e `eda-correlacao`. Se preferir, posso renomear (por exemplo, `eda-correlacao` também apoia a H1) ou removê-los.

**Dataset:** está em [data/processed/customer-churn-sem-duplicatas.csv](data/processed/customer-churn-sem-duplicatas.csv), com 2850 linhas.
- Os nomes das colunas ficaram sem os espaços duplos.
- Só as duplicatas foram removidas. `Age Group` continua no arquivo; tirar essa coluna fica para a etapa de pré-processamento.

**Mudanças no notebook:** acrescentei código para que tudo seja salvo de novo quando você rodar.
- Na primeira célula, defini `FIG_DIR` (a pasta dos gráficos).
- Antes de cada `plt.show()`, há agora um `plt.savefig`.
- A célula que cria `dfu` passou a gravar o CSV.

As imagens que salvei agora vieram das saídas que já estavam no notebook, ou seja, da sua última execução. Rodar o notebook de novo sobrescreve os arquivos com o mesmo conteúdo.
```

Usuário
```
coloca toda essa sessão escrita na integra em @prompts.md  seguindo o padrão que as outras estao escritas.
```
IA
```
Registrei a sessão 2 inteira no fim de [prompts.md](prompts.md), no mesmo formato das anteriores: blocos "Usuário" e "IA" alternados, cada um entre crases. Antes, a sessão só tinha o primeiro prompt e um bloco IA vazio, que preenchi.

Três detalhes de formato:
- **Perguntas com opções:** a rodada de perguntas da sessão ficou como um bloco IA com as perguntas e as opções. Logo depois vem um bloco "Usuário" com as opções que você escolheu.
- **Resposta sobre a MLP:** ela tem um bloco de código dentro. Para não quebrar a formatação, envolvi essa resposta com quatro crases em vez de três.
- **Último pedido:** o pedido de registrar a sessão ficou como o último bloco "Usuário", ainda sem resposta. Se quiser, coloque esta resposta depois dele.

Copiei só as mensagens de texto da sessão. Os comandos e scripts que rodei por baixo ficaram de fora, como nas outras sessões. Se o item 4 do enunciado exigir mostrar também como as respostas foram verificadas, posso acrescentar um resumo dessas checagens: os números conferidos direto no CSV e as saídas do notebook comparadas com as conclusões.
```

# Pré-processamento e transformação dos dados

## Claude Code - Opus 5.5 low - Sessão 3
Usuário
```
```
IA
```
```