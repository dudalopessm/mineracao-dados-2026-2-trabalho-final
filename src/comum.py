"""Funções compartilhadas pelos notebooks de modelagem.

Concentra num lugar só o que precisa ser igual para todos os algoritmos:
leitura das bases, listas de colunas, folds da validação cruzada,
pré-processamento dentro do fold e cálculo das métricas.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- caminhos
RAIZ = Path(__file__).resolve().parent.parent
PROCESSED_DIR = RAIZ / 'data' / 'processed'
RESULTS_DIR = RAIZ / 'data' / 'results' / 'modelagem'
FOLDS_CSV = PROCESSED_DIR / 'folds.csv'

# ---------------------------------------------------------------- constantes
SEMENTE = 42
N_FOLDS = 10

N_LINHAS = 2850
N_CHURN = 446

ALVO = 'churn'
NUMERICAS = [
    'falhas_ligacao', 'meses_assinatura', 'faixa_cobranca', 'segundos_uso',
    'qtd_ligacoes', 'qtd_sms', 'numeros_distintos', 'valor_cliente',
]
BINARIAS = [
    'reclamacao', 'plano_contratual', 'inativo',
    'idade_15', 'idade_25', 'idade_30', 'idade_45', 'idade_55',
]
# atributos removidos na versão do Naive Bayes sem redundantes
REDUNDANTES_NB = ['segundos_uso']

METRICAS = ['recall', 'especificidade', 'f1_macro', 'log_loss', 'pr_auc']


# ---------------------------------------------------------------- dados
def colunas(com_status):
    """Devolve (numericas, binarias) da base com ou sem `inativo`."""
    binarias = BINARIAS if com_status else [c for c in BINARIAS if c != 'inativo']
    return list(NUMERICAS), list(binarias)


COLS_IDADE = ['idade_15', 'idade_25', 'idade_30', 'idade_45', 'idade_55']
CONTAGEM_IDADE = {15: 112, 25: 921, 30: 1296, 45: 367, 55: 154}


def idade_numerica(X):
    """Troca as 5 colunas one-hot da idade por uma coluna `idade` (15 a 55).

    Usada no Naive Bayes e no XGBoost, que não precisam do one-hot.
    Transformação linha a linha, sem vazamento entre treino e teste.
    """
    assert X[COLS_IDADE].sum(axis=1).eq(1).all(), 'cada cliente deve estar em exatamente uma faixa'
    idade = X[COLS_IDADE].idxmax(axis=1).str.removeprefix('idade_').astype(int)
    assert idade.value_counts().to_dict() == CONTAGEM_IDADE
    return X.drop(columns=COLS_IDADE).assign(idade=idade)


def nome_base(com_status):
    return 'com-status' if com_status else 'sem-status'


def carregar_base(com_status):
    """Lê a base pré-processada e devolve (X, y), sem reordenar as linhas."""
    df = pd.read_csv(PROCESSED_DIR / f'churn-{nome_base(com_status)}.csv')

    numericas, binarias = colunas(com_status)
    assert list(df.columns) == numericas + binarias + [ALVO], 'colunas inesperadas'
    assert len(df) == N_LINHAS
    assert df[ALVO].sum() == N_CHURN

    return df.drop(columns=ALVO), df[ALVO].to_numpy()


# ---------------------------------------------------------------- folds
def gerar_folds(y):
    """Número do fold de teste (0 a N_FOLDS-1) de cada linha."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    fold = np.empty(len(y), dtype=int)
    # o StratifiedKFold usa só o y (o X entra apenas pelo número de linhas)
    for k, (_, idx_teste) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[idx_teste] = k
    return fold


def conferir_folds(fold, y):
    """Asserts sobre um vetor de folds; usado na geração e na leitura."""
    y = np.asarray(y)
    assert len(fold) == len(y), 'folds e base com tamanhos diferentes'
    assert set(np.unique(fold)) == set(range(N_FOLDS)), 'folds fora de 0..9'

    churn_por_fold = np.bincount(fold, weights=y, minlength=N_FOLDS)
    minimo = N_CHURN // N_FOLDS
    assert np.isin(churn_por_fold, [minimo, minimo + 1]).all(), 'folds não estratificados'

    # se uma versão futura do scikit-learn mudar o embaralhamento, avisa aqui
    assert np.array_equal(fold, gerar_folds(y)), 'folds diferentes dos gerados com a SEMENTE'


def carregar_folds(y):
    """Lê `folds.csv` e devolve a lista de (idx_treino, idx_teste)."""
    fold = pd.read_csv(FOLDS_CSV)['fold'].to_numpy()
    conferir_folds(fold, y)
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(N_FOLDS)]


# ---------------------------------------------------------------- pré-processamento
def criar_preprocessador(com_status):
    """Padroniza as numéricas e passa as binárias/one-hot sem mudança.

    Usado nos pipelines da MLP e da regressão logística. Dentro do pipeline,
    a média e o desvio são aprendidos só no treino de cada fold.
    """
    numericas, binarias = colunas(com_status)
    return ColumnTransformer(
        [('num', StandardScaler(), numericas), ('bin', 'passthrough', binarias)],
        remainder='drop',
    )


# ---------------------------------------------------------------- avaliação
def calcular_metricas(y_true, y_pred, proba):
    return {
        'recall': recall_score(y_true, y_pred, pos_label=1),
        'especificidade': recall_score(y_true, y_pred, pos_label=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'log_loss': log_loss(y_true, proba, labels=[0, 1]),
        'pr_auc': average_precision_score(y_true, proba),
    }


def avaliar(criar_modelo, X, y, folds, parametros_fit=None):
    """Validação cruzada com os folds fixos.

    criar_modelo(y_treino) -> estimador novo, sem ajuste. Recebe o y do treino
        para parâmetros que dependem dele (ex.: scale_pos_weight do XGBoost).
    parametros_fit(y_treino) -> dict de kwargs do fit (ex.: sample_weight da MLP;
        dentro de um Pipeline, use o prefixo do passo, como 'mlp__sample_weight').

    Devolve um dict com:
        por_fold  : DataFrame (um fold por linha) com as métricas
        resumo    : DataFrame com média e desvio de cada métrica
        matriz    : matriz de confusão somada nos folds ([[TN, FP], [FN, TP]])
        proba_oof : probabilidade de churn de cada cliente, prevista quando ele
                    estava no fold de teste
        modelos   : modelo ajustado em cada fold (para coeficientes e importâncias)
    """
    X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else X
    y = np.asarray(y)

    linhas, modelos = [], []
    matriz = np.zeros((2, 2), dtype=int)
    proba_oof = np.full(len(y), np.nan)

    for k, (idx_treino, idx_teste) in enumerate(folds):
        X_tr, X_te = _linhas(X, idx_treino), _linhas(X, idx_teste)
        y_tr, y_te = y[idx_treino], y[idx_teste]

        modelo = criar_modelo(y_tr)
        kwargs = parametros_fit(y_tr) if parametros_fit else {}
        modelo.fit(X_tr, y_tr, **kwargs)

        proba = modelo.predict_proba(X_te)[:, 1]
        pred = modelo.predict(X_te)

        linhas.append({'fold': k, **calcular_metricas(y_te, pred, proba)})
        matriz += confusion_matrix(y_te, pred, labels=[0, 1])
        proba_oof[idx_teste] = proba
        modelos.append(modelo)

    assert not np.isnan(proba_oof).any(), 'algum cliente ficou fora dos folds de teste'

    por_fold = pd.DataFrame(linhas).set_index('fold')
    resumo = por_fold[METRICAS].agg(['mean', 'std']).T.rename(
        columns={'mean': 'media', 'std': 'desvio'})
    return {
        'por_fold': por_fold,
        'resumo': resumo,
        'matriz': matriz,
        'proba_oof': proba_oof,
        'modelos': modelos,
    }


def formatar_metricas(por_fold):
    """Média ± desvio de cada métrica, como texto."""
    return {m: f'{por_fold[m].mean():.3f} ± {por_fold[m].std():.3f}' for m in METRICAS}


def diferenca_pareada(r_var, r_ref):
    """Diferença fold a fold (variação − referência) de cada métrica, como média ± desvio.

    Válida porque todas as configurações usam os mesmos folds.
    """
    dif = r_var['por_fold'][METRICAS] - r_ref['por_fold'][METRICAS]
    return {m: f'{dif[m].mean():+.3f} ± {dif[m].std():.3f}' for m in METRICAS}


def tabela_desbalanceamento(r_com, r_sem, base):
    """Linhas comparando a versão com tratamento do desbalanceamento e a sem.

    r_com: resultado com pesos de classe (ou priors uniformes); r_sem: sem.
    """
    def erros(r):
        (tn, fp), (fn, tp) = r['matriz']
        return {'FN (churns perdidos)': fn, 'FP (falsos alarmes)': fp}

    return [
        {'base': base, 'versão': 'com tratamento', **formatar_metricas(r_com['por_fold']), **erros(r_com)},
        {'base': base, 'versão': 'sem tratamento', **formatar_metricas(r_sem['por_fold']), **erros(r_sem)},
        {'base': base, 'versão': 'diferença (sem − com)', **diferenca_pareada(r_sem, r_com)},
    ]


def _linhas(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


# ---------------------------------------------------------------- resultados
def salvar_resultados(resultado, algoritmo, config, com_status):
    """Grava métricas por fold, matriz somada e probabilidades fora do fold.

    Arquivos em data/results/modelagem/, com prefixo <algoritmo>_<config>_<base>.
    Devolve o prefixo usado.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = nome_base(com_status)
    prefixo = f'{algoritmo}_{config}_{base}'
    rotulos = {'algoritmo': algoritmo, 'config': config, 'base': base}

    por_fold = resultado['por_fold'].reset_index().assign(**rotulos)
    por_fold.to_csv(RESULTS_DIR / f'{prefixo}-metricas.csv', index=False)

    tn, fp, fn, tp = resultado['matriz'].ravel()
    pd.DataFrame([{**rotulos, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}]).to_csv(
        RESULTS_DIR / f'{prefixo}-matriz.csv', index=False)

    pd.DataFrame({'proba': resultado['proba_oof']}).to_csv(
        RESULTS_DIR / f'{prefixo}-proba.csv', index=False)
    return prefixo


ALGORITMOS = ['mlp', 'nb', 'xgb', 'rl']


def registrar_escolhida(algoritmo, config):
    """Grava a configuração escolhida de um algoritmo, lida pelo notebook de comparação."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f'{algoritmo}-escolhida.txt').write_text(config, encoding='utf-8')


def ler_escolhida(algoritmo):
    return (RESULTS_DIR / f'{algoritmo}-escolhida.txt').read_text(encoding='utf-8').strip()


def _rotulos_do_arquivo(caminho, sufixo):
    # <algoritmo>_<config>_<base><sufixo>; a config pode ter "_" (ex.: relu_adam_32)
    partes = caminho.name[:-len(sufixo)].split('_')
    return partes[0], '_'.join(partes[1:-1]), partes[-1]


def ler_resultados():
    """Lê tudo o que `salvar_resultados` gravou.

    Devolve (metricas, matrizes, probas):
        metricas : DataFrame com uma linha por fold (algoritmo, config, base, fold, métricas)
        matrizes : DataFrame com tn, fp, fn, tp por (algoritmo, config, base)
        probas   : dict (algoritmo, config, base) -> probabilidades fora do fold
    Ignora os arquivos de grade (`mlp-grade-...`), que não seguem o padrão.
    """
    def arquivos(sufixo):
        return sorted(f for f in RESULTS_DIR.glob(f'*{sufixo}')
                      if f.name.split('_')[0] in ALGORITMOS)

    metricas = pd.concat([pd.read_csv(f) for f in arquivos('-metricas.csv')], ignore_index=True)
    matrizes = pd.concat([pd.read_csv(f) for f in arquivos('-matriz.csv')], ignore_index=True)
    probas = {_rotulos_do_arquivo(f, '-proba.csv'): pd.read_csv(f)['proba'].to_numpy()
              for f in arquivos('-proba.csv')}
    return metricas, matrizes, probas


def resultado_salvo(metricas, matrizes, algoritmo, config, base):
    """Reconstrói, a partir dos arquivos, o dict no formato do `avaliar` (por_fold e matriz)."""
    sel = (metricas['algoritmo'] == algoritmo) & (metricas['config'] == config) & (metricas['base'] == base)
    por_fold = metricas[sel].set_index('fold').sort_index()
    assert len(por_fold) == N_FOLDS, f'{algoritmo} {config} {base}: resultados não encontrados'
    m = matrizes[(matrizes['algoritmo'] == algoritmo) & (matrizes['config'] == config)
                 & (matrizes['base'] == base)].iloc[0]
    return {'por_fold': por_fold, 'matriz': np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])}
