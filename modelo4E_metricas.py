# ==================================================================================================
# MODELO 4E
# PREVISÃO DAS PRINCIPAIS MÉTRICAS DO MERCADO
#
# Framework Híbrido Markov-Neural
#
# Objetivo:
# Prever, para o trimestre seguinte:
#
#   1. juros_cartao_pf_aa_pct
#   2. inadimplencia_90d_pct
#   3. hhi_credito_0_1
#   4. cr5_credito
#
# Modelos comparados:
#
#   - Persistência
#   - Regressão Ridge
#   - Rede Neural MLP
#
# Estratégia:
#
#   - previsão t -> t+1
#   - validação walk-forward com janela expansiva
#   - nenhuma informação futura utilizada no treinamento
#   - imputação ajustada somente dentro da janela histórica
#   - padronização ajustada somente dentro da janela histórica
#
# Seleção:
#
#   Para cada métrica, o modelo com menor MAE histórico é selecionado
#   para a previsão prospectiva.
#
# Adicionalmente:
#
#   - análise de robustez da MLP para juros
#   - teste de Diebold-Mariano:
#       MLP × Persistência
#
# ==================================================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


warnings.filterwarnings("ignore")


# ==================================================================================================
# CONFIGURAÇÕES
# ==================================================================================================

ARQUIVO_ENTRADA = Path(__file__).resolve().parent / "dataset_model_ready.csv"

PASTA_SAIDA = Path(
    "outputs_modelo4_final"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

MIN_TREINO = 16

RANDOM_STATE = 42


# ==================================================================================================
# MÉTRICAS A PREVER
# ==================================================================================================

METRICAS_ALVO = [
    "juros_cartao_pf_aa_pct",
    "inadimplencia_90d_pct",
    "hhi_credito_0_1",
    "cr5_credito",
]


# ==================================================================================================
# VARIÁVEIS EXPLICATIVAS
# ==================================================================================================

FEATURES = [
    "juros_cartao_pf_aa_pct",
    "selic_am_pct",
    "inadimplencia_90d_pct",
    "hhi_credito_0_1",
    "cr5_credito",
    "comprometimento_renda_pct",
    "taxa_desocupacao_trimestral_pct",
    "rendimento_medio_real_brl",
    "prop_rotativo",
]


# ==================================================================================================
# FUNÇÕES AUXILIARES
# ==================================================================================================

def linha():
    print("=" * 100)


def titulo(texto):
    print()
    linha()
    print(texto)
    linha()


def quarter_label(data):
    """
    Converte uma data para identificação trimestral no formato YYYYQX.
    """

    data = pd.Timestamp(data)

    return str(
        data.to_period("Q")
    )


def limitar_previsao(
    valor,
    metrica
):
    """
    Aplica limites economicamente plausíveis.

    Juros:
        >= 0

    Inadimplência:
        0 a 100

    HHI:
        0 a 1

    CR5:
        0 a 1
    """

    valor = float(valor)

    if metrica == "inadimplencia_90d_pct":

        return float(
            np.clip(
                valor,
                0,
                100
            )
        )

    if metrica in [
        "hhi_credito_0_1",
        "cr5_credito",
    ]:

        return float(
            np.clip(
                valor,
                0,
                1
            )
        )

    return float(
        max(
            0,
            valor
        )
    )


# ==================================================================================================
# MODELO MLP
# ==================================================================================================

def treinar_mlp(
    X_treino,
    y_treino,
    X_pred
):
    """
    MLP extremamente compacta devido ao tamanho reduzido da amostra.

    A variável alvo também é padronizada para melhorar a estabilidade
    numérica do treinamento.
    """

    media_y = np.mean(
        y_treino
    )

    desvio_y = np.std(
        y_treino,
        ddof=0
    )

    if (
        not np.isfinite(desvio_y)
        or desvio_y == 0
    ):

        desvio_y = 1.0

    y_padronizado = (
        y_treino
        - media_y
    ) / desvio_y

    modelo = MLPRegressor(
        hidden_layer_sizes=(4,),
        activation="tanh",
        solver="lbfgs",
        alpha=2.0,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    modelo.fit(
        X_treino,
        y_padronizado
    )

    pred_padronizada = (
        modelo.predict(
            X_pred
        )[0]
    )

    pred = (
        pred_padronizada
        * desvio_y
        + media_y
    )

    return float(pred)


# ==================================================================================================
# TESTE DE DIEBOLD-MARIANO
# ==================================================================================================

def diebold_mariano_test(
    y_real,
    pred_modelo_1,
    pred_modelo_2,
    h=1,
    loss="squared",
):
    """
    Teste simplificado de Diebold-Mariano para previsão 1 passo à frente.

    H0:
        mesma capacidade preditiva

    H1:
        capacidade preditiva diferente

    Convenção:

        d = perda_modelo_1 - perda_modelo_2

    Assim:

        DM < 0
            modelo 1 possui menor perda média

        DM > 0
            modelo 2 possui menor perda média

    Neste trabalho:

        modelo 1 = MLP
        modelo 2 = Persistência
    """

    y_real = np.asarray(
        y_real,
        dtype=float
    )

    pred_modelo_1 = np.asarray(
        pred_modelo_1,
        dtype=float
    )

    pred_modelo_2 = np.asarray(
        pred_modelo_2,
        dtype=float
    )

    mascara = (
        np.isfinite(y_real)
        & np.isfinite(pred_modelo_1)
        & np.isfinite(pred_modelo_2)
    )

    y_real = y_real[
        mascara
    ]

    pred_modelo_1 = pred_modelo_1[
        mascara
    ]

    pred_modelo_2 = pred_modelo_2[
        mascara
    ]

    n = len(
        y_real
    )

    if n < 2:

        return (
            np.nan,
            np.nan
        )

    erro_1 = (
        y_real
        - pred_modelo_1
    )

    erro_2 = (
        y_real
        - pred_modelo_2
    )

    if loss == "absolute":

        perda_1 = np.abs(
            erro_1
        )

        perda_2 = np.abs(
            erro_2
        )

    elif loss == "squared":

        perda_1 = (
            erro_1 ** 2
        )

        perda_2 = (
            erro_2 ** 2
        )

    else:

        raise ValueError(
            "loss deve ser 'absolute' ou 'squared'"
        )

    d = (
        perda_1
        - perda_2
    )

    media_d = np.mean(
        d
    )

    variancia_d = np.var(
        d,
        ddof=1
    )

    if (
        not np.isfinite(
            variancia_d
        )
        or variancia_d <= 0
    ):

        return (
            np.nan,
            np.nan
        )

    dm = (
        media_d
        /
        np.sqrt(
            variancia_d
            / n
        )
    )

    # Correção Harvey, Leybourne e Newbold
    fator_correcao = np.sqrt(
        (
            n
            + 1
            - 2 * h
            + (
                h
                * (h - 1)
                / n
            )
        )
        / n
    )

    dm_corrigido = (
        dm
        * fator_correcao
    )

    graus_liberdade = (
        n
        - 1
    )

    p_valor = (
        2
        * stats.t.sf(
            abs(
                dm_corrigido
            ),
            df=graus_liberdade,
        )
    )

    return (
        float(
            dm_corrigido
        ),
        float(
            p_valor
        ),
    )


# ==================================================================================================
# LEITURA DA BASE
# ==================================================================================================

titulo(
    "MODELO 4E — PREVISÃO DAS PRINCIPAIS MÉTRICAS"
)

print(
    f"\nArquivo de entrada: "
    f"{ARQUIVO_ENTRADA}"
)

df = pd.read_csv(
    ARQUIVO_ENTRADA
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = (
    df
    .dropna(
        subset=["date"]
    )
    .sort_values(
        "date"
    )
    .reset_index(
        drop=True
    )
)


# ==================================================================================================
# CONVERSÃO PARA FREQUÊNCIA TRIMESTRAL
# ==================================================================================================
#
# O framework de regimes trabalha em frequência trimestral.
# Portanto, o Modelo 4E também deve prever t+1 em termos de trimestre,
# e não da próxima linha mensal.
#
# Variáveis mensais são agregadas pela média dentro do trimestre.
# Variáveis estruturais trimestrais, como HHI e CR5, possuem apenas uma
# observação efetiva no trimestre; portanto, a média preserva o valor
# observado sem preenchimento artificial.
#
# O trimestre calendário ainda em andamento é excluído da modelagem.
# =============================================================================

df["_quarter"] = (
    df["date"]
    .dt.to_period("Q")
)

# Exigir os três meses distintos do trimestre na base de entrada.
# Isso evita agregar um trimestre com apenas um ou dois meses.
df["_month"] = df["date"].dt.to_period("M")

contagem_meses = (
    df.groupby("_quarter")["_month"]
    .nunique()
)

trimestres_completos_na_base = set(
    contagem_meses[
        contagem_meses == 3
    ].index
)

df = (
    df.loc[
        df["_quarter"].isin(
            trimestres_completos_na_base
        )
    ]
    .copy()
)

df = df.drop(columns=["_month"])

if df.empty:
    raise ValueError(
        "Nenhum trimestre com três meses disponíveis "
        "na base de entrada."
    )

trimestre_atual = (
    pd.Timestamp
    .today()
    .to_period("Q")
)

ultimo_trimestre_completo = (
    trimestre_atual - 1
)

df = (
    df.loc[
        df["_quarter"]
        <= ultimo_trimestre_completo
    ]
    .copy()
)

colunas_numericas = (
    df
    .select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)

df = (
    df
    .groupby(
        "_quarter",
        as_index=False
    )[colunas_numericas]
    .mean()
)

df["date"] = (
    df["_quarter"]
    .dt.end_time
    .dt.normalize()
)

df = (
    df
    .drop(
        columns=["_quarter"]
    )
    .sort_values(
        "date"
    )
    .reset_index(
        drop=True
    )
)

print(
    f"\\nFrequência convertida para trimestral."
)

print(
    f"Trimestres disponíveis: {len(df)}"
)

print(
    f"Último trimestre completo: "
    f"{quarter_label(df['date'].max())}"
)


# ==================================================================================================
# VERIFICAÇÃO DAS COLUNAS
# ==================================================================================================

colunas_necessarias = (
    ["date"]
    + FEATURES
    + METRICAS_ALVO
)

colunas_necessarias = list(
    dict.fromkeys(
        colunas_necessarias
    )
)

faltantes = [
    c
    for c in colunas_necessarias
    if c not in df.columns
]

if faltantes:

    raise ValueError(
        "\nColunas ausentes no dataset:\n"
        + "\n".join(
            faltantes
        )
    )


# ==================================================================================================
# INFORMAÇÕES DA BASE
# ==================================================================================================

print(
    f"\nNúmero de observações: "
    f"{len(df)}"
)

print(
    f"Período: "
    f"{quarter_label(df['date'].min())}"
    f" até "
    f"{quarter_label(df['date'].max())}"
)

print(
    f"\nNúmero de features: "
    f"{len(FEATURES)}"
)

print(
    "\nCobertura das features:"
)

for coluna in FEATURES:

    validos = (
        df[coluna]
        .notna()
        .sum()
    )

    print(
        f"  {coluna:<45}"
        f"{validos:>3}/{len(df)}"
    )


# ==================================================================================================
# CONSTRUÇÃO DO TARGET t+1
# ==================================================================================================

# ==================================================================================================
# BASE DO MODELO
#
# FEATURES e METRICAS_ALVO possuem algumas colunas em comum.
# Por isso, removemos duplicidades antes de selecionar as colunas.
# ==================================================================================================

colunas_modelo = list(
    dict.fromkeys(
        ["date"]
        + FEATURES
        + METRICAS_ALVO
    )
)

dados_modelo = (
    df[
        colunas_modelo
    ]
    .copy()
)

for metrica in METRICAS_ALVO:

    dados_modelo[
        f"{metrica}__t1"
    ] = (
        dados_modelo[
            metrica
        ]
        .shift(-1)
    )


# ==================================================================================================
# VALIDAÇÃO WALK-FORWARD
# ==================================================================================================

titulo(
    "VALIDAÇÃO WALK-FORWARD"
)

print(
    f"\nJanela mínima de treinamento: "
    f"{MIN_TREINO} observações"
)

resultados_validacao = []


for metrica in METRICAS_ALVO:

    print(
        f"\nMétrica: "
        f"{metrica}"
    )

    target_futuro = (
        f"{metrica}__t1"
    )

    dados_metrica = (
        dados_modelo[
            ["date"]
            + FEATURES
            + [target_futuro]
        ]
        .copy()
    )

    # última linha não possui y(t+1)
    dados_metrica_treino = (
        dados_metrica[
            dados_metrica[
                target_futuro
            ].notna()
        ]
        .reset_index(
            drop=True
        )
    )

    n_total_treino = len(
        dados_metrica_treino
    )

    print(
        f"  Pares t → t+1: "
        f"{n_total_treino}"
    )

    for i in range(
        MIN_TREINO,
        n_total_treino
    ):

        treino = (
            dados_metrica_treino
            .iloc[:i]
            .copy()
        )

        teste = (
            dados_metrica_treino
            .iloc[[i]]
            .copy()
        )

        X_treino_raw = (
            treino[
                FEATURES
            ]
        )

        X_teste_raw = (
            teste[
                FEATURES
            ]
        )

        y_treino = (
            treino[
                target_futuro
            ]
            .astype(float)
            .values
        )

        y_real = float(
            teste[
                target_futuro
            ]
            .iloc[0]
        )

        # ------------------------------------------------------------------------------------------
        # IMPUTAÇÃO
        # ------------------------------------------------------------------------------------------

        imputador = SimpleImputer(
            strategy="median"
        )

        X_treino_imp = (
            imputador.fit_transform(
                X_treino_raw
            )
        )

        X_teste_imp = (
            imputador.transform(
                X_teste_raw
            )
        )

        # ------------------------------------------------------------------------------------------
        # PADRONIZAÇÃO
        # ------------------------------------------------------------------------------------------

        scaler = StandardScaler()

        X_treino = (
            scaler.fit_transform(
                X_treino_imp
            )
        )

        X_teste = (
            scaler.transform(
                X_teste_imp
            )
        )

        # ------------------------------------------------------------------------------------------
        # PERSISTÊNCIA
        # ------------------------------------------------------------------------------------------

        persistencia = float(
            teste[
                metrica
            ]
            .iloc[0]
        )

        persistencia = limitar_previsao(
            persistencia,
            metrica
        )

        # ------------------------------------------------------------------------------------------
        # RIDGE
        # ------------------------------------------------------------------------------------------

        ridge = Ridge(
            alpha=1.0
        )

        ridge.fit(
            X_treino,
            y_treino
        )

        pred_ridge = float(
            ridge.predict(
                X_teste
            )[0]
        )

        pred_ridge = limitar_previsao(
            pred_ridge,
            metrica
        )

        # ------------------------------------------------------------------------------------------
        # MLP
        # ------------------------------------------------------------------------------------------

        pred_mlp = treinar_mlp(
            X_treino,
            y_treino,
            X_teste
        )

        pred_mlp = limitar_previsao(
            pred_mlp,
            metrica
        )

        data_t = (
            teste[
                "date"
            ]
            .iloc[0]
        )

        trimestre_t = (
            data_t.to_period("Q")
        )

        trimestre_t1 = (
            trimestre_t
            + 1
        )

        resultados_validacao.append(
            {
                "metrica": metrica,
                "date_t": data_t,
                "periodo_t": str(
                    trimestre_t
                ),
                "periodo_t1": str(
                    trimestre_t1
                ),
                "valor_t": persistencia,
                "valor_real_t1": y_real,
                "persistencia": persistencia,
                "ridge": pred_ridge,
                "mlp": pred_mlp,
            }
        )


# ==================================================================================================
# DATAFRAME DE VALIDAÇÃO
# ==================================================================================================

df_validacao = pd.DataFrame(
    resultados_validacao
)


# ==================================================================================================
# EXPORTAÇÃO DA VALIDAÇÃO
# ==================================================================================================

arquivo_validacao = (
    PASTA_SAIDA
    / "modelo4E_validacao_metricas.csv"
)

df_validacao.to_csv(
    arquivo_validacao,
    index=False
)

print(
    f"\n✓ {arquivo_validacao}"
)


# ==================================================================================================
# DESEMPENHO DOS MODELOS
# ==================================================================================================

titulo(
    "DESEMPENHO HISTÓRICO"
)

resultados_desempenho = []


for metrica in METRICAS_ALVO:

    temp = (
        df_validacao[
            df_validacao["metrica"]
            == metrica
        ]
        .copy()
    )

    y_real = (
        temp[
            "valor_real_t1"
        ]
        .astype(float)
        .values
    )

    print(
        f"\n{metrica}"
    )

    for modelo in [
        "persistencia",
        "ridge",
        "mlp",
    ]:

        y_pred = (
            temp[
                modelo
            ]
            .astype(float)
            .values
        )

        mae = mean_absolute_error(
            y_real,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_real,
                y_pred
            )
        )

        r2 = r2_score(
            y_real,
            y_pred
        )

        resultados_desempenho.append(
            {
                "metrica": metrica,
                "modelo": modelo,
                "n_previsoes": len(
                    temp
                ),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

        print(
            f"  {modelo:<15}"
            f"MAE={mae:.6f}  "
            f"RMSE={rmse:.6f}  "
            f"R²={r2:.6f}"
        )


df_desempenho = pd.DataFrame(
    resultados_desempenho
)


arquivo_desempenho = (
    PASTA_SAIDA
    / "modelo4E_desempenho_modelos.csv"
)

df_desempenho.to_csv(
    arquivo_desempenho,
    index=False
)

print(
    f"\n✓ {arquivo_desempenho}"
)


# ==================================================================================================
# SELEÇÃO DO MELHOR MODELO POR MAE
# ==================================================================================================

titulo(
    "MELHOR MODELO POR MÉTRICA"
)

melhores_modelos = (
    df_desempenho
    .sort_values(
        [
            "metrica",
            "mae",
        ]
    )
    .groupby(
        "metrica",
        as_index=False
    )
    .first()
)

melhores_modelos = (
    melhores_modelos[
        [
            "metrica",
            "modelo",
            "n_previsoes",
            "mae",
            "rmse",
            "r2",
        ]
    ]
)

for _, row in melhores_modelos.iterrows():

    print(
        f"\n{row['metrica']}"
    )

    print(
        f"  Melhor modelo: "
        f"{row['modelo']}"
    )

    print(
        f"  MAE:  "
        f"{row['mae']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{row['rmse']:.6f}"
    )

    print(
        f"  R²:   "
        f"{row['r2']:.6f}"
    )


arquivo_melhores = (
    PASTA_SAIDA
    / "modelo4E_melhores_modelos.csv"
)

melhores_modelos.to_csv(
    arquivo_melhores,
    index=False
)

print(
    f"\n✓ {arquivo_melhores}"
)


# ==================================================================================================
# ROBUSTEZ — JUROS
# ==================================================================================================

titulo(
    "ROBUSTEZ — PREVISÃO DOS JUROS"
)

METRICA_JUROS = (
    "juros_cartao_pf_aa_pct"
)

juros_rob = (
    df_validacao[
        df_validacao["metrica"]
        == METRICA_JUROS
    ]
    .copy()
)

juros_rob[
    "erro_abs_persistencia"
] = np.abs(
    juros_rob[
        "valor_real_t1"
    ]
    - juros_rob[
        "persistencia"
    ]
)

juros_rob[
    "erro_abs_ridge"
] = np.abs(
    juros_rob[
        "valor_real_t1"
    ]
    - juros_rob[
        "ridge"
    ]
)

juros_rob[
    "erro_abs_mlp"
] = np.abs(
    juros_rob[
        "valor_real_t1"
    ]
    - juros_rob[
        "mlp"
    ]
)


def vencedor_linha(row):

    erros = {
        "persistencia":
            row[
                "erro_abs_persistencia"
            ],
        "ridge":
            row[
                "erro_abs_ridge"
            ],
        "mlp":
            row[
                "erro_abs_mlp"
            ],
    }

    return min(
        erros,
        key=erros.get
    )


juros_rob[
    "melhor_modelo_periodo"
] = juros_rob.apply(
    vencedor_linha,
    axis=1
)

juros_rob[
    "ganho_mlp_vs_persistencia"
] = (
    juros_rob[
        "erro_abs_persistencia"
    ]
    - juros_rob[
        "erro_abs_mlp"
    ]
)

juros_rob[
    "mlp_melhor_que_persistencia"
] = (
    juros_rob[
        "erro_abs_mlp"
    ]
    <
    juros_rob[
        "erro_abs_persistencia"
    ]
)

juros_rob[
    "persistencia_melhor_que_mlp"
] = (
    juros_rob[
        "erro_abs_persistencia"
    ]
    <
    juros_rob[
        "erro_abs_mlp"
    ]
)


n_juros = len(
    juros_rob
)

n_mlp_melhor = int(
    juros_rob[
        "mlp_melhor_que_persistencia"
    ]
    .sum()
)

n_persistencia_melhor = int(
    juros_rob[
        "persistencia_melhor_que_mlp"
    ]
    .sum()
)

n_empates = (
    n_juros
    - n_mlp_melhor
    - n_persistencia_melhor
)

print(
    f"\nNúmero de previsões: "
    f"{n_juros}"
)

print(
    "\nMLP × Persistência:"
)

print(
    f"  MLP menor erro: "
    f"{n_mlp_melhor}"
    f"/{n_juros} "
    f"({100 * n_mlp_melhor / n_juros:.2f}%)"
)

print(
    f"  Persistência menor erro: "
    f"{n_persistencia_melhor}"
    f"/{n_juros} "
    f"({100 * n_persistencia_melhor / n_juros:.2f}%)"
)

print(
    f"  Empates: "
    f"{n_empates}"
)

print(
    "\nVencedor entre os três modelos:"
)

contagem_vencedores = (
    juros_rob[
        "melhor_modelo_periodo"
    ]
    .value_counts()
)

for modelo in [
    "persistencia",
    "ridge",
    "mlp",
]:

    quantidade = int(
        contagem_vencedores.get(
            modelo,
            0
        )
    )

    percentual = (
        100
        * quantidade
        / n_juros
    )

    print(
        f"  {modelo:<15}"
        f"{quantidade:>3} "
        f"({percentual:.2f}%)"
    )


ganho_medio = (
    juros_rob[
        "ganho_mlp_vs_persistencia"
    ]
    .mean()
)

ganho_mediano = (
    juros_rob[
        "ganho_mlp_vs_persistencia"
    ]
    .median()
)

ganho_maximo = (
    juros_rob[
        "ganho_mlp_vs_persistencia"
    ]
    .max()
)

perda_maxima = (
    juros_rob[
        "ganho_mlp_vs_persistencia"
    ]
    .min()
)

print(
    "\nGanho MLP vs Persistência:"
)

print(
    f"  Médio:   "
    f"{ganho_medio:.6f} p.p."
)

print(
    f"  Mediano: "
    f"{ganho_mediano:.6f} p.p."
)

print(
    f"  Maior ganho: "
    f"{ganho_maximo:.6f} p.p."
)

print(
    f"  Maior perda: "
    f"{perda_maxima:.6f} p.p."
)


print(
    "\n5 maiores ganhos da MLP:"
)

top_ganhos = (
    juros_rob
    .sort_values(
        "ganho_mlp_vs_persistencia",
        ascending=False
    )
    .head(5)
)

for _, row in top_ganhos.iterrows():

    print(
        f"  {row['periodo_t1']} | "
        f"real={row['valor_real_t1']:.6f} | "
        f"persistência={row['persistencia']:.6f} | "
        f"MLP={row['mlp']:.6f} | "
        f"ganho={row['ganho_mlp_vs_persistencia']:.6f}"
    )


print(
    "\n5 maiores perdas da MLP:"
)

top_perdas = (
    juros_rob
    .sort_values(
        "ganho_mlp_vs_persistencia",
        ascending=True
    )
    .head(5)
)

for _, row in top_perdas.iterrows():

    print(
        f"  {row['periodo_t1']} | "
        f"real={row['valor_real_t1']:.6f} | "
        f"persistência={row['persistencia']:.6f} | "
        f"MLP={row['mlp']:.6f} | "
        f"ganho={row['ganho_mlp_vs_persistencia']:.6f}"
    )


arquivo_robustez = (
    PASTA_SAIDA
    / "modelo4E_robustez_juros.csv"
)

juros_rob.to_csv(
    arquivo_robustez,
    index=False
)

print(
    f"\n✓ {arquivo_robustez}"
)


# ==================================================================================================
# TESTE DE DIEBOLD-MARIANO — JUROS
# ==================================================================================================

titulo(
    "TESTE DE DIEBOLD-MARIANO — JUROS"
)

juros_dm = (
    df_validacao[
        df_validacao["metrica"]
        == METRICA_JUROS
    ]
    .copy()
)

juros_dm = juros_dm.dropna(
    subset=[
        "valor_real_t1",
        "mlp",
        "persistencia",
    ]
)

if len(
    juros_dm
) > 1:

    y_real_dm = (
        juros_dm[
            "valor_real_t1"
        ]
        .astype(float)
        .values
    )

    pred_mlp_dm = (
        juros_dm[
            "mlp"
        ]
        .astype(float)
        .values
    )

    pred_persistencia_dm = (
        juros_dm[
            "persistencia"
        ]
        .astype(float)
        .values
    )

    # ----------------------------------------------------------------------------------------------
    # PERDA QUADRÁTICA
    # ----------------------------------------------------------------------------------------------

    dm_sq, p_sq = (
        diebold_mariano_test(
            y_real_dm,
            pred_mlp_dm,
            pred_persistencia_dm,
            h=1,
            loss="squared",
        )
    )

    # ----------------------------------------------------------------------------------------------
    # PERDA ABSOLUTA
    # ----------------------------------------------------------------------------------------------

    dm_abs, p_abs = (
        diebold_mariano_test(
            y_real_dm,
            pred_mlp_dm,
            pred_persistencia_dm,
            h=1,
            loss="absolute",
        )
    )

    print(
        f"\nNúmero de previsões: "
        f"{len(juros_dm)}"
    )

    print(
        "\nMLP × Persistência"
    )

    print(
        "\nPerda quadrática:"
    )

    print(
        f"  Estatística DM: "
        f"{dm_sq:.6f}"
    )

    print(
        f"  p-valor:        "
        f"{p_sq:.6f}"
    )

    print(
        "\nPerda absoluta:"
    )

    print(
        f"  Estatística DM: "
        f"{dm_abs:.6f}"
    )

    print(
        f"  p-valor:        "
        f"{p_abs:.6f}"
    )

    # ----------------------------------------------------------------------------------------------
    # INTERPRETAÇÃO
    # ----------------------------------------------------------------------------------------------

    if np.isnan(
        p_sq
    ):

        interpretacao_sq = (
            "O teste com perda quadrática "
            "não pôde ser calculado."
        )

    elif p_sq < 0.05:

        if dm_sq < 0:

            interpretacao_sq = (
                "A MLP apresentou perda quadrática "
                "estatisticamente inferior à persistência "
                "ao nível de 5%."
            )

        else:

            interpretacao_sq = (
                "A persistência apresentou perda quadrática "
                "estatisticamente inferior à MLP "
                "ao nível de 5%."
            )

    else:

        interpretacao_sq = (
            "Não há evidência estatística suficiente, "
            "ao nível de 5%, de diferença na capacidade "
            "preditiva entre MLP e persistência considerando "
            "a perda quadrática."
        )


    if np.isnan(
        p_abs
    ):

        interpretacao_abs = (
            "O teste com perda absoluta "
            "não pôde ser calculado."
        )

    elif p_abs < 0.05:

        if dm_abs < 0:

            interpretacao_abs = (
                "A MLP apresentou perda absoluta "
                "estatisticamente inferior à persistência "
                "ao nível de 5%."
            )

        else:

            interpretacao_abs = (
                "A persistência apresentou perda absoluta "
                "estatisticamente inferior à MLP "
                "ao nível de 5%."
            )

    else:

        interpretacao_abs = (
            "Não há evidência estatística suficiente, "
            "ao nível de 5%, de diferença na capacidade "
            "preditiva entre MLP e persistência considerando "
            "a perda absoluta."
        )


    print(
        "\nInterpretação:"
    )

    print(
        "\n  RMSE / perda quadrática:"
    )

    print(
        f"  {interpretacao_sq}"
    )

    print(
        "\n  MAE / perda absoluta:"
    )

    print(
        f"  {interpretacao_abs}"
    )


    # ----------------------------------------------------------------------------------------------
    # EXPORTAÇÃO
    # ----------------------------------------------------------------------------------------------

    df_dm = pd.DataFrame(
        [
            {
                "metrica":
                    METRICA_JUROS,

                "comparacao":
                    "MLP_vs_Persistencia",

                "funcao_perda":
                    "quadratica",

                "n_previsoes":
                    len(
                        juros_dm
                    ),

                "estatistica_dm":
                    dm_sq,

                "p_valor":
                    p_sq,

                "significativo_5pct":
                    (
                        False
                        if np.isnan(
                            p_sq
                        )
                        else p_sq < 0.05
                    ),
            },

            {
                "metrica":
                    METRICA_JUROS,

                "comparacao":
                    "MLP_vs_Persistencia",

                "funcao_perda":
                    "absoluta",

                "n_previsoes":
                    len(
                        juros_dm
                    ),

                "estatistica_dm":
                    dm_abs,

                "p_valor":
                    p_abs,

                "significativo_5pct":
                    (
                        False
                        if np.isnan(
                            p_abs
                        )
                        else p_abs < 0.05
                    ),
            },
        ]
    )

    arquivo_dm = (
        PASTA_SAIDA
        / "modelo4E_diebold_mariano_juros.csv"
    )

    df_dm.to_csv(
        arquivo_dm,
        index=False
    )

    print(
        f"\n✓ {arquivo_dm}"
    )

else:

    print(
        "\nNúmero insuficiente de previsões "
        "para executar o teste."
    )


# ==================================================================================================
# PREVISÃO PROSPECTIVA
# ==================================================================================================

titulo(
    "PREVISÃO PROSPECTIVA — PRÓXIMO TRIMESTRE"
)

ultima_linha = (
    dados_modelo
    .iloc[[-1]]
    .copy()
)

data_atual = (
    ultima_linha[
        "date"
    ]
    .iloc[0]
)

periodo_atual = (
    data_atual
    .to_period("Q")
)

periodo_futuro = (
    periodo_atual
    + 1
)

print(
    f"\nTrimestre atual: "
    f"{periodo_atual}"
)

print(
    f"Previsão para: "
    f"{periodo_futuro}"
)

resultados_previsao = []


for metrica in METRICAS_ALVO:

    target_futuro = (
        f"{metrica}__t1"
    )

    treino_final = (
        dados_modelo[
            dados_modelo[
                target_futuro
            ].notna()
        ]
        .copy()
    )

    X_treino_raw = (
        treino_final[
            FEATURES
        ]
    )

    # ==============================================================================================
    # VETOR PROSPECTIVO COM A ÚLTIMA INFORMAÇÃO EFETIVAMENTE DISPONÍVEL
    #
    # Para cada feature, utilizamos o último valor não nulo observado até o
    # trimestre corrente. Isso evita tratar como observado um dado ainda não
    # publicado no trimestre atual.
    # ==============================================================================================

    valores_futuros = {}
    fontes_futuras = {}

    for feature in FEATURES:

        serie_valida = (
            dados_modelo[
                ["date", feature]
            ]
            .dropna(
                subset=[feature]
            )
        )

        if serie_valida.empty:

            raise ValueError(
                f"Nenhum valor disponível para a feature: {feature}"
            )

        ultima_obs_feature = (
            serie_valida
            .iloc[-1]
        )

        valores_futuros[feature] = float(
            ultima_obs_feature[feature]
        )

        fontes_futuras[feature] = (
            quarter_label(
                ultima_obs_feature["date"]
            )
        )

    X_futuro_raw = pd.DataFrame(
        [valores_futuros]
    )

    y_treino = (
        treino_final[
            target_futuro
        ]
        .astype(float)
        .values
    )

    # ----------------------------------------------------------------------------------------------
    # IMPUTAÇÃO
    # ----------------------------------------------------------------------------------------------

    imputador = SimpleImputer(
        strategy="median"
    )

    X_treino_imp = (
        imputador.fit_transform(
            X_treino_raw
        )
    )

    X_futuro_imp = (
        imputador.transform(
            X_futuro_raw
        )
    )

    # ----------------------------------------------------------------------------------------------
    # PADRONIZAÇÃO
    # ----------------------------------------------------------------------------------------------

    scaler = StandardScaler()

    X_treino = (
        scaler.fit_transform(
            X_treino_imp
        )
    )

    X_futuro = (
        scaler.transform(
            X_futuro_imp
        )
    )

    # ----------------------------------------------------------------------------------------------
    # PERSISTÊNCIA
    # ----------------------------------------------------------------------------------------------

    serie_metrica_valida = (
        dados_modelo[
            ["date", metrica]
        ]
        .dropna(
            subset=[metrica]
        )
    )

    ultima_obs_metrica = (
        serie_metrica_valida
        .iloc[-1]
    )

    valor_atual_metrica = float(
        ultima_obs_metrica[
            metrica
        ]
    )

    periodo_valor_atual = (
        quarter_label(
            ultima_obs_metrica[
                "date"
            ]
        )
    )

    pred_persistencia = (
        valor_atual_metrica
    )

    pred_persistencia = limitar_previsao(
        pred_persistencia,
        metrica
    )

    # ----------------------------------------------------------------------------------------------
    # RIDGE
    # ----------------------------------------------------------------------------------------------

    ridge = Ridge(
        alpha=1.0
    )

    ridge.fit(
        X_treino,
        y_treino
    )

    pred_ridge = float(
        ridge.predict(
            X_futuro
        )[0]
    )

    pred_ridge = limitar_previsao(
        pred_ridge,
        metrica
    )

    # ----------------------------------------------------------------------------------------------
    # MLP
    # ----------------------------------------------------------------------------------------------

    pred_mlp = treinar_mlp(
        X_treino,
        y_treino,
        X_futuro
    )

    pred_mlp = limitar_previsao(
        pred_mlp,
        metrica
    )

    # ----------------------------------------------------------------------------------------------
    # MODELO VENCEDOR
    # ----------------------------------------------------------------------------------------------

    linha_melhor = (
        melhores_modelos[
            melhores_modelos[
                "metrica"
            ]
            == metrica
        ]
        .iloc[0]
    )

    melhor_modelo = (
        linha_melhor[
            "modelo"
        ]
    )

    previsoes = {
        "persistencia":
            pred_persistencia,

        "ridge":
            pred_ridge,

        "mlp":
            pred_mlp,
    }

    previsao_selecionada = (
        previsoes[
            melhor_modelo
        ]
    )

    valor_atual = (
        valor_atual_metrica
    )

    resultados_previsao.append(
        {
            "metrica":
                metrica,

            "periodo_atual":
                str(
                    periodo_atual
                ),

            "periodo_ultimo_dado":
                str(
                    periodo_valor_atual
                ),

            "periodo_previsto":
                str(
                    periodo_futuro
                ),

            "valor_atual":
                valor_atual,

            "persistencia":
                pred_persistencia,

            "ridge":
                pred_ridge,

            "mlp":
                pred_mlp,

            "modelo_selecionado":
                melhor_modelo,

            "previsao_selecionada":
                previsao_selecionada,

            "mae_historico_modelo_selecionado":
                float(
                    linha_melhor[
                        "mae"
                    ]
                ),

            "rmse_historico_modelo_selecionado":
                float(
                    linha_melhor[
                        "rmse"
                    ]
                ),

            "r2_historico_modelo_selecionado":
                float(
                    linha_melhor[
                        "r2"
                    ]
                ),
        }
    )

    print(
        f"\n{metrica}"
    )

    print(
        f"  Último valor disponível: "
        f"{valor_atual:.6f}"
    )

    print(
        f"  Período da fonte:        "
        f"{periodo_valor_atual}"
    )

    print(
        f"  Persistência:    "
        f"{pred_persistencia:.6f}"
    )

    print(
        f"  Ridge:           "
        f"{pred_ridge:.6f}"
    )

    print(
        f"  MLP:             "
        f"{pred_mlp:.6f}"
    )

    print(
        f"  Melhor modelo:   "
        f"{melhor_modelo}"
    )

    print(
        f"  PREVISÃO FINAL:  "
        f"{previsao_selecionada:.6f}"
    )


df_previsao = pd.DataFrame(
    resultados_previsao
)


arquivo_previsao = (
    PASTA_SAIDA
    / "modelo4E_previsao_metricas.csv"
)

df_previsao.to_csv(
    arquivo_previsao,
    index=False
)

# ==================================================================================================
# MANIFESTO AUDITÁVEL DA EXECUÇÃO
# ==================================================================================================

import hashlib as _hashlib_4e
import json as _json_4e
import os as _os_4e
import tempfile as _tempfile_4e
import uuid as _uuid_4e
from datetime import datetime as _datetime_4e, timezone as _timezone_4e


def _sha256_arquivo_4e(caminho):
    digest = _hashlib_4e.sha256()

    with open(caminho, "rb") as entrada:
        for bloco_bytes in iter(
            lambda: entrada.read(1024 * 1024),
            b""
        ):
            digest.update(bloco_bytes)

    return digest.hexdigest()


_codigo_4e = Path(__file__).resolve()

_base_4e = (
    _codigo_4e.parent
    / "dataset_model_ready.csv"
)

if not _base_4e.is_file():
    raise FileNotFoundError(
        "Base de entrada não encontrada para auditoria: "
        f"{_base_4e}"
    )

if not arquivo_previsao.is_file():
    raise FileNotFoundError(
        "Snapshot não encontrado após gravação: "
        f"{arquivo_previsao}"
    )

_manifesto_4e = {
    "id_execucao": str(_uuid_4e.uuid4()),
    "data_execucao_utc": _datetime_4e.now(
        _timezone_4e.utc
    ).isoformat(),
    "modelo": "modelo4E_metricas",
    "hash_codigo": _sha256_arquivo_4e(_codigo_4e),
    "hash_dados": _sha256_arquivo_4e(_base_4e),
    "hash_snapshot": _sha256_arquivo_4e(
        arquivo_previsao
    ),
    "quantidade_previsoes": int(len(df_previsao)),
    "periodos_previstos": sorted(
        df_previsao["periodo_previsto"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    ),
    "arquivo_snapshot": arquivo_previsao.name,
    "arquivo_base": _base_4e.name,
}

_caminho_manifesto_4e = (
    PASTA_SAIDA
    / "modelo4E_manifesto_execucao.json"
)

_fd_4e, _temporario_4e = _tempfile_4e.mkstemp(
    prefix="modelo4E_manifesto_",
    suffix=".json.tmp",
    dir=str(PASTA_SAIDA)
)

try:
    with _os_4e.fdopen(
        _fd_4e,
        "w",
        encoding="utf-8"
    ) as _saida_4e:
        _json_4e.dump(
            _manifesto_4e,
            _saida_4e,
            ensure_ascii=False,
            indent=2
        )

        _saida_4e.flush()
        _os_4e.fsync(_saida_4e.fileno())

    _os_4e.replace(
        _temporario_4e,
        _caminho_manifesto_4e
    )

finally:
    if _os_4e.path.exists(_temporario_4e):
        _os_4e.unlink(_temporario_4e)

print(
    "\\n✓ Manifesto auditável:",
    _caminho_manifesto_4e
)


print(
    f"\n✓ {arquivo_previsao}"
)


# ==================================================================================================
# RESUMO FINAL
# ==================================================================================================

titulo(
    "RESUMO FINAL — MODELO 4E"
)

for _, row in df_previsao.iterrows():

    metrica = (
        row[
            "metrica"
        ]
    )

    modelo = (
        row[
            "modelo_selecionado"
        ]
    )

    previsao = float(
        row[
            "previsao_selecionada"
        ]
    )

    atual = float(
        row[
            "valor_atual"
        ]
    )

    if metrica == "juros_cartao_pf_aa_pct":

        nome = (
            "Juros cartão PF"
        )

        print(
            f"\n{nome}:"
        )

        print(
            f"  Atual:    "
            f"{atual:.2f}% a.a."
        )

        print(
            f"  Previsto: "
            f"{previsao:.2f}% a.a."
        )

    elif metrica == "inadimplencia_90d_pct":

        nome = (
            "Inadimplência"
        )

        print(
            f"\n{nome}:"
        )

        print(
            f"  Atual:    "
            f"{atual:.2f}%"
        )

        print(
            f"  Previsto: "
            f"{previsao:.2f}%"
        )

    elif metrica == "hhi_credito_0_1":

        nome = (
            "HHI"
        )

        print(
            f"\n{nome}:"
        )

        print(
            f"  Atual:    "
            f"{atual:.6f}"
        )

        print(
            f"  Previsto: "
            f"{previsao:.6f}"
        )

    elif metrica == "cr5_credito":

        nome = (
            "CR5"
        )

        print(
            f"\n{nome}:"
        )

        print(
            f"  Atual:    "
            f"{100 * atual:.2f}%"
        )

        print(
            f"  Previsto: "
            f"{100 * previsao:.2f}%"
        )

    print(
        f"  Modelo:   "
        f"{modelo}"
    )


print()

linha()

print(
    "ARQUIVOS GERADOS"
)

linha()

print(
    f"\n✓ {arquivo_validacao}"
)

print(
    f"✓ {arquivo_desempenho}"
)

print(
    f"✓ {arquivo_melhores}"
)

print(
    f"✓ {arquivo_robustez}"
)

if "arquivo_dm" in globals():

    print(
        f"✓ {arquivo_dm}"
    )

print(
    f"✓ {arquivo_previsao}"
)


print()

linha()

print(
    "FIM DO MODELO 4E"
)

linha()