# ============================================================
# MODELO 3C — MARKOV SWITCHING TRIMESTRAL
# + MÉTRICAS COMPETITIVAS
# + CONTEXTO MACROECONÔMICO / IBGE
#
# Objetivos:
# 1. Identificar regimes latentes do mercado de cartão
# 2. Comparar especificações com 2 e 3 regimes
# 3. Caracterizar economicamente os regimes
# 4. Integrar competição, risco, famílias e contexto macro
# 5. Gerar rótulos para o componente Neural posterior
#
# IMPORTANTE:
# O Modelo 3C NÃO realiza previsão futura.
#
# A previsão do regime t+1 será realizada posteriormente
# pelo componente Neural do framework híbrido Markov-Neural.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

ARQUIVO_CARTAO = "bcb_creditcard_competition_dataset_v4.csv"
ARQUIVO_IBGE = "ibge_macro_dataset.csv"

OUT_DIR = "outputs_modelo3C"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

print("\n" + "=" * 90)
print(
    "MODELO 3C — MARKOV SWITCHING TRIMESTRAL "
    "+ COMPETIÇÃO + IBGE"
)
print("=" * 90)


# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def encontrar_coluna_data(df):
    """
    Procura automaticamente uma coluna temporal.
    """

    candidatas = [
        "date",
        "data",
        "periodo",
        "period",
        "dt",
        "mes",
        "month"
    ]

    for col in candidatas:

        if col in df.columns:
            return col

    return None


def preparar_data(df, nome_base):
    """
    Padroniza a coluna temporal para 'date'.
    """

    coluna_data = encontrar_coluna_data(
        df
    )

    if coluna_data is None:

        raise ValueError(
            f"Não encontrei uma coluna temporal em {nome_base}.\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    if coluna_data != "date":

        df = df.rename(
            columns={
                coluna_data: "date"
            }
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
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def converter_numericas(df, ignorar=None):
    """
    Tenta transformar colunas textuais em numéricas,
    considerando inclusive padrão brasileiro.
    """

    ignorar = ignorar or []

    for col in df.columns:

        if col in ignorar:
            continue

        if df[col].dtype != "object":
            continue

        original = df[col].copy()

        serie = (
            df[col]
            .astype(str)
            .str.replace(
                "R$",
                "",
                regex=False
            )
            .str.replace(
                "%",
                "",
                regex=False
            )
            .str.strip()
        )

        convertido = pd.to_numeric(
            serie,
            errors="coerce"
        )

        # Se a conversão direta falhar,
        # tenta padrão brasileiro: 1.234,56
        if convertido.notna().mean() < 0.50:

            serie_br = (
                serie
                .str.replace(
                    ".",
                    "",
                    regex=False
                )
                .str.replace(
                    ",",
                    ".",
                    regex=False
                )
            )

            convertido_br = pd.to_numeric(
                serie_br,
                errors="coerce"
            )

            if (
                convertido_br.notna().sum()
                >
                convertido.notna().sum()
            ):
                convertido = convertido_br

        if convertido.notna().mean() >= 0.50:

            df[col] = convertido

        else:

            df[col] = original

    return df


def trimestralizar_media(df):
    """
    Converte base mensal para frequência trimestral.

    A média trimestral é utilizada para preservar
    consistência com as versões anteriores do Modelo 3C.

    IMPORTANTE:
    somente trimestres calendário completamente encerrados
    são mantidos na base final.

    Exemplo:
    em 07/09/2026, 2026Q3 ainda está em andamento,
    portanto o último trimestre aceito é 2026Q2.
    """

    temp = df.copy()

    temp["date"] = pd.to_datetime(
        temp["date"],
        errors="coerce"
    )

    temp = (
        temp
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    temp["quarter"] = (
        temp["date"]
        .dt.to_period("Q")
    )

    # ========================================================
    # CORTE DE TRIMESTRE COMPLETO
    # ========================================================

    hoje = pd.Timestamp.today().normalize()

    trimestre_atual = hoje.to_period("Q")

    ultimo_trimestre_completo = (
        trimestre_atual - 1
    )

    # Remove trimestre corrente e qualquer período futuro
    temp = temp[
        temp["quarter"]
        <= ultimo_trimestre_completo
    ].copy()

    # ========================================================
    # AGREGAÇÃO TRIMESTRAL
    # ========================================================

    numericas = (
        temp
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    tri = (
        temp
        .groupby(
            "quarter"
        )[numericas]
        .mean()
        .reset_index()
    )

    tri["date"] = (
        tri["quarter"]
        .dt.to_timestamp(
            how="end"
        )
        .dt.normalize()
    )

    return tri


def obter_probabilidades(
    resultado,
    n_regimes
):
    """
    Extrai probabilidades suavizadas do Markov.
    """

    prob = (
        resultado
        .smoothed_marginal_probabilities
    )

    if isinstance(
        prob,
        pd.DataFrame
    ):

        probs = prob.copy()

        probs.columns = [
            f"prob_regime_{i}"
            for i in range(n_regimes)
        ]

        return probs.reset_index(
            drop=True
        )

    arr = np.asarray(
        prob
    )

    if arr.ndim != 2:

        raise ValueError(
            "Formato inesperado das probabilidades suavizadas."
        )

    if arr.shape[1] == n_regimes:

        return pd.DataFrame(
            arr,
            columns=[
                f"prob_regime_{i}"
                for i in range(n_regimes)
            ]
        )

    if arr.shape[0] == n_regimes:

        return pd.DataFrame(
            arr.T,
            columns=[
                f"prob_regime_{i}"
                for i in range(n_regimes)
            ]
        )

    raise ValueError(
        "Não consegui interpretar as probabilidades. "
        f"Formato recebido: {arr.shape}"
    )


def obter_matriz_transicao(
    resultado,
    n_regimes
):
    """
    Extrai a matriz de transição.
    """

    matriz = np.asarray(
        resultado.regime_transition
    )

    if matriz.ndim == 3:

        matriz = matriz[:, :, 0]

    if matriz.shape != (
        n_regimes,
        n_regimes
    ):

        raise ValueError(
            "Formato inesperado da matriz "
            f"de transição: {matriz.shape}"
        )

    return matriz


def tentar_carregar_csv(nome):
    """
    Carrega CSV apenas se existir.
    """

    if not os.path.exists(nome):
        return None

    try:

        return pd.read_csv(
            nome
        )

    except Exception as erro:

        print(
            f"Aviso: não foi possível ler {nome}: {erro}"
        )

        return None


def preparar_base_auxiliar(
    nome_arquivo
):
    """
    Carrega e trimestraliza uma base auxiliar.
    """

    df = tentar_carregar_csv(
        nome_arquivo
    )

    if df is None:
        return None

    coluna_data = encontrar_coluna_data(
        df
    )

    if coluna_data is None:
        return None

    df = preparar_data(
        df,
        nome_arquivo
    )

    df = converter_numericas(
        df,
        ignorar=["date"]
    )

    return trimestralizar_media(
        df
    )


def adicionar_variaveis_auxiliares(
    base,
    arquivo,
    candidatas
):
    """
    Integra variáveis de uma base auxiliar.

    Diferença importante desta versão:
    NÃO deixa cadeias de colunas _x/_y/_aux.

    Cada variável é integrada individualmente e
    imediatamente consolidada.
    """

    aux = preparar_base_auxiliar(
        arquivo
    )

    if aux is None:
        return base

    encontradas = [
        col
        for col in candidatas
        if col in aux.columns
    ]

    if not encontradas:
        return base

    for col in encontradas:

        temp_nome = (
            f"__aux__{col}"
        )

        temp = aux[
            [
                "quarter",
                col
            ]
        ].copy()

        temp = temp.rename(
            columns={
                col: temp_nome
            }
        )

        base = pd.merge(
            base,
            temp,
            on="quarter",
            how="left"
        )

        base[temp_nome] = pd.to_numeric(
            base[temp_nome],
            errors="coerce"
        )

        if col not in base.columns:

            base[col] = base[
                temp_nome
            ]

        else:

            base[col] = pd.to_numeric(
                base[col],
                errors="coerce"
            )

            base[col] = (
                base[col]
                .combine_first(
                    base[temp_nome]
                )
            )

        base = base.drop(
            columns=[
                temp_nome
            ]
        )

    return base


def consolidar_variavel_merge(
    df,
    nome
):
    """
    Segurança adicional para versões antigas
    eventualmente existentes na base:

    nome
    nome_x
    nome_y
    nome_aux

    Consolida todas em uma única variável.
    """

    candidatas = [
        nome,
        f"{nome}_y",
        f"{nome}_aux",
        f"{nome}_x"
    ]

    existentes = [
        col
        for col in candidatas
        if col in df.columns
    ]

    if not existentes:
        return df

    consolidada = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64"
    )

    for col in existentes:

        serie = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        consolidada = (
            consolidada
            .combine_first(
                serie
            )
        )

    df[nome] = consolidada

    remover = [
        col
        for col in [
            f"{nome}_x",
            f"{nome}_y",
            f"{nome}_aux"
        ]
        if col in df.columns
    ]

    if remover:

        df = df.drop(
            columns=remover
        )

    return df


# ============================================================
# 3. BASE PRINCIPAL — CARTÃO
# ============================================================

if not os.path.exists(
    ARQUIVO_CARTAO
):

    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_CARTAO}"
    )

cartao = pd.read_csv(
    ARQUIVO_CARTAO
)

cartao = preparar_data(
    cartao,
    "base de cartão"
)

cartao = converter_numericas(
    cartao,
    ignorar=["date"]
)

print("\nBASE PRINCIPAL — CARTÃO")
print("-" * 90)

print(
    f"Linhas: {len(cartao)}"
)

print(
    f"Colunas: {len(cartao.columns)}"
)

print(
    f"Período: "
    f"{cartao['date'].min().strftime('%Y-%m')} até "
    f"{cartao['date'].max().strftime('%Y-%m')}"
)

cartao_tri = trimestralizar_media(
    cartao
)


# ============================================================
# 4. BASE IBGE
# ============================================================

if not os.path.exists(
    ARQUIVO_IBGE
):

    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_IBGE}"
    )

ibge = pd.read_csv(
    ARQUIVO_IBGE
)

ibge = preparar_data(
    ibge,
    "base IBGE"
)

ibge = converter_numericas(
    ibge,
    ignorar=["date"]
)

print("\nBASE IBGE")
print("-" * 90)

print(
    f"Linhas: {len(ibge)}"
)

print(
    f"Colunas: {len(ibge.columns)}"
)

ibge_tri = trimestralizar_media(
    ibge
)

ibge_cols = [
    col
    for col in ibge_tri.columns
    if col not in [
        "quarter",
        "date"
    ]
]

ibge_tri = ibge_tri.rename(
    columns={
        col: f"ibge_{col}"
        for col in ibge_cols
    }
)


# ============================================================
# 5. INTEGRAÇÃO CARTÃO + IBGE
# ============================================================

base = pd.merge(
    cartao_tri,
    ibge_tri.drop(
        columns=["date"]
    ),
    on="quarter",
    how="left"
)


# ============================================================
# 6. MÉTRICAS DE CONCENTRAÇÃO
# ============================================================

arquivos_concentracao = [

    "cr5_hhi_ajustado_completo.csv",
    "cr5_hhi_ajustado.csv",
    "cr5_hhi_final.csv",
    "metricas_concentracao.csv",
    "hhi_credito_total.csv",
    "cr5_credito_total.csv",
    "hhi_credito_classificado.csv",
    "cr5_credito_classificado.csv",
    "cr5_credito_por_periodo.csv"
]

candidatas_concentracao = [

    "hhi_credito_0_1",
    "hhi_credito_0_10000",
    "hhi_credito",

    "hhi",
    "HHI",

    "cr5_credito",
    "cr5",
    "CR5",

    "hhi_cartao",
    "cr5_cartao"
]

for arquivo in arquivos_concentracao:

    base = adicionar_variaveis_auxiliares(
        base,
        arquivo,
        candidatas_concentracao
    )


# ============================================================
# 7. SHARES DE FINTECHS / BANCOS TRADICIONAIS
# ============================================================

arquivos_shares = [

    "shares_por_tipo_instituicao.csv",
    "shares_servicos_trimestral_2015_2025.csv",
    "shares_servicos_2015_2025.csv",
    "ifdata_features_competicao.csv"
    'outputs_competicao_auditada/shares_institucionais_dashboard.csv',
]

candidatas_shares = [

    "share_fintechs",
    "share_bancos_tradicionais",

    "share_fintech",
    "fintech_share",

    "share_tradicionais"
]

for arquivo in arquivos_shares:

    base = adicionar_variaveis_auxiliares(
        base,
        arquivo,
        candidatas_shares
    )


# ============================================================
# 8. CONSOLIDAÇÃO FINAL DAS MÉTRICAS COMPETITIVAS
# ============================================================

variaveis_para_consolidar = [

    "hhi_credito_0_1",
    "hhi_credito_0_10000",
    "hhi_credito",

    "hhi",
    "cr5_credito",
    "cr5",

    "hhi_cartao",
    "cr5_cartao",

    "share_fintechs",
    "share_bancos_tradicionais",

    "share_fintech",
    "fintech_share",
    "share_tradicionais"
]

for nome in variaveis_para_consolidar:

    base = consolidar_variavel_merge(
        base,
        nome
    )


# ============================================================
# 8B. PADRONIZAÇÃO DOS SHARES
# ============================================================

if (
    "share_fintechs" not in base.columns
    or base["share_fintechs"].notna().sum() == 0
):

    for alternativa in [
        "share_fintech",
        "fintech_share"
    ]:

        if (
            alternativa in base.columns
            and
            base[alternativa].notna().sum() > 0
        ):

            base[
                "share_fintechs"
            ] = pd.to_numeric(
                base[alternativa],
                errors="coerce"
            )

            break


if (
    "share_bancos_tradicionais"
    not in base.columns
    or
    base[
        "share_bancos_tradicionais"
    ].notna().sum() == 0
):

    if (
        "share_tradicionais" in base.columns
        and
        base[
            "share_tradicionais"
        ].notna().sum() > 0
    ):

        base[
            "share_bancos_tradicionais"
        ] = pd.to_numeric(
            base[
                "share_tradicionais"
            ],
            errors="coerce"
        )


base = (
    base
    .sort_values("quarter")
    .reset_index(drop=True)
)


# ============================================================
# 8C. CONFERÊNCIA DAS MÉTRICAS COMPETITIVAS
# ============================================================

print("\n" + "=" * 90)
print(
    "CONFERÊNCIA DAS MÉTRICAS COMPETITIVAS CONSOLIDADAS"
)
print("=" * 90)

metricas_conferencia = [

    "hhi_credito_0_1",
    "hhi_credito_0_10000",
    "cr5_credito",

    "hhi_cartao",
    "cr5_cartao",

    "share_fintechs",
    "share_bancos_tradicionais"
]

for col in metricas_conferencia:

    if col not in base.columns:

        print(
            f" - {col}: NÃO ENCONTRADA"
        )

        continue

    serie = pd.to_numeric(
        base[col],
        errors="coerce"
    )

    n_validos = (
        serie
        .notna()
        .sum()
    )

    print(
        f" - {col}: {n_validos} valores válidos"
    )

    if n_validos > 0:

        print(
            f"      mínimo = {serie.min():.6f}"
        )

        print(
            f"      média  = {serie.mean():.6f}"
        )

        print(
            f"      máximo = {serie.max():.6f}"
        )


# ============================================================
# 9. DIAGNÓSTICO DA BASE INTEGRADA
# ============================================================

print("\n" + "=" * 90)
print(
    "DIAGNÓSTICO DA BASE INTEGRADA"
)
print("=" * 90)

print(
    f"Trimestres: {len(base)}"
)

print(
    f"Período: "
    f"{base['quarter'].min()} até "
    f"{base['quarter'].max()}"
)

print(
    "\nPrincipais variáveis competitivas encontradas:"
)

for col in base.columns:

    nome_lower = col.lower()

    if (
        "hhi" in nome_lower
        or
        "cr5" in nome_lower
        or
        "fintech" in nome_lower
        or
        "tradicion" in nome_lower
    ):

        print(
            f" - {col}: "
            f"{base[col].notna().sum()} valores válidos"
        )


# ============================================================
# 10. VARIÁVEL DE IDENTIFICAÇÃO DOS REGIMES
# ============================================================

CANDIDATAS_Y = [

    "spread_cartao_pf",
    "spread_cartao_pf_aa",
    "spread_cartao",

    "juros_cartao_pf_aa_pct"
]

Y = None

for col in CANDIDATAS_Y:

    if (
        col in base.columns
        and
        base[col].notna().sum() >= 20
    ):

        Y = col
        break


if Y is None:

    raise ValueError(
        "Não encontrei variável adequada "
        "para identificação dos regimes."
    )


print("\n" + "=" * 90)
print(
    "VARIÁVEL DE IDENTIFICAÇÃO"
)
print("=" * 90)

print(
    Y
)


# ============================================================
# 11. AMOSTRA DO MODELO
# ============================================================

dados = base.copy()

dados[Y] = pd.to_numeric(
    dados[Y],
    errors="coerce"
)

dados = dados.replace(
    [np.inf, -np.inf],
    np.nan
)

dados = (
    dados
    .dropna(
        subset=[Y]
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 90)
print(
    "AMOSTRA MODELO 3C"
)
print("=" * 90)

print(
    f"Observações trimestrais: {len(dados)}"
)

print(
    f"Período: "
    f"{dados['quarter'].min()} até "
    f"{dados['quarter'].max()}"
)

print(
    "\nEstatísticas da variável principal:"
)

print(
    dados[Y]
    .describe()
)


# ============================================================
# 12. PADRONIZAÇÃO
# ============================================================

media_y = (
    dados[Y]
    .mean()
)

desvio_y = (
    dados[Y]
    .std()
)

if (
    desvio_y == 0
    or
    np.isnan(desvio_y)
):

    raise ValueError(
        "Variável de identificação sem variabilidade."
    )


dados["y_z"] = (
    dados[Y] - media_y
) / desvio_y


# ============================================================
# 13. MARKOV — 2 REGIMES
# ============================================================

print("\n" + "=" * 90)
print(
    "MODELO 3C-A — 2 REGIMES"
)
print("=" * 90)


modelo2 = MarkovRegression(
    endog=dados["y_z"],
    k_regimes=2,
    trend="c",
    switching_variance=True
)


resultado2 = modelo2.fit(
    search_reps=50,
    em_iter=10,
    maxiter=5000,
    disp=False
)


print(
    resultado2.summary()
)


# ============================================================
# 14. MARKOV — 3 REGIMES
# ============================================================

print("\n" + "=" * 90)
print(
    "MODELO 3C-B — 3 REGIMES"
)
print("=" * 90)


modelo3 = MarkovRegression(
    endog=dados["y_z"],
    k_regimes=3,
    trend="c",
    switching_variance=True
)


resultado3 = modelo3.fit(
    search_reps=100,
    em_iter=20,
    maxiter=10000,
    disp=False
)


print(
    resultado3.summary()
)


# ============================================================
# 15. COMPARAÇÃO 2 × 3 REGIMES
# ============================================================

comparacao = pd.DataFrame({

    "modelo": [
        "2 regimes",
        "3 regimes"
    ],

    "loglik": [
        resultado2.llf,
        resultado3.llf
    ],

    "AIC": [
        resultado2.aic,
        resultado3.aic
    ],

    "BIC": [
        resultado2.bic,
        resultado3.bic
    ],

    "HQIC": [
        resultado2.hqic,
        resultado3.hqic
    ]
})


print("\n" + "=" * 90)
print(
    "COMPARAÇÃO 2 × 3 REGIMES"
)
print("=" * 90)

print(
    comparacao
    .round(4)
    .to_string(
        index=False
    )
)


comparacao.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_comparacao_regimes.csv"
    ),
    index=False
)


# ============================================================
# 16. PROBABILIDADES SUAVIZADAS
# ============================================================

probs = obter_probabilidades(
    resultado3,
    3
)

dados = pd.concat(
    [
        dados.reset_index(
            drop=True
        ),
        probs
    ],
    axis=1
)


dados["regime_estimado"] = (
    dados[
        [
            "prob_regime_0",
            "prob_regime_1",
            "prob_regime_2"
        ]
    ]
    .values
    .argmax(
        axis=1
    )
)


# ============================================================
# 17. PERFIL DA VARIÁVEL PRINCIPAL
# ============================================================

estatisticas_regimes = (
    dados
    .groupby(
        "regime_estimado"
    )[Y]
    .agg(
        media="mean",
        mediana="median",
        minimo="min",
        maximo="max",
        desvio="std",
        observacoes="count"
    )
)


print("\n" + "=" * 90)
print(
    "VARIÁVEL PRINCIPAL POR REGIME"
)
print("=" * 90)

print(
    estatisticas_regimes
    .round(4)
)


# ============================================================
# 18. ORDENAÇÃO DESCRITIVA
# ============================================================

ordem = (
    estatisticas_regimes[
        "media"
    ]
    .sort_values()
    .index
    .tolist()
)


if len(ordem) != 3:

    raise ValueError(
        "O modelo de 3 regimes não produziu "
        "três estados observados."
    )


rotulos = {

    ordem[0]:
        "Nível baixo",

    ordem[1]:
        "Nível intermediário",

    ordem[2]:
        "Nível alto"
}


dados[
    "nivel_variavel_principal"
] = (
    dados[
        "regime_estimado"
    ]
    .map(
        rotulos
    )
)


print(
    "\nOrdenação descritiva:"
)


for regime in ordem:

    print(
        f"Regime {regime}: "
        f"{rotulos[regime]} | "
        f"média = "
        f"{estatisticas_regimes.loc[regime, 'media']:.4f}"
    )


# ============================================================
# 19. MATRIZ DE TRANSIÇÃO
# ============================================================

matriz = obter_matriz_transicao(
    resultado3,
    3
)


matriz_df = pd.DataFrame(

    matriz,

    index=[
        "Destino_R0",
        "Destino_R1",
        "Destino_R2"
    ],

    columns=[
        "Origem_R0",
        "Origem_R1",
        "Origem_R2"
    ]
)


print("\n" + "=" * 90)
print(
    "MATRIZ DE TRANSIÇÃO"
)
print("=" * 90)

print(
    matriz_df
    .round(5)
)


matriz_df.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_matriz_transicao.csv"
    )
)


# ============================================================
# 20. DURAÇÃO ESPERADA
# ============================================================

duracoes = []


print("\n" + "=" * 90)
print(
    "DURAÇÃO ESPERADA DOS REGIMES"
)
print("=" * 90)


for regime in range(3):

    p = matriz[
        regime,
        regime
    ]

    duracao = (
        1 / (1 - p)
        if p < 0.999999
        else np.inf
    )

    print(
        f"Regime {regime}: "
        f"p permanência = {p:.4f} | "
        f"duração ≈ {duracao:.2f} trimestres"
    )

    duracoes.append({

        "regime":
            regime,

        "p_permanencia":
            p,

        "duracao_esperada_trimestres":
            duracao
    })


pd.DataFrame(
    duracoes
).to_csv(

    os.path.join(
        OUT_DIR,
        "modelo3C_duracao_regimes.csv"
    ),

    index=False
)


# ============================================================
# 21. MÉTRICAS DE MERCADO
# ============================================================

candidatas_mercado = [

    # Preço do crédito
    "juros_cartao_pf_aa_pct",
    "spread_cartao_pf",
    "spread_cartao_pf_aa",
    "selic_am_pct",
    "selic_aa_pct",

    # Risco
    "inadimplencia_90d_pct",

    # Tamanho / composição
    "saldo_cartao_pf_total_mr",
    "saldo_rotativo_mr",
    "saldo_parcelado_mr",
    "prop_rotativo",

    # Competição
    "hhi_credito_0_1",
    "hhi_credito_0_10000",
    "cr5_credito",
    "hhi_credito",
    "hhi",
    "cr5",
    "hhi_cartao",
    "cr5_cartao",

    "share_fintechs",
    "share_bancos_tradicionais",

    # Famílias
    "endividamento_pct",
    "comprometimento_renda_pct"
]


# ESTA PARTE ESTAVA FALTANDO NA SUA VERSÃO

variaveis_mercado = [

    col

    for col in candidatas_mercado

    if (
        col in dados.columns
        and
        dados[col].notna().sum() > 0
    )
]


# ============================================================
# 22. MÉTRICAS IBGE
# ============================================================

variaveis_ibge = [

    col

    for col in dados.columns

    if (
        col.startswith(
            "ibge_"
        )
        and
        dados[col].notna().sum() > 0
    )
]


print("\n" + "=" * 90)
print(
    "MÉTRICAS DE MERCADO UTILIZADAS NA CARACTERIZAÇÃO"
)
print("=" * 90)


for col in variaveis_mercado:

    print(
        f" - {col}: "
        f"{dados[col].notna().sum()} observações"
    )


print("\n" + "=" * 90)
print(
    "MÉTRICAS IBGE UTILIZADAS"
)
print("=" * 90)


for col in variaveis_ibge:

    print(
        f" - {col}: "
        f"{dados[col].notna().sum()} observações"
    )


# ============================================================
# 23. PERFIL ECONÔMICO COMPLETO
# ============================================================

variaveis_perfil = list(
    dict.fromkeys(
        variaveis_mercado
        +
        variaveis_ibge
    )
)


for col in variaveis_perfil:

    dados[col] = pd.to_numeric(
        dados[col],
        errors="coerce"
    )


perfil_media = (
    dados
    .groupby(
        "regime_estimado"
    )[variaveis_perfil]
    .mean()
    .T
)


perfil_media.columns = [
    f"Regime_{col}"
    for col
    in perfil_media.columns
]


perfil_mediana = (
    dados
    .groupby(
        "regime_estimado"
    )[variaveis_perfil]
    .median()
    .T
)


perfil_mediana.columns = [
    f"Regime_{col}"
    for col
    in perfil_mediana.columns
]


print("\n" + "=" * 90)
print(
    "PERFIL ECONÔMICO + IBGE — MÉDIAS"
)
print("=" * 90)

print(
    perfil_media
    .round(4)
    .to_string()
)


perfil_media.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_perfil_regimes_media.csv"
    )
)


perfil_mediana.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_perfil_regimes_mediana.csv"
    )
)


# ============================================================
# 24. COBERTURA DAS VARIÁVEIS POR REGIME
# ============================================================

cobertura = (
    dados
    .groupby(
        "regime_estimado"
    )[variaveis_perfil]
    .count()
    .T
)


cobertura.columns = [
    f"Regime_{col}"
    for col
    in cobertura.columns
]


print("\n" + "=" * 90)
print(
    "COBERTURA DAS VARIÁVEIS POR REGIME"
)
print("=" * 90)

print(
    cobertura
    .to_string()
)


cobertura.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_cobertura_variaveis_regime.csv"
    )
)


# ============================================================
# 25. MUDANÇAS HISTÓRICAS DE REGIME
# ============================================================

dados[
    "mudanca_regime"
] = (
    dados[
        "regime_estimado"
    ]
    !=
    dados[
        "regime_estimado"
    ].shift(1)
)


mudancas = dados.loc[

    dados[
        "mudanca_regime"
    ],

    [
        "date",
        "quarter",
        Y,

        "regime_estimado",

        "nivel_variavel_principal",

        "prob_regime_0",
        "prob_regime_1",
        "prob_regime_2"
    ]

].copy()


print("\n" + "=" * 90)
print(
    "MUDANÇAS HISTÓRICAS DE REGIME"
)
print("=" * 90)

print(
    mudancas
    .to_string(
        index=False
    )
)


mudancas.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_mudancas_regime.csv"
    ),
    index=False
)


# ============================================================
# 26. PRÉ-COVID / COVID / PÓS-COVID
# ============================================================

def classificar_periodo(
    data
):

    if data < pd.Timestamp(
        "2020-03-01"
    ):

        return "Pré-Covid"

    if data <= pd.Timestamp(
        "2021-12-31"
    ):

        return "Covid"

    return "Pós-Covid"


dados[
    "periodo_macro"
] = (
    dados[
        "date"
    ]
    .apply(
        classificar_periodo
    )
)


regime_periodo = pd.crosstab(

    dados[
        "periodo_macro"
    ],

    dados[
        "regime_estimado"
    ],

    normalize="index"

) * 100


print("\n" + "=" * 90)
print(
    "REGIME × PERÍODO MACROECONÔMICO"
)
print("=" * 90)

print(
    regime_periodo
    .round(2)
)


regime_periodo.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_regime_periodo_macro.csv"
    )
)


# ============================================================
# 27. DISTRIBUIÇÃO DOS REGIMES
# ============================================================

distribuicao = (
    dados[
        "regime_estimado"
    ]
    .value_counts()
    .sort_index()
    .to_frame(
        "observacoes"
    )
)


distribuicao[
    "percentual"
] = (
    distribuicao[
        "observacoes"
    ]
    /
    len(dados)
    *
    100
)


print("\n" + "=" * 90)
print(
    "DISTRIBUIÇÃO DOS REGIMES"
)
print("=" * 90)

print(
    distribuicao
    .round(2)
)


distribuicao.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_distribuicao_regimes.csv"
    )
)


# ============================================================
# 28. GRÁFICOS DAS PROBABILIDADES SUAVIZADAS
# ============================================================

for regime in range(3):

    plt.figure(
        figsize=(
            14,
            5
        )
    )

    plt.plot(
        dados[
            "date"
        ],
        dados[
            f"prob_regime_{regime}"
        ],
        linewidth=2
    )

    plt.ylim(
        0,
        1
    )

    plt.xlabel(
        "Data"
    )

    plt.ylabel(
        "Probabilidade suavizada"
    )

    plt.title(
        f"Modelo 3C — "
        f"Probabilidade suavizada do Regime {regime}"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"modelo3C_prob_regime_{regime}.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 29. GRÁFICO DA VARIÁVEL PRINCIPAL
# ============================================================

plt.figure(
    figsize=(
        14,
        6
    )
)


plt.plot(
    dados[
        "date"
    ],
    dados[Y],
    linewidth=2
)


plt.xlabel(
    "Data"
)


plt.ylabel(
    Y
)


plt.title(
    "Modelo 3C — Evolução da variável de identificação"
)


plt.grid(
    alpha=0.2
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUT_DIR,
        "modelo3C_serie_variavel_principal.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# 30. BASE PARA A REDE NEURAL
# ============================================================
#
# A Neural terá como objetivo:
#
# variáveis observadas em t
#          ↓
# prever Regime(t+1)
#
# IMPORTANTE
# ----------
# As probabilidades suavizadas do Markov são exportadas
# apenas para auditoria.
#
# NÃO deverão ser utilizadas como features diretamente,
# porque as probabilidades suavizadas utilizam informação
# de toda a série e podem gerar leakage.
#
# O regime_estimado também será preservado como informação
# de auditoria. A decisão sobre usá-lo no componente Neural
# deverá considerar esse mesmo problema.
# ============================================================

base_neural = (
    dados.copy()
)


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

base_neural[
    "regime_proximo"
] = (
    base_neural[
        "regime_estimado"
    ]
    .shift(-1)
)


base_neural[
    "regime_proximo"
] = pd.to_numeric(
    base_neural[
        "regime_proximo"
    ],
    errors="coerce"
)


# ------------------------------------------------------------
# IDENTIFICADORES / AUDITORIA
# ------------------------------------------------------------

colunas_neural = [

    "date",
    "quarter",

    "regime_estimado",

    "regime_proximo",

    "prob_regime_0",
    "prob_regime_1",
    "prob_regime_2"
]


# ------------------------------------------------------------
# VARIÁVEIS ECONÔMICAS
# ------------------------------------------------------------

for col in variaveis_perfil:

    if col not in colunas_neural:

        colunas_neural.append(
            col
        )


base_neural_export = (
    base_neural[
        colunas_neural
    ]
    .copy()
)


# ------------------------------------------------------------
# BASE COMPLETA
# ------------------------------------------------------------

base_neural_export.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_base_para_neural.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# BASE COM TARGET DISPONÍVEL
# ------------------------------------------------------------

base_neural_treino = (
    base_neural_export
    .dropna(
        subset=[
            "regime_proximo"
        ]
    )
    .copy()
)


base_neural_treino[
    "regime_proximo"
] = (
    base_neural_treino[
        "regime_proximo"
    ]
    .astype(int)
)


base_neural_treino.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_base_para_neural_treino.csv"
    ),
    index=False
)


print("\n" + "=" * 90)
print(
    "BASE PARA O COMPONENTE NEURAL"
)
print("=" * 90)


print(
    f"Base completa: "
    f"{len(base_neural_export)} observações"
)


print(
    f"Base com alvo para treino: "
    f"{len(base_neural_treino)} observações"
)


print(
    "\nIMPORTANTE:"
)


print(
    " - prob_regime_0/1/2 = somente auditoria."
)


print(
    " - regime_estimado = rótulo ex-post do Markov."
)


print(
    " - Essas colunas não devem ser automaticamente "
    "usadas como features da rede Neural."
)


# ============================================================
# 31. EXPORTAR BASE COMPLETA DO MARKOV
# ============================================================

dados.to_csv(
    os.path.join(
        OUT_DIR,
        "modelo3C_base_regimes_completa.csv"
    ),
    index=False
)


# ============================================================
# 32. RESUMO FINAL
# ============================================================

print("\n" + "=" * 90)
print(
    "MODELO 3C CONCLUÍDO"
)
print("=" * 90)


print(
    "\nVariável de identificação:"
)

print(
    Y
)


print(
    "\nFrequência:"
)

print(
    "Trimestral"
)


print(
    "\nObservações:"
)

print(
    len(dados)
)


print(
    "\nPeríodo:"
)

print(
    dados[
        "quarter"
    ].min(),
    "até",
    dados[
        "quarter"
    ].max()
)


print("\n" + "-" * 90)
print(
    "COMPARAÇÃO DOS MODELOS"
)
print("-" * 90)


print(
    "\n2 REGIMES"
)

print(
    f"AIC: {resultado2.aic:.4f}"
)

print(
    f"BIC: {resultado2.bic:.4f}"
)


print(
    "\n3 REGIMES"
)

print(
    f"AIC: {resultado3.aic:.4f}"
)

print(
    f"BIC: {resultado3.bic:.4f}"
)


delta_aic = (
    resultado2.aic
    -
    resultado3.aic
)

delta_bic = (
    resultado2.bic
    -
    resultado3.bic
)


print(
    f"\nΔAIC (2 - 3): {delta_aic:.4f}"
)

print(
    f"ΔBIC (2 - 3): {delta_bic:.4f}"
)


print("\n" + "-" * 90)
print(
    "FUNÇÃO DO MODELO 3C"
)
print("-" * 90)


print(
    "\nO Modelo 3C identifica e caracteriza "
    "regimes latentes."
)


print(
    "Ele NÃO realiza previsão futura."
)


print(
    "A previsão do regime seguinte ficará "
    "para o componente Neural."
)


print("\n" + "-" * 90)
print(
    "BASE PARA A REDE NEURAL"
)
print("-" * 90)


print(
    "\nArquivo completo:"
)

print(
    os.path.join(
        OUT_DIR,
        "modelo3C_base_para_neural.csv"
    )
)


print(
    "\nArquivo de treinamento:"
)

print(
    os.path.join(
        OUT_DIR,
        "modelo3C_base_para_neural_treino.csv"
    )
)


print(
    "\nregime_estimado = "
    "estado identificado pelo Markov no trimestre t."
)


print(
    "regime_proximo = "
    "estado Markov correspondente ao trimestre t+1."
)


print("\n" + "-" * 90)
print(
    "ARQUIVOS GERADOS"
)
print("-" * 90)


arquivos_gerados = [

    "modelo3C_comparacao_regimes.csv",

    "modelo3C_matriz_transicao.csv",

    "modelo3C_duracao_regimes.csv",

    "modelo3C_perfil_regimes_media.csv",

    "modelo3C_perfil_regimes_mediana.csv",

    "modelo3C_cobertura_variaveis_regime.csv",

    "modelo3C_mudancas_regime.csv",

    "modelo3C_regime_periodo_macro.csv",

    "modelo3C_distribuicao_regimes.csv",

    "modelo3C_base_regimes_completa.csv",

    "modelo3C_base_para_neural.csv",

    "modelo3C_base_para_neural_treino.csv",

    "modelo3C_prob_regime_0.png",

    "modelo3C_prob_regime_1.png",

    "modelo3C_prob_regime_2.png",

    "modelo3C_serie_variavel_principal.png"
]


print(
    f"\nPasta: {OUT_DIR}/\n"
)


for arquivo in arquivos_gerados:

    caminho = os.path.join(
        OUT_DIR,
        arquivo
    )

    status = (
        "OK"
        if os.path.exists(caminho)
        else "NÃO GERADO"
    )

    print(
        f" - {arquivo}: {status}"
    )


print("\n" + "-" * 90)
print(
    "INTERPRETAÇÃO"
)
print("-" * 90)


print(
    "\nA numeração dos regimes é estatística."
)


print(
    '"Nível baixo", "Nível intermediário" e '
    '"Nível alto" referem-se ao nível da '
    "variável utilizada para identificação."
)


print(
    "\nA interpretação econômica definitiva "
    "deve considerar conjuntamente:"
)


itens_interpretacao = [

    "juros / spread",

    "inadimplência",

    "HHI",

    "CR5",

    "share de fintechs",

    "share de bancos tradicionais",

    "composição do crédito",

    "endividamento / comprometimento de renda",

    "indicadores macroeconômicos do IBGE"
]


for item in itens_interpretacao:

    print(
        " -",
        item
    )


print("\n" + "=" * 90)
print(
    "FIM DO MODELO 3C"
)
print("=" * 90)