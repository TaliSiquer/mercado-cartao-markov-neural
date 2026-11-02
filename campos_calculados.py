import pandas as pd
import numpy as np

# ============================================================
# CARREGAR DATASET
# ============================================================

df = pd.read_csv("dataset_final_dissertacao.csv")

# Garantir ordem temporal
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 1. CRESCIMENTO LOGARÍTMICO DO MERCADO DE CARTÕES
# ============================================================

df["dlog_total_cartao"] = (
    np.log(df["total_cartao"])
    - np.log(df["total_cartao"].shift(1))
)


# ============================================================
# 2. VARIAÇÃO DO HHI DO MERCADO DE CARTÕES
# ============================================================

df["delta_hhi_cartao"] = (
    df["ihhn_cartao_0_1"]
    - df["ihhn_cartao_0_1"].shift(1)
)


# ============================================================
# 3. SPREAD DO CARTÃO PF EM RELAÇÃO À SELIC
# ============================================================
#
# Definição utilizada na dissertação:
#
# spread_cartao_pf =
# juros_cartao_pf_aa_pct - (selic_am_pct * 12)
#
# A Selic mensal é anualizada de forma aproximada multiplicando
# a taxa mensal por 12, mantendo a metodologia já definida
# no trabalho.
# ============================================================

df["spread_cartao_pf"] = (
    df["juros_cartao_pf_aa_pct"]
    - (df["selic_am_pct"] * 12)
)


# ============================================================
# 4. DIFERENCIAL ENTRE ROTATIVO E PARCELADO
# ============================================================
#
# Esta era a métrica anteriormente chamada de spread_cartao_pf.
# Ela continua sendo mantida na base, mas agora com um nome
# que representa corretamente o seu significado.
# ============================================================

df["diferencial_rotativo_parcelado"] = (
    df["juros_cartao_rotativo_pf_mensal_pct"]
    - df["juros_cartao_parcelado_pf_mensal_pct"]
)


# ============================================================
# 5. PROPORÇÃO DE CRÉDITO ROTATIVO
# ============================================================

df["prop_rotativo"] = (
    df["saldo_cartao_rotativo_pf_mr"]
    / df["saldo_cartao_pf_total_mr"]
)


# ============================================================
# 6. CRESCIMENTO DO SALDO TOTAL DE CARTÃO
# ============================================================

df["dlog_saldo_cartao_pf_total_mr"] = (
    np.log(df["saldo_cartao_pf_total_mr"])
    - np.log(df["saldo_cartao_pf_total_mr"].shift(1))
)


# ============================================================
# 7. VARIAÇÃO DA CONCENTRAÇÃO DO MERCADO DE CRÉDITO
# ============================================================

df["delta_hhi_credito"] = (
    df["hhi_credito_0_1"]
    - df["hhi_credito_0_1"].shift(1)
)


# ============================================================
# 8. ÍNDICE DE ESTRESSE FINANCEIRO DAS FAMÍLIAS
# ============================================================
#
# stress_familias =
# (endividamento * comprometimento da renda) / 100
#
# Já deixa a variável pronta para os próximos modelos.
# ============================================================

df["stress_familias"] = (
    df["endividamento_familias_pct_renda"]
    * df["comprometimento_renda_pct"]
) / 100


# ============================================================
# SALVAR DATASET MODEL-READY
# ============================================================

df.to_csv(
    "dataset_model_ready.csv",
    index=False
)


# ============================================================
# VALIDAÇÃO
# ============================================================

print("=" * 70)
print("DATASET MODEL-READY CRIADO COM SUCESSO")
print("=" * 70)

print("\nDimensão:", df.shape)

print("\nNovas métricas:")
print([
    "dlog_total_cartao",
    "delta_hhi_cartao",
    "spread_cartao_pf",
    "diferencial_rotativo_parcelado",
    "prop_rotativo",
    "dlog_saldo_cartao_pf_total_mr",
    "delta_hhi_credito",
    "stress_familias"
])

print("\nValidação do spread:")

print(
    df[
        [
            "date",
            "juros_cartao_pf_aa_pct",
            "selic_am_pct",
            "spread_cartao_pf",
            "diferencial_rotativo_parcelado"
        ]
    ].head(12)
)