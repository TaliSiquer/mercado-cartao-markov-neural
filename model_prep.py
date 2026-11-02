# ============================================================
# MODEL_PREP.PY
# Transformações para modelagem econométrica (robusto)
# ============================================================

import pandas as pd
import numpy as np

# ADF é opcional: se não tiver statsmodels instalado, o script ainda roda
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ------------------------------------------------------------
# 1) Carregar dataset original
# ------------------------------------------------------------

INPUT_FILE = "bcb_creditcard_competition_dataset_v4.csv"
OUTPUT_FILE = "dataset_model_trimestral.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print("Dataset original:", df.shape)
print("Intervalo:", df["date"].min().date(), "->", df["date"].max().date())


# ------------------------------------------------------------
# 2) Converter para base trimestral (Mar/Jun/Set/Dez)
#    (usando o ÚLTIMO valor disponível do mês do trimestre)
# ------------------------------------------------------------

df_q = df[df["date"].dt.month.isin([3, 6, 9, 12])].copy().reset_index(drop=True)
print("Dataset trimestral:", df_q.shape)

# Se quiser limitar o período (opcional):
# df_q = df_q[(df_q["date"] >= "2018-03-01") & (df_q["date"] <= "2025-12-01")].copy()


# ------------------------------------------------------------
# 3) Criar variáveis derivadas (com checagens)
# ------------------------------------------------------------

def safe_log_diff(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)  # log só para >0
    return np.log(s).diff()

def safe_diff(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").diff()

def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return np.where(d != 0, n / d, np.nan)

# Δ log do total_cartao (IF.DATA)
if "total_cartao" in df_q.columns:
    df_q["dlog_total_cartao"] = safe_log_diff(df_q["total_cartao"])
else:
    df_q["dlog_total_cartao"] = np.nan

# Δ HHI do cartão
if "ihhn_cartao_0_1" in df_q.columns:
    df_q["delta_hhi_cartao"] = safe_diff(df_q["ihhn_cartao_0_1"])
else:
    df_q["delta_hhi_cartao"] = np.nan

# Spread cartão PF (juros anual - selic anual aproximada)
# OBS: selic_am_pct é AO MÊS (%) no seu dataset -> anual aprox = selic_am_pct * 12
if "juros_cartao_pf_aa_pct" in df_q.columns and "selic_am_pct" in df_q.columns:
    df_q["spread_cartao_pf"] = pd.to_numeric(df_q["juros_cartao_pf_aa_pct"], errors="coerce") - (
        pd.to_numeric(df_q["selic_am_pct"], errors="coerce") * 12
    )
else:
    df_q["spread_cartao_pf"] = np.nan

# Proporção rotativo / total (estoques)
if "saldo_cartao_rotativo_pf_mr" in df_q.columns and "saldo_cartao_pf_total_mr" in df_q.columns:
    df_q["prop_rotativo"] = safe_ratio(df_q["saldo_cartao_rotativo_pf_mr"], df_q["saldo_cartao_pf_total_mr"])
else:
    df_q["prop_rotativo"] = np.nan

# Δ log do saldo total do cartão (SGS total) — alternativa importante
if "saldo_cartao_pf_total_mr" in df_q.columns:
    df_q["dlog_saldo_cartao_pf_total_mr"] = safe_log_diff(df_q["saldo_cartao_pf_total_mr"])
else:
    df_q["dlog_saldo_cartao_pf_total_mr"] = np.nan

# Δ HHI crédito (HHI mercado crédito classificado)
if "hhi_credito_0_1" in df_q.columns:
    df_q["delta_hhi_credito"] = safe_diff(df_q["hhi_credito_0_1"])
else:
    df_q["delta_hhi_credito"] = np.nan


# ------------------------------------------------------------
# 4) Remover apenas linhas onde TODAS as derivadas são NaN
#    (não “mata” o dataset inteiro)
# ------------------------------------------------------------

derived_cols = [
    "dlog_total_cartao",
    "delta_hhi_cartao",
    "spread_cartao_pf",
    "prop_rotativo",
    "dlog_saldo_cartao_pf_total_mr",
    "delta_hhi_credito",
]
keep_mask = df_q[derived_cols].notna().any(axis=1)
df_q = df_q[keep_mask].reset_index(drop=True)

print("Após filtrar linhas sem derivadas:", df_q.shape)


# ------------------------------------------------------------
# 5) Teste de estacionariedade (ADF) — robusto
# ------------------------------------------------------------

def adf_test(series: pd.Series, name: str, min_n: int = 12):
    s = pd.to_numeric(series, errors="coerce").dropna()
    print("\nADF Test -", name)
    print("N:", len(s))

    if not HAS_STATSMODELS:
        print("statsmodels não instalado => pulei ADF (instale com: pip install statsmodels)")
        return

    if len(s) < min_n:
        print(f"Pulei ADF: pouca amostra (min {min_n})")
        return

    if s.nunique() <= 1:
        print("Pulei ADF: série constante (sem variância)")
        return

    try:
        result = adfuller(s)
        print("ADF Statistic:", round(result[0], 4))
        print("p-value:", round(result[1], 4))
        if result[1] < 0.05:
            print("=> Série estacionária")
        else:
            print("=> Série NÃO estacionária")
    except Exception as e:
        print("Falhou ADF:", repr(e))


print("\n==============================")
print("TESTES DE ESTACIONARIEDADE")
print("==============================")

adf_test(df_q["dlog_total_cartao"], "Δ log Total Cartão (IF.DATA)")
adf_test(df_q["delta_hhi_cartao"], "Δ HHI Cartão (IF.DATA)")
adf_test(df_q["spread_cartao_pf"], "Spread Cartão (juros - selic anual aprox)")
adf_test(df_q["prop_rotativo"], "Proporção Rotativo / Total")
adf_test(df_q["dlog_saldo_cartao_pf_total_mr"], "Δ log Saldo Cartão PF Total (SGS)")
adf_test(df_q["delta_hhi_credito"], "Δ HHI Crédito (IF.DATA)")


# ------------------------------------------------------------
# 6) Salvar dataset pronto para modelagem
# ------------------------------------------------------------

df_q.to_csv(OUTPUT_FILE, index=False)
print("\nArquivo gerado:", OUTPUT_FILE)
print("Observações finais:", df_q.shape)