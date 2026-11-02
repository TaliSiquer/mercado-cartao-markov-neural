from __future__ import annotations

import re
from functools import reduce

import pandas as pd
import requests

# =========================
# CONFIG
# =========================

OUT_CSV = "ibge_macro_dataset.csv"

URLS = {
    # IPCA por grupos
    "ipca_grupos": "https://apisidra.ibge.gov.br/values/t/7060/n1/all/v/63/p/all/c315/7170,7625,7660,7712,7766/d/v63%202 - IPCA",
    "ipca15_grupos": "https://apisidra.ibge.gov.br/values/t/7062/n1/all/v/355/p/all/c315/7169,7170,7625,7660,7712,7766/d/v355%202 - IPCA-15",

    # IPCA geral
    "ipca_geral": "https://apisidra.ibge.gov.br/values/t/7060/n1/all/v/63/p/all/c315/7169/d/v63%202 - ipca geral",
    "ipca15_geral": "https://apisidra.ibge.gov.br/values/t/7062/n1/all/v/355/p/all/c315/7169/d/v355%202 - ipca15 geral",

    # Mercado de trabalho
    "desocupacao_movel": "https://apisidra.ibge.gov.br/values/t/6381/n1/all/v/4099/p/all/d/v4099%201 - Taxa de desocupação",
    "subutilizacao_movel": "https://apisidra.ibge.gov.br/values/t/6441/n1/all/v/4118/p/all/d/v4118%201 - Subutilização",
    "desocupacao_trimestral": "https://apisidra.ibge.gov.br/values/t/6468/n1/all/v/4099/p/all/d/v4099%201 - Taxa de desocupação Trimestral",
    "subutilizacao_trimestral": "https://apisidra.ibge.gov.br/values/t/6485/n1/all/v/4118/p/all/d/v4118%201 - Subutilização Trimestral",
    "rendimento_medio_real": "https://apisidra.ibge.gov.br/values/t/6470/n1/all/v/5934/p/all - Rendimento médio mensal real",
    "massa_rendimento_real": "https://apisidra.ibge.gov.br/values/t/6474/n1/all/v/6293/p/all - Massa de rendimento mensal real",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


# =========================
# HELPERS
# =========================

def clean_url(url: str) -> str:
    return url.split(" - ")[0].strip()


def fetch_sidra(url: str) -> pd.DataFrame:
    url = clean_url(url)
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError(f"Resposta vazia para URL: {url}")

    return df


def detect_value_col(df: pd.DataFrame) -> str:
    for col in ["V", "Valor", "valor"]:
        if col in df.columns:
            return col
    raise ValueError(f"Não encontrei coluna de valor. Colunas: {df.columns.tolist()}")


def detect_period_col(df: pd.DataFrame) -> str:
    # Primeiro tenta colunas de código
    for col in ["D1C", "D2C", "D3C", "D4C"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            if s.str.match(r"^\d{6}$").any():   # AAAAMM
                return col
            if s.str.match(r"^\d{5}$").any():   # AAAAT
                return col

    # Depois tenta colunas nominais
    for col in ["D1N", "D2N", "D3N", "D4N"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()

            if s.str.contains("trimestre", regex=False).any():
                return col

            if s.str.contains(
                r"jan-fev-mar|fev-mar-abr|mar-abr-mai|abr-mai-jun|mai-jun-jul|jun-jul-ago|jul-ago-set|ago-set-out|set-out-nov|out-nov-dez",
                regex=True
            ).any():
                return col

    raise ValueError(f"Não encontrei coluna de período. Colunas: {df.columns.tolist()}")


def parse_sidra_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # AAAAMM
    yyyymm = s.str.extract(r"^(\d{6})$", expand=False)
    if yyyymm.notna().any():
        out = pd.to_datetime(yyyymm + "01", format="%Y%m%d", errors="coerce")
        if out.notna().any():
            return out

    # AAAAT
    q = s.str.extract(r"^(\d{4})([1-4])$", expand=True)
    if q.notna().all(axis=None):
        month_map = {"1": "03", "2": "06", "3": "09", "4": "12"}
        out = pd.to_datetime(q[0] + q[1].map(month_map) + "01", format="%Y%m%d", errors="coerce")
        if out.notna().any():
            return out

    # "1º trimestre 2012"
    qtxt = s.str.extract(r"([1-4]).*?trimestre.*?(\d{4})", expand=True)
    if qtxt.notna().all(axis=None):
        month_map = {"1": "03", "2": "06", "3": "09", "4": "12"}
        out = pd.to_datetime(qtxt[1] + qtxt[0].map(month_map) + "01", format="%Y%m%d", errors="coerce")
        if out.notna().any():
            return out

    # "jan-fev-mar 2012"
    tri_movel_map = {
        "jan-fev-mar": "03",
        "fev-mar-abr": "04",
        "mar-abr-mai": "05",
        "abr-mai-jun": "06",
        "mai-jun-jul": "07",
        "jun-jul-ago": "08",
        "jul-ago-set": "09",
        "ago-set-out": "10",
        "set-out-nov": "11",
        "out-nov-dez": "12",
    }

    def parse_rolling_quarter(x: str):
        x = x.lower().strip()
        m = re.match(r"([a-zç\-]+)\s+(\d{4})", x)
        if not m:
            return pd.NaT
        tri, year = m.group(1), m.group(2)
        month = tri_movel_map.get(tri)
        if month is None:
            return pd.NaT
        return pd.to_datetime(f"{year}{month}01", format="%Y%m%d", errors="coerce")

    out = s.apply(parse_rolling_quarter)
    if out.notna().any():
        return out

    raise ValueError(
        "Não consegui interpretar a coluna de período do SIDRA. "
        f"Amostra: {series.astype(str).head(10).tolist()}"
    )


def normalize_colname(text: str, prefix: str) -> str:
    t = str(text).strip().lower()

    mapping = {
        "índice geral": f"{prefix}_mensal_pct",
        "1.alimentação e bebidas": f"{prefix}_alimentacao_pct",
        "5.transportes": f"{prefix}_transportes_pct",
        "6.saúde e cuidados pessoais": f"{prefix}_saude_pct",
        "7.despesas pessoais": f"{prefix}_despesas_pessoais_pct",
        "8.educação": f"{prefix}_educacao_pct",
    }
    if t in mapping:
        return mapping[t]

    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", "_", t)
    return f"{prefix}_{t}"


def build_general_series(url: str, value_name: str) -> pd.DataFrame:
    raw = fetch_sidra(url)
    value_col = detect_value_col(raw)
    period_col = detect_period_col(raw)

    df = raw.copy()
    df = df[df[value_col].astype(str).str.strip() != "..."].copy()
    df["date"] = parse_sidra_date(df[period_col])
    df[value_name] = pd.to_numeric(
        df[value_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df = df.dropna(subset=["date", value_name]).copy()
    df = df[["date", value_name]].sort_values("date").reset_index(drop=True)
    return df


def build_ipca_block(url: str, prefix: str) -> pd.DataFrame:
    raw = fetch_sidra(url)
    print(f"[{prefix}] colunas:", raw.columns.tolist())

    value_col = detect_value_col(raw)
    period_col = detect_period_col(raw)

    if "D4N" not in raw.columns:
        raise ValueError(f"[{prefix}] Esperava coluna D4N com grupos. Colunas: {raw.columns.tolist()}")

    df = raw.copy()
    df = df[df[value_col].astype(str).str.strip() != "..."].copy()
    df["date"] = parse_sidra_date(df[period_col])
    df["value"] = pd.to_numeric(
        df[value_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    df["group_name"] = df["D4N"].astype(str).str.strip()

    df = df.dropna(subset=["date", "value", "group_name"]).copy()

    print(f"\n[{prefix}] grupos encontrados:")
    print(sorted(df["group_name"].unique().tolist()))

    out = (
        df.pivot_table(index="date", columns="group_name", values="value", aggfunc="first")
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    rename_map = {}
    for col in out.columns:
        if col == "date":
            continue
        rename_map[col] = normalize_colname(col, prefix)

    out = out.rename(columns=rename_map)
    return out


def prepare_single_series(
    df_raw: pd.DataFrame,
    value_name: str,
    frequencia: str = "mensal",
) -> pd.DataFrame:
    df = df_raw.copy()

    value_col = detect_value_col(df)
    period_col = detect_period_col(df)

    # Valor numérico; isso também elimina a linha de cabeçalho
    df[value_name] = pd.to_numeric(
        df[value_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    if frequencia == "trimestral":
        s = df[period_col].astype(str).str.strip()

        # SIDRA:
        # 201201 = 1º trimestre de 2012
        # 201202 = 2º trimestre de 2012
        # 201203 = 3º trimestre de 2012
        # 201204 = 4º trimestre de 2012
        validos = s.str.fullmatch("[0-9]{4}0[1-4]")

        df = df.loc[validos].copy()
        s = s.loc[validos]

        ano = s.str.slice(0, 4)
        trimestre = s.str.slice(4, 6)

        month_map = {
            "01": "03",
            "02": "06",
            "03": "09",
            "04": "12",
        }

        mes = trimestre.map(month_map)

        df["date"] = pd.to_datetime(
            ano + mes + "01",
            format="%Y%m%d",
            errors="coerce",
        )

    else:
        df["date"] = parse_sidra_date(df[period_col])

    df = df.dropna(subset=["date", value_name]).copy()

    df = (
        df[["date", value_name]]
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise RuntimeError("Nenhum DataFrame foi gerado.")
    out = reduce(lambda l, r: pd.merge(l, r, on="date", how="outer"), frames)
    return out.sort_values("date").reset_index(drop=True)


# =========================
# BUILD DATASET
# =========================

def build_ibge_dataset() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # IPCA geral
    print("Baixando IPCA geral...")
    frames.append(build_general_series(URLS["ipca_geral"], "ipca_mensal_pct"))

    # IPCA por grupos
    print("Baixando IPCA por grupos...")
    frames.append(build_ipca_block(URLS["ipca_grupos"], "ipca"))

    # IPCA-15 geral
    print("Baixando IPCA-15 geral...")
    frames.append(build_general_series(URLS["ipca15_geral"], "ipca15_mensal_pct"))

    # IPCA-15 por grupos
    print("Baixando IPCA-15 por grupos...")
    frames.append(build_ipca_block(URLS["ipca15_grupos"], "ipca15"))

    # Mercado de trabalho - móvel
    print("Baixando desocupação móvel...")
    raw = fetch_sidra(URLS["desocupacao_movel"])
    frames.append(prepare_single_series(raw, "taxa_desocupacao_movel_pct"))

    print("Baixando subutilização móvel...")
    raw = fetch_sidra(URLS["subutilizacao_movel"])
    frames.append(prepare_single_series(raw, "taxa_subutilizacao_movel_pct"))

    # Mercado de trabalho - trimestral
    print("Baixando desocupação trimestral...")
    raw = fetch_sidra(URLS["desocupacao_trimestral"])
    frames.append(
        prepare_single_series(
            raw,
            "taxa_desocupacao_trimestral_pct",
            frequencia="trimestral",
        )
    )

    print("Baixando subutilização trimestral...")
    raw = fetch_sidra(URLS["subutilizacao_trimestral"])
    frames.append(
        prepare_single_series(
            raw,
            "taxa_subutilizacao_trimestral_pct",
            frequencia="trimestral",
        )
    )

    print("Baixando rendimento médio real...")
    raw = fetch_sidra(URLS["rendimento_medio_real"])
    frames.append(
        prepare_single_series(
            raw,
            "rendimento_medio_real_brl",
            frequencia="trimestral",
        )
    )

    print("Baixando massa de rendimento real...")
    raw = fetch_sidra(URLS["massa_rendimento_real"])
    frames.append(
        prepare_single_series(
            raw,
            "massa_rendimento_real_mr",
            frequencia="trimestral",
        )
    )

    df = merge_frames(frames)

    # grade mensal contínua
    if not df.empty:
        full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
        df = pd.DataFrame({"date": full_dates}).merge(df, on="date", how="left")

    # consolidar duplicidades de merge
    if "ipca15_mensal_pct_x" in df.columns or "ipca15_mensal_pct_y" in df.columns:
        left = df["ipca15_mensal_pct_x"] if "ipca15_mensal_pct_x" in df.columns else pd.Series(pd.NA, index=df.index)
        right = df["ipca15_mensal_pct_y"] if "ipca15_mensal_pct_y" in df.columns else pd.Series(pd.NA, index=df.index)
        df["ipca15_mensal_pct"] = left.combine_first(right)
        drop_cols = [c for c in ["ipca15_mensal_pct_x", "ipca15_mensal_pct_y"] if c in df.columns]
        df = df.drop(columns=drop_cols)

    if "ipca_mensal_pct_x" in df.columns or "ipca_mensal_pct_y" in df.columns:
        left = df["ipca_mensal_pct_x"] if "ipca_mensal_pct_x" in df.columns else pd.Series(pd.NA, index=df.index)
        right = df["ipca_mensal_pct_y"] if "ipca_mensal_pct_y" in df.columns else pd.Series(pd.NA, index=df.index)
        df["ipca_mensal_pct"] = left.combine_first(right)
        drop_cols = [c for c in ["ipca_mensal_pct_x", "ipca_mensal_pct_y"] if c in df.columns]
        df = df.drop(columns=drop_cols)

    # As séries trimestrais permanecem somente nos meses
    # de fechamento do trimestre: março, junho, setembro e dezembro.
    # Não aplicar ffill, evitando criar observações artificiais.
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# MAIN
# =========================

def main():
    df = build_ibge_dataset()

    ordered_cols = [
        "date",
        "ipca_mensal_pct",
        "ipca_alimentacao_pct",
        "ipca_transportes_pct",
        "ipca_saude_pct",
        "ipca_despesas_pessoais_pct",
        "ipca_educacao_pct",
        "ipca15_mensal_pct",
        "ipca15_alimentacao_pct",
        "ipca15_transportes_pct",
        "ipca15_saude_pct",
        "ipca15_despesas_pessoais_pct",
        "ipca15_educacao_pct",
        "taxa_desocupacao_movel_pct",
        "taxa_subutilizacao_movel_pct",
        "taxa_desocupacao_trimestral_pct",
        "taxa_subutilizacao_trimestral_pct",
        "rendimento_medio_real_brl",
        "massa_rendimento_real_mr",
    ]

    # evita erro se alguma coluna não existir
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\n✅ Arquivo gerado: {OUT_CSV}")
    print(df.head(12).to_string(index=False))
    print("\nColunas:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()