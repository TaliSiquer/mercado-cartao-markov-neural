from __future__ import annotations

import os
import re
import time
import zipfile
from io import BytesIO
from datetime import datetime

import requests
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# Amostra ampliada:
# começa em 2012Q2 (junho/2012)
START_DATE = "01/06/2012"
END_DATE = datetime.now().strftime("%d/%m/%Y")

OUT_CSV = "bcb_creditcard_competition_dataset_v4.csv"

# Caches
HHI_CACHE_FILE = "ifdata_credito_hhi_cache.csv"
CARTAO_CACHE_FILE = "ifdata_cartao_cache.csv"
CVM_SERVICOS_CACHE_FILE = "cvm_servicos_proxy_hhi_cache.csv"


# ============================================================
# SGS (BCB)
# ============================================================

SGS_SERIES = {
    "selic_am_pct": 4390,
    "juros_cartao_pf_aa_pct": 22024,
    "inadimplencia_90d_pct": 21082,
    "concessoes_total_mr": 20631,
    "saldo_cartao_pf_total_mr": 20590,
    "cartoes_emitidos_milhoes": 25147,
    "juros_cartao_rotativo_pf_mensal_pct": 25477,
    "juros_cartao_parcelado_pf_mensal_pct": 25478,
    "saldo_cartao_rotativo_pf_mr": 20587,
    "saldo_cartao_parcelado_pf_mr": 20588,
    "concessoes_cartao_rotativo_pf_mr": 20679,
    "concessoes_cartao_parcelado_pf_mr": 20680,
    "endividamento_familias_pct_renda": 29038,
    "comprometimento_renda_pct": 29265,
}


# ============================================================
# IF.DATA (Olinda/OData)
# ============================================================

IFDATA_BASE = (
    "https://olinda.bcb.gov.br/"
    "olinda/servico/IFDATA/versao/v1/odata"
)

IFDATA_TIPO_INSTITUICAO = 1

# ------------------------------------------------------------
# HHI / CR5 — Crédito
# ------------------------------------------------------------

IFDATA_RELATORIO_HHI = "1"

# Até 2024Q4:
IFDATA_NOMECOLUNA_CREDITO_ATE_2024 = (
    "Carteira de Crédito Classificada"
)

# A partir de 2025Q1:
IFDATA_NOMECOLUNA_CREDITO_DESDE_2025 = (
    "Carteira de Crédito"
)

# ------------------------------------------------------------
# Cartão PF — relatório 11
# ------------------------------------------------------------

IFDATA_RELATORIO_CARTAO = "11"
IFDATA_GRUPO_CARTAO = "Cartão de Crédito"


# ============================================================
# CVM ITR (Serviços)
# ============================================================

CVM_ITR_ZIP_URL = (
    "https://dados.cvm.gov.br/"
    "dados/CIA_ABERTA/DOC/ITR/DADOS/"
    "itr_cia_aberta_{year}.zip"
)

SERVICOS_CONTA_REGEX = re.compile(
    r"(prest[aã]c[aã]o\s+de\s+servi[cç]os|"
    r"receita\s+de\s+servi[cç]os|"
    r"tarifas|comiss[oõ]es|servi[cç]os)",
    flags=re.IGNORECASE,
)


# ============================================================
# COLUNAS PARA CORTE FINAL
# ============================================================

SGS_CORE_REQUIRED_COLS = [
    "selic_am_pct",
    "juros_cartao_pf_aa_pct",
    "inadimplencia_90d_pct",
    "concessoes_total_mr",
    "saldo_cartao_pf_total_mr",
    "juros_cartao_rotativo_pf_mensal_pct",
    "juros_cartao_parcelado_pf_mensal_pct",
    "saldo_cartao_rotativo_pf_mr",
    "saldo_cartao_parcelado_pf_mr",
    "concessoes_cartao_rotativo_pf_mr",
    "concessoes_cartao_parcelado_pf_mr",
]


# ============================================================
# HTTP HELPERS
# ============================================================

def _get_json(
    url: str,
    params: dict | None = None,
    timeout: int = 240,
    tries: int = 5,
) -> dict:

    last_err = None

    for attempt in range(tries):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
            )

            r.raise_for_status()

            txt = (r.text or "").strip()

            if (
                txt.startswith("/*")
                and txt.endswith("*/")
            ):
                txt = txt[2:-2].strip()

            if not txt.startswith("{"):
                raise ValueError(
                    "Resposta não-JSON "
                    f"(início): {txt[:120]}"
                )

            return r.json()

        except Exception as e:
            last_err = e
            time.sleep(
                2 * (attempt + 1)
            )

    raise RuntimeError(
        f"Falha HTTP após {tries} tentativas. "
        f"URL={url} "
        f"params={params} "
        f"err={last_err!r}"
    )


def _download_bytes(
    url: str,
    timeout: int = 240,
    tries: int = 3,
) -> bytes:

    last_err = None

    for attempt in range(tries):
        try:
            r = requests.get(
                url,
                timeout=timeout,
            )

            r.raise_for_status()
            return r.content

        except Exception as e:
            last_err = e
            time.sleep(
                2 * (attempt + 1)
            )

    raise RuntimeError(
        f"Falha download após {tries} tentativas: "
        f"{url} err={last_err!r}"
    )


# ============================================================
# SGS
# ============================================================

def fetch_sgs_series(
    code: int
) -> pd.DataFrame:

    url = (
        f"https://api.bcb.gov.br/dados/serie/"
        f"bcdata.sgs.{code}/dados"
        f"?formato=json"
        f"&dataInicial={START_DATE}"
        f"&dataFinal={END_DATE}"
    )

    for attempt in range(3):
        try:
            r = requests.get(
                url,
                timeout=90,
            )

            r.raise_for_status()

            if not r.text.strip().startswith("["):
                return pd.DataFrame(
                    columns=[
                        "date",
                        "value",
                    ]
                )

            df = pd.DataFrame(
                r.json()
            )

            df["date"] = pd.to_datetime(
                df["data"],
                format="%d/%m/%Y",
                errors="coerce",
            )

            df["value"] = pd.to_numeric(
                df["valor"]
                .astype(str)
                .str.replace(
                    ",",
                    ".",
                    regex=False,
                ),
                errors="coerce",
            )

            df = (
                df[
                    [
                        "date",
                        "value",
                    ]
                ]
                .dropna()
            )

            return df

        except Exception:
            time.sleep(
                attempt + 1
            )

    return pd.DataFrame(
        columns=[
            "date",
            "value",
        ]
    )


def build_sgs_dataset() -> pd.DataFrame:

    dfs: list[pd.DataFrame] = []

    for name, code in SGS_SERIES.items():
        df = fetch_sgs_series(
            code
        )

        if df.empty:
            continue

        dfs.append(
            df.rename(
                columns={
                    "value": name
                }
            )
        )

    if not dfs:
        raise RuntimeError(
            "Nenhuma série SGS retornou dados."
        )

    base = dfs[0]

    for df in dfs[1:]:
        base = pd.merge(
            base,
            df,
            on="date",
            how="outer",
        )

    base = base.loc[
        :,
        ~base.columns.duplicated()
    ]

    return (
        base
        .sort_values("date")
        .reset_index(drop=True)
    )


def to_monthly_last(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = (
        df.loc[
            :,
            ~df.columns.duplicated()
        ]
        .copy()
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=["date"]
        )
        .sort_values("date")
    )

    df["month"] = (
        df["date"]
        .dt
        .to_period("M")
        .dt
        .to_timestamp()
    )

    df = df.drop(
        columns=["date"]
    )

    df = (
        df
        .groupby(
            "month",
            as_index=False,
        )
        .last()
    )

    df = df.rename(
        columns={
            "month": "date"
        }
    )

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# IF.DATA — DATAS TRIMESTRAIS
# ============================================================

def build_anomes_quarterly(
    start_year: int = 2012,
    start_month: int = 6,
    end_year: int | None = None,
) -> list[int]:

    if end_year is None:
        end_year = datetime.now().year

    quarters = []

    for y in range(
        start_year,
        end_year + 1,
    ):
        for m in (
            3,
            6,
            9,
            12,
        ):
            if (
                y == start_year
                and m < start_month
            ):
                continue

            quarters.append(
                y * 100 + m
            )

    return quarters


def _anomes_to_date(
    anomes: int
) -> pd.Timestamp:

    return (
        pd.to_datetime(
            f"{anomes}01",
            format="%Y%m%d",
        )
        .to_period("M")
        .to_timestamp()
    )


# ============================================================
# IF.DATA — HHI + CR5 CRÉDITO
# ============================================================

def compute_hhi_cr5(
    anomes_list: list[int]
) -> pd.DataFrame:

    rows = []

    url = (
        f"{IFDATA_BASE}/"
        "IfDataValores("
        "AnoMes=@AnoMes,"
        "TipoInstituicao=@TipoInstituicao,"
        "Relatorio=@Relatorio)"
    )

    for anomes in anomes_list:

        params = {
            "$format": "json",
            "$top": "250000",
            "@AnoMes": str(anomes),
            "@TipoInstituicao":
                str(
                    IFDATA_TIPO_INSTITUICAO
                ),
            "@Relatorio":
                f"'{IFDATA_RELATORIO_HHI}'",
        }

        try:
            payload = _get_json(
                url,
                params=params,
                timeout=240,
                tries=5,
            )

        except Exception as e:
            print(
                f"⚠️ IF.data HHI/CR5 "
                f"{anomes}: falha na consulta: {e}"
            )
            continue

        df = pd.DataFrame(
            payload.get(
                "value",
                [],
            )
        )

        if df.empty:
            print(
                f"   HHI/CR5 {anomes}: "
                "sem dados no Relatório 1"
            )
            continue

        if "NomeColuna" not in df.columns:
            print(
                f"⚠️ HHI/CR5 {anomes}: "
                "coluna NomeColuna não encontrada"
            )
            continue

        df["NomeColuna"] = (
            df["NomeColuna"]
            .astype(str)
            .str.strip()
        )

        # ----------------------------------------------------
        # Escolha da rubrica
        # ----------------------------------------------------

        if anomes <= 202412:
            rubricas_tentativa = [
                IFDATA_NOMECOLUNA_CREDITO_ATE_2024,
                IFDATA_NOMECOLUNA_CREDITO_DESDE_2025,
            ]
        else:
            rubricas_tentativa = [
                IFDATA_NOMECOLUNA_CREDITO_DESDE_2025,
                IFDATA_NOMECOLUNA_CREDITO_ATE_2024,
            ]

        credito = pd.DataFrame()
        rubrica_utilizada = None

        for rubrica in rubricas_tentativa:
            teste = df[
                df["NomeColuna"]
                .eq(rubrica)
            ].copy()

            if not teste.empty:
                credito = teste
                rubrica_utilizada = rubrica
                break

        if credito.empty:
            print(
                f"   HHI/CR5 {anomes}: "
                "rubrica de carteira de crédito "
                "não encontrada"
            )
            continue

        # ----------------------------------------------------
        # Remoção de duplicidades exatas
        # ----------------------------------------------------

        duplicate_cols = [
            c
            for c in [
                "CodInst",
                "Conta",
                "NomeColuna",
                "Saldo",
            ]
            if c in credito.columns
        ]

        n_antes = len(
            credito
        )

        if duplicate_cols:
            credito = (
                credito
                .drop_duplicates(
                    subset=duplicate_cols
                )
                .copy()
            )

        n_depois = len(
            credito
        )

        duplicadas_removidas = (
            n_antes
            - n_depois
        )

        # ----------------------------------------------------
        # Conversão dos saldos
        # ----------------------------------------------------

        credito["Saldo"] = pd.to_numeric(
            credito["Saldo"],
            errors="coerce",
        )

        credito = credito.dropna(
            subset=[
                "CodInst",
                "Saldo",
            ]
        )

        # ----------------------------------------------------
        # Agregação por instituição
        # ----------------------------------------------------

        by_inst = (
            credito
            .groupby(
                "CodInst",
                as_index=False,
            )["Saldo"]
            .sum()
        )

        # Apenas saldos positivos
        by_inst = (
            by_inst[
                by_inst["Saldo"] > 0
            ]
            .copy()
        )

        if by_inst.empty:
            print(
                f"   HHI/CR5 {anomes}: "
                "nenhum saldo positivo válido"
            )
            continue

        total = float(
            by_inst[
                "Saldo"
            ].sum()
        )

        if total <= 0:
            print(
                f"   HHI/CR5 {anomes}: "
                "saldo total inválido"
            )
            continue

        # ----------------------------------------------------
        # Shares
        # ----------------------------------------------------

        by_inst["share"] = (
            by_inst["Saldo"]
            / total
        )

        # ----------------------------------------------------
        # HHI
        # ----------------------------------------------------

        hhi = float(
            (
                by_inst["share"]
                ** 2
            ).sum()
        )

        # ----------------------------------------------------
        # CR5
        # ----------------------------------------------------

        cr5 = float(
            by_inst
            .nlargest(
                5,
                "share",
            )[
                "share"
            ]
            .sum()
        )

        rows.append(
            {
                "date":
                    _anomes_to_date(
                        anomes
                    ),
                "hhi_credito_0_1":
                    hhi,
                "hhi_credito_0_10000":
                    10000 * hhi,
                "cr5_credito":
                    cr5,
                "n_instituicoes_credito":
                    int(
                        len(
                            by_inst
                        )
                    ),
            }
        )

        msg = (
            f"   HHI/CR5 {anomes}: "
            f"{rubrica_utilizada} | "
            f"N inst.={len(by_inst)} | "
            f"HHI={hhi:.6f} | "
            f"CR5={cr5:.6f}"
        )

        if duplicadas_removidas > 0:
            msg += (
                f" | duplicadas removidas="
                f"{duplicadas_removidas}"
            )

        print(
            msg
        )

    out = pd.DataFrame(
        rows
    )

    if not out.empty:
        out["date"] = pd.to_datetime(
            out["date"]
        )

        out = (
            out
            .sort_values(
                "date"
            )
            .drop_duplicates(
                subset=["date"],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

    return out


# ============================================================
# IF.DATA — CARTÃO
# ============================================================

def compute_cartao_metrics(
    anomes_list: list[int]
) -> pd.DataFrame:

    rows = []

    url = (
        f"{IFDATA_BASE}/"
        "IfDataValores("
        "AnoMes=@AnoMes,"
        "TipoInstituicao=@TipoInstituicao,"
        "Relatorio=@Relatorio)"
    )

    for anomes in anomes_list:

        params = {
            "$format": "json",
            "$top": "250000",
            "@AnoMes":
                str(anomes),
            "@TipoInstituicao":
                str(
                    IFDATA_TIPO_INSTITUICAO
                ),
            "@Relatorio":
                f"'{IFDATA_RELATORIO_CARTAO}'",
        }

        payload = _get_json(
            url,
            params=params,
            timeout=240,
            tries=5,
        )

        df = pd.DataFrame(
            payload.get(
                "value",
                [],
            )
        )

        if df.empty:
            continue

        df["Grupo"] = (
            df["Grupo"]
            .astype(str)
            .str.strip()
        )

        df = df[
            df["Grupo"]
            .str
            .casefold()
            == IFDATA_GRUPO_CARTAO.casefold()
        ].copy()

        if df.empty:
            continue

        df["NomeColuna"] = (
            df["NomeColuna"]
            .astype(str)
            .str.strip()
        )

        df["Saldo"] = pd.to_numeric(
            df["Saldo"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "CodInst",
                "Saldo",
            ]
        )

        # Remove duplicidades exatas retornadas pelo IF.Data.
        # Ex.: em 2025Q3 cada instituição veio 3 vezes.
        n_antes = len(df)

        df = (
            df
            .drop_duplicates(
                subset=[
                    "CodInst",
                    "NomeColuna",
                    "Saldo",
                ]
            )
            .copy()
        )

        n_removidas = n_antes - len(df)

        if n_removidas > 0:
            print(
                f"   IF.data Cartão {anomes}: "
                f"{n_removidas} linhas duplicadas removidas"
            )

        total_by_inst = (
            df[
                df["NomeColuna"]
                .str
                .casefold()
                == "total"
            ]
            .groupby(
                "CodInst",
                as_index=False,
            )["Saldo"]
            .sum()
            .rename(
                columns={
                    "Saldo":
                        "total"
                }
            )
        )

        venc_by_inst = (
            df[
                df["NomeColuna"]
                .str
                .contains(
                    r"\bvencid",
                    case=False,
                    regex=True,
                )
            ]
            .groupby(
                "CodInst",
                as_index=False,
            )["Saldo"]
            .sum()
            .rename(
                columns={
                    "Saldo":
                        "vencido"
                }
            )
        )

        if total_by_inst.empty:
            continue

        by_inst = (
            total_by_inst
            .merge(
                venc_by_inst,
                on="CodInst",
                how="left",
            )
        )

        by_inst["vencido"] = (
            by_inst["vencido"]
            .fillna(0.0)
        )

        by_inst["ativo"] = (
            by_inst["total"]
            - by_inst["vencido"]
        ).clip(
            lower=0.0
        )

        total_cartao = float(
            by_inst[
                "total"
            ].sum()
        )

        total_venc = float(
            by_inst[
                "vencido"
            ].sum()
        )

        total_ativo = float(
            by_inst[
                "ativo"
            ].sum()
        )

        tx_venc = (
            total_venc
            / total_cartao
            if total_cartao > 0
            else pd.NA
        )

        ihhn = pd.NA
        en = pd.NA

        if total_cartao > 0:
            shares = (
                by_inst["total"]
                / total_cartao
            ).astype(float)

            ihhn = float(
                (
                    shares
                    ** 2
                ).sum()
            )

            en = (
                float(
                    1.0 / ihhn
                )
                if ihhn > 0
                else pd.NA
            )

        rows.append(
            {
                "date":
                    _anomes_to_date(
                        anomes
                    ),
                "total_cartao":
                    total_cartao,
                "total_cartao_ativo":
                    total_ativo,
                "total_cartao_vencido":
                    total_venc,
                "tx_vencido_cartao":
                    tx_venc,
                "ihhn_cartao_0_1":
                    ihhn,
                "en_cartao":
                    en,
            }
        )

    out = pd.DataFrame(
        rows
    )

    if not out.empty:
        out["date"] = pd.to_datetime(
            out["date"]
        )

        out = (
            out
            .sort_values("date")
            .drop_duplicates(
                subset=["date"]
            )
        )

    return out


# ============================================================
# CVM / ITR
# ============================================================

def _read_csv_from_zip(
    zip_bytes: bytes,
    member_name: str,
) -> pd.DataFrame:

    with zipfile.ZipFile(
        BytesIO(
            zip_bytes
        )
    ) as z:

        with z.open(
            member_name
        ) as f:

            return pd.read_csv(
                f,
                sep=";",
                encoding="latin-1",
                low_memory=False,
            )


def _pick_best_dre_members(
    names: list[str],
    year: int,
) -> list[str]:

    year_str = str(
        year
    )

    cand = []

    def add_if_exists(
        pat: str
    ):

        for n in names:
            if (
                n.lower()
                == pat.lower()
            ):
                cand.append(
                    n
                )

    add_if_exists(
        f"itr_cia_aberta_DRE_con_{year_str}.csv"
    )

    add_if_exists(
        f"itr_cia_aberta_DRE_ind_{year_str}.csv"
    )

    if not cand:
        for n in names:
            nl = n.lower()

            if (
                "dre" in nl
                and year_str in nl
                and nl.endswith(".csv")
            ):
                cand.append(
                    n
                )

    seen = set()
    out = []

    for c in cand:
        if c not in seen:
            out.append(
                c
            )
            seen.add(
                c
            )

    return out


def periodo_to_date(
    periodo: str
) -> pd.Timestamp:

    p = str(
        periodo
    ).strip().upper()

    year = int(
        p[:4]
    )

    t = int(
        p[-1]
    )

    month_map = {
        1: 3,
        2: 6,
        3: 9,
        4: 12,
    }

    m = month_map[t]

    return (
        pd.to_datetime(
            f"{year}-{m:02d}-01"
        )
        .to_period("M")
        .to_timestamp()
    )


def compute_cvm_servicos_hhi_en(
    start_year: int = 2015
) -> pd.DataFrame:

    BANK_CANDIDATES_FILE = (
        "bank_candidates.csv"
    )

    if not os.path.exists(
        BANK_CANDIDATES_FILE
    ):
        return pd.DataFrame(
            columns=[
                "date",
                "hhi_servicos_0_1",
                "en_servicos",
            ]
        )

    banks = pd.read_csv(
        BANK_CANDIDATES_FILE
    )

    banks.columns = [
        c.strip().upper()
        for c in banks.columns
    ]

    for req in [
        "BANK",
        "CD_CVM",
        "CNPJ_CIA",
    ]:
        if req not in banks.columns:
            raise RuntimeError(
                "bank_candidates.csv precisa "
                f"ter {req}. "
                f"Colunas: "
                f"{banks.columns.tolist()}"
            )

    banks["CD_CVM"] = pd.to_numeric(
        banks["CD_CVM"],
        errors="coerce",
    )

    banks = (
        banks
        .dropna(
            subset=[
                "CD_CVM"
            ]
        )
        .copy()
    )

    banks["CD_CVM"] = (
        banks["CD_CVM"]
        .astype(int)
    )

    if os.path.exists(
        CVM_SERVICOS_CACHE_FILE
    ):
        cache = pd.read_csv(
            CVM_SERVICOS_CACHE_FILE,
            parse_dates=[
                "date"
            ],
        )
    else:
        cache = pd.DataFrame()

    end_year = (
        datetime.now().year
    )

    all_years = list(
        range(
            start_year,
            end_year + 1,
        )
    )

    cached_years = (
        set(
            cache["date"]
            .dt
            .year
            .unique()
        )
        if not cache.empty
        else set()
    )

    missing_years = [
        y
        for y in all_years
        if y not in cached_years
    ]

    if missing_years:
        print(
            "==> CVM/ITR serviços: "
            f"anos faltantes "
            f"{missing_years}"
        )

    new_rows = []

    for year in missing_years:
        try:
            print(
                "==> CVM/ITR baixando "
                f"{year} ..."
            )

            url = (
                CVM_ITR_ZIP_URL
                .format(
                    year=year
                )
            )

            zbytes = _download_bytes(
                url,
                timeout=240,
                tries=3,
            )

            with zipfile.ZipFile(
                BytesIO(
                    zbytes
                )
            ) as z:
                names = (
                    z.namelist()
                )

            dre_members = (
                _pick_best_dre_members(
                    names,
                    year,
                )
            )

            if not dre_members:
                raise RuntimeError(
                    "Não encontrei DRE "
                    "dentro do ZIP "
                    f"de {year}. "
                    f"Exemplos: "
                    f"{names[:10]}"
                )

            dre_all = []

            for mem in dre_members:
                dre_all.append(
                    _read_csv_from_zip(
                        zbytes,
                        mem,
                    )
                )

            dre = pd.concat(
                dre_all,
                ignore_index=True,
            )

            dre.columns = [
                c.strip().upper()
                for c in dre.columns
            ]

            required_any = [
                "CD_CVM",
                "DT_REFER",
                "DS_CONTA",
                "VL_CONTA",
            ]

            for c in required_any:
                if c not in dre.columns:
                    raise RuntimeError(
                        f"Coluna {c} "
                        "não encontrada "
                        f"no DRE {year}. "
                        f"Colunas: "
                        f"{list(dre.columns)[:25]} ..."
                    )

            dre["CD_CVM"] = pd.to_numeric(
                dre["CD_CVM"],
                errors="coerce",
            )

            dre["DT_REFER"] = pd.to_datetime(
                dre["DT_REFER"],
                errors="coerce",
            )

            dre["VL_CONTA"] = pd.to_numeric(
                dre["VL_CONTA"],
                errors="coerce",
            )

            dre["DS_CONTA"] = (
                dre["DS_CONTA"]
                .astype(str)
            )

            dre = (
                dre
                .dropna(
                    subset=[
                        "CD_CVM",
                        "DT_REFER",
                        "VL_CONTA",
                    ]
                )
                .copy()
            )

            dre["CD_CVM"] = (
                dre["CD_CVM"]
                .astype(int)
            )

            dre = dre.merge(
                banks[
                    [
                        "BANK",
                        "CD_CVM",
                    ]
                ],
                on="CD_CVM",
                how="inner",
            )

            if dre.empty:
                continue

            dre = dre[
                dre["DS_CONTA"]
                .apply(
                    lambda s:
                    bool(
                        SERVICOS_CONTA_REGEX
                        .search(
                            s
                        )
                    )
                )
            ]

            if dre.empty:
                continue

            dre["PERIODO"] = (
                dre["DT_REFER"]
                .dt
                .to_period("Q")
                .astype(str)
                .str
                .replace(
                    "Q",
                    "T",
                    regex=False,
                )
            )

            by_bank = (
                dre
                .groupby(
                    [
                        "PERIODO",
                        "BANK",
                    ],
                    as_index=False,
                )["VL_CONTA"]
                .sum()
            )

            by_bank = (
                by_bank
                .rename(
                    columns={
                        "VL_CONTA":
                            "PROXY_SERVICOS"
                    }
                )
            )

            for periodo, grp in (
                by_bank
                .groupby(
                    "PERIODO"
                )
            ):

                total = float(
                    grp[
                        "PROXY_SERVICOS"
                    ].sum()
                )

                if total <= 0:
                    continue

                shares = (
                    grp[
                        "PROXY_SERVICOS"
                    ]
                    / total
                ).astype(float)

                hhi = float(
                    (
                        shares
                        ** 2
                    ).sum()
                )

                en = (
                    float(
                        1.0 / hhi
                    )
                    if hhi > 0
                    else pd.NA
                )

                new_rows.append(
                    {
                        "date":
                            periodo_to_date(
                                periodo
                            ),
                        "hhi_servicos_0_1":
                            hhi,
                        "en_servicos":
                            en,
                    }
                )

        except Exception as e:
            print(
                f"⚠️ CVM {year} falhou: "
                f"{e}"
            )

    new_df = pd.DataFrame(
        new_rows
    )

    if not new_df.empty:
        new_df["date"] = pd.to_datetime(
            new_df["date"]
        )

        new_df = (
            new_df
            .sort_values("date")
            .drop_duplicates(
                subset=[
                    "date"
                ]
            )
        )

        if cache.empty:
            cache = new_df
        else:
            cache = (
                pd.concat(
                    [
                        cache,
                        new_df,
                    ],
                    ignore_index=True,
                )
                .drop_duplicates(
                    subset=[
                        "date"
                    ]
                )
                .sort_values(
                    "date"
                )
            )

        cache.to_csv(
            CVM_SERVICOS_CACHE_FILE,
            index=False,
        )

    if cache.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "hhi_servicos_0_1",
                "en_servicos",
            ]
        )

    return cache[
        [
            "date",
            "hhi_servicos_0_1",
            "en_servicos",
        ]
    ].copy()


# ============================================================
# CORTE FINAL — MÊS CALENDÁRIO COMPLETO
# ============================================================

def cut_to_last_complete_month(
    df: pd.DataFrame,
    required_cols: list[str],
) -> tuple[
    pd.DataFrame,
    pd.Timestamp | None,
]:

    df = df.copy()

    if (
        df.empty
        or "date" not in df.columns
    ):
        return df, None

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "date"
            ]
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    hoje = (
        pd.Timestamp
        .today()
        .normalize()
    )

    ultimo_mes_calendario_completo = (
        hoje
        .to_period("M")
        - 1
    ).to_timestamp()

    missing_cols = [
        c
        for c in required_cols
        if c not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Colunas SGS obrigatórias ausentes: "
            + ", ".join(missing_cols)
        )

    cols = required_cols.copy()

    calendario_ok = (
        df["date"]
        <= ultimo_mes_calendario_completo
    )

    dados_ok = (
        df[cols]
        .notna()
        .all(
            axis=1
        )
    )

    complete_mask = (
        calendario_ok
        & dados_ok
    )

    if not complete_mask.any():
        return df, None

    last_complete = (
        df.loc[
            complete_mask,
            "date",
        ]
        .max()
    )

    out = (
        df[
            df["date"]
            <= last_complete
        ]
        .reset_index(
            drop=True
        )
    )

    return (
        out,
        last_complete,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "==> SGS"
    )

    sgs = to_monthly_last(
        build_sgs_dataset()
    )


# ========================================================
    # SGS — PRESERVAR OBSERVAÇÕES ORIGINAIS
    # ========================================================

    sgs = (
        sgs
        .sort_values("date")
        .reset_index(drop=True)
    )

    sgs_cols = [
        c
        for c in SGS_SERIES.keys()
        if c in sgs.columns
    ]

    # IMPORTANTE:
    # Não aplicar ffill/bfill/interpolação nas séries SGS.
    # Quando uma série ainda não tiver sido divulgada para determinado mês,
    # o valor permanece NaN.
    #
    # Isso preserva:
    # - o período real da última observação;
    # - a cobertura real de cada série;
    # - a distinção entre "sem mudança" e "dado ainda não publicado".


    # ========================================================
    # IF.DATA — HHI / CR5
    # ========================================================

    if os.path.exists(
        HHI_CACHE_FILE
    ):
        hhi_cache = pd.read_csv(
            HHI_CACHE_FILE,
            parse_dates=[
                "date"
            ],
        )
    else:
        hhi_cache = (
            pd.DataFrame()
        )

    anomes_list = (
        build_anomes_quarterly(
            start_year=2012,
            start_month=6,
        )
    )

    cached_dates = (
        set(
            pd.to_datetime(
                hhi_cache[
                    "date"
                ]
            )
        )
        if not hhi_cache.empty
        else set()
    )

    missing = []

    for anomes in anomes_list:
        d = _anomes_to_date(
            anomes
        )

        if d not in cached_dates:
            missing.append(
                anomes
            )

    if missing:
        print(
            "==> IF.data calculando "
            f"{len(missing)} "
            "trimestres (HHI/CR5)"
        )

        new_df = compute_hhi_cr5(
            missing
        )

        if not new_df.empty:
            hhi_cache = (
                pd.concat(
                    [
                        hhi_cache,
                        new_df,
                    ],
                    ignore_index=True,
                )
                .drop_duplicates(
                    subset=[
                        "date"
                    ],
                    keep="last",
                )
                .sort_values(
                    "date"
                )
            )

            hhi_cache.to_csv(
                HHI_CACHE_FILE,
                index=False,
            )

    if not hhi_cache.empty:

        sgs["date"] = (
            pd.to_datetime(
                sgs["date"]
            )
        )

        hhi_cache["date"] = (
            pd.to_datetime(
                hhi_cache[
                    "date"
                ]
            )
        )

        sgs = sgs.merge(
            hhi_cache,
            on="date",
            how="left",
        )

        # NÃO fazer ffill em HHI/CR5.
        for c in [
            "hhi_credito_0_1",
            "hhi_credito_0_10000",
            "cr5_credito",
            "n_instituicoes_credito",
        ]:
            if c in sgs.columns:
                sgs[c] = pd.to_numeric(
                    sgs[c],
                    errors="coerce",
                )

    # ========================================================
    # IF.DATA — CARTÃO
    # ========================================================

    print(
        "==> IF.data (Cartão)"
    )

    if os.path.exists(
        CARTAO_CACHE_FILE
    ):
        cart_cache = pd.read_csv(
            CARTAO_CACHE_FILE,
            parse_dates=[
                "date"
            ],
        )
    else:
        cart_cache = (
            pd.DataFrame()
        )

    cached_cart_dates = (
        set(
            pd.to_datetime(
                cart_cache[
                    "date"
                ]
            )
        )
        if not cart_cache.empty
        else set()
    )

    missing_cart = []

    for anomes in anomes_list:
        d = _anomes_to_date(
            anomes
        )

        if d not in cached_cart_dates:
            missing_cart.append(
                anomes
            )

    if missing_cart:
        print(
            "==> IF.data (Cartão) "
            f"calculando "
            f"{len(missing_cart)} "
            "trimestres faltantes"
        )

        new_cart = (
            compute_cartao_metrics(
                missing_cart
            )
        )

        if not new_cart.empty:
            cart_cache = (
                pd.concat(
                    [
                        cart_cache,
                        new_cart,
                    ],
                    ignore_index=True,
                )
                .drop_duplicates(
                    subset=[
                        "date"
                    ]
                )
                .sort_values(
                    "date"
                )
            )

            cart_cache.to_csv(
                CARTAO_CACHE_FILE,
                index=False,
            )

    if not cart_cache.empty:

        sgs["date"] = (
            pd.to_datetime(
                sgs["date"]
            )
        )

        cart_cache["date"] = (
            pd.to_datetime(
                cart_cache[
                    "date"
                ]
            )
        )

        sgs = sgs.merge(
            cart_cache,
            on="date",
            how="left",
        )

        # NÃO fazer ffill.
        for c in [
            "total_cartao",
            "total_cartao_ativo",
            "total_cartao_vencido",
            "tx_vencido_cartao",
            "ihhn_cartao_0_1",
            "en_cartao",
        ]:
            if c in sgs.columns:
                sgs[c] = pd.to_numeric(
                    sgs[c],
                    errors="coerce",
                )

    # ========================================================
    # CVM / ITR
    # ========================================================

    print(
        "==> CVM/ITR (Serviços) — "
        "automático (proxy + HHI/EN)"
    )

    cvm_serv = (
        compute_cvm_servicos_hhi_en(
            start_year=2015
        )
    )

    if not cvm_serv.empty:

        sgs["date"] = (
            pd.to_datetime(
                sgs[
                    "date"
                ]
            )
        )

        cvm_serv["date"] = (
            pd.to_datetime(
                cvm_serv[
                    "date"
                ]
            )
        )

        sgs = sgs.merge(
            cvm_serv,
            on="date",
            how="left",
        )

        # NÃO fazer ffill.
        for c in [
            "hhi_servicos_0_1",
            "en_servicos",
        ]:
            if c in sgs.columns:
                sgs[c] = pd.to_numeric(
                    sgs[c],
                    errors="coerce",
                )

    # ========================================================
    # CORTE FINAL
    # ========================================================

    print(
        "==> Corte final "
        "(último mês calendário completo com SGS core)"
    )

    sgs_cut, last_complete = (
        cut_to_last_complete_month(
            sgs,
            SGS_CORE_REQUIRED_COLS,
        )
    )

    if last_complete is not None:
        print(
            "✅ Corte: "
            "último mês calendário com "
            "SGS core completo = "
            f"{last_complete.date()}"
        )

        sgs = sgs_cut

    else:
        print(
            "⚠️ Não foi possível identificar "
            "um último mês com SGS core completo. "
            "A base não foi cortada."
        )

    # ========================================================
    # DIAGNÓSTICO DE COBERTURA
    # ========================================================

    print(
        "\n==> Cobertura das principais variáveis"
    )

    diagnostico_cols = [
        "juros_cartao_pf_aa_pct",
        "inadimplencia_90d_pct",
        "hhi_credito_0_1",
        "cr5_credito",
        "total_cartao",
        "ihhn_cartao_0_1",
        "hhi_servicos_0_1",
    ]

    for c in diagnostico_cols:

        if c not in sgs.columns:
            continue

        valid = (
            sgs.loc[
                sgs[c].notna(),
                [
                    "date",
                    c,
                ],
            ]
        )

        if valid.empty:
            print(
                f" - {c}: "
                "sem observações"
            )

        else:
            print(
                f" - {c}: "
                f"N={len(valid)} | "
                f"início={valid['date'].min().date()} | "
                f"fim={valid['date'].max().date()}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    sgs.to_csv(
        OUT_CSV,
        index=False,
    )

    print(
        "\n✅ Arquivo gerado:",
        OUT_CSV,
    )


if __name__ == "__main__":
    main()
