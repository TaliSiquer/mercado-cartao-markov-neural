from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import os
import shutil

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import config
except Exception:
    config = None


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Mercado Brasileiro de Cartão de Crédito PF",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = Path(__file__).resolve().parent

if not (PROJECT_ROOT / "outputs_base_oficial").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "Mercado_de_Cartao" / "data"

ARQ_MENSAL = DATA_DIR / "metricas_master_mensal.csv"
ARQ_TRIMESTRAL = DATA_DIR / "metricas_master_trimestral.csv"
ARQ_SNAPSHOT = DATA_DIR / "metricas_dashboard.csv"
ARQ_STATUS = DATA_DIR / "status_atualizacao.csv"
ARQ_CORRELACOES = DATA_DIR / "correlacoes_dashboard.csv"
ARQ_PERIODOS = DATA_DIR / "comparacao_pre_pandemia_pos.csv"
ARQ_PREVISOES = DATA_DIR / "historico_previsoes.csv"
ARQ_DICIONARIO = DATA_DIR / "dicionario_metricas.csv"
ARQ_HISTORICO_V13 = DATA_DIR / "historico_previsoes_v13.csv"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #07111c; --panel: #0a1825; --border: rgba(89,153,198,.24);
        --text:#f4f7fb; --muted:#93a4b7; --monthly:#ff5d73;
        --quarterly:#6385ff; --annual:#f5c84c; --good:#20d79a;
        --bad:#ff6170; --neutral:#8da0b3;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(circle at 45% -15%, #0c2132 0%, #07111c 42%, #050d16 100%);
        color: var(--text);
    }
    .block-container {padding-top:1.15rem; padding-bottom:2.2rem; max-width:1760px;}
    header[data-testid="stHeader"] {background:transparent;}
    [data-testid="stSidebar"] {min-width:275px; max-width:275px;}
    .dashboard-title {font-size:clamp(2rem,3vw,2.85rem); font-weight:800; letter-spacing:-.035em; line-height:1.06; margin:0 0 .25rem 0; color:#f8fbff;}
    .dashboard-subtitle {color:#95a5b6; font-size:.98rem; margin-bottom:.25rem;}
    .section-title {font-size:1.34rem; font-weight:780; margin:.55rem 0 .05rem 0; color:#f5f8fc;}
    .section-subtitle {color:#8fa0b2; font-size:.92rem; margin:0 0 .85rem 0;}
    .update-wrap {display:flex; justify-content:flex-end;}
    .update-box {min-width:220px; padding-top:.12rem;}
    .update-label {color:#dbe5ef; font-size:.76rem; font-weight:700; margin-bottom:.4rem;}
    .update-pill {border:1px solid rgba(68,156,217,.42); background:rgba(18,48,72,.72); border-radius:9px; padding:.58rem .8rem; font-weight:730; font-size:.86rem; color:#eef7ff;}
    .update-source {color:#8295a8; font-size:.70rem; margin-top:.42rem;}
    .kpi-card {position:relative; border:1px solid var(--border); border-radius:13px; padding:13px 14px 12px 14px; min-height:164px; background:linear-gradient(145deg,rgba(13,31,46,.96),rgba(7,21,33,.92)); box-shadow:inset 0 1px 0 rgba(255,255,255,.015),0 8px 26px rgba(0,0,0,.10);}
    .kpi-top {display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:5px;}
    .kpi-name-wrap {display:flex; align-items:center; gap:9px; min-width:0;}
    .kpi-icon {width:28px; height:28px; display:inline-flex; align-items:center; justify-content:center; flex:0 0 28px;}
    .kpi-icon svg {width:22px; height:22px; stroke:currentColor; fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round;}
    .icon-monthly {color:var(--monthly);} .icon-quarterly {color:var(--quarterly);} .icon-annual {color:var(--annual);}
    .kpi-label {font-size:.77rem; font-weight:690; color:#d9e2ec; line-height:1.15;}
    .freq-badge {flex:0 0 auto; border-radius:9px; padding:4px 9px; font-size:.65rem; font-weight:760; color:#f7fbff; letter-spacing:.01em;}
    .freq-monthly {background:rgba(133,49,63,.74);} .freq-quarterly {background:rgba(28,72,139,.82);} .freq-annual {background:rgba(134,99,23,.80);}
    .kpi-value {font-size:1.50rem; font-weight:810; line-height:1.05; margin:8px 0 7px 37px; color:#f7fbff; letter-spacing:-.02em;}
    .kpi-delta {font-size:.78rem; font-weight:800; margin:0 0 3px 37px;}
    .delta-good {color:var(--good);} .delta-bad {color:var(--bad);} .delta-neutral {color:var(--neutral);}
    .kpi-vs {margin-left:37px; color:#a5b4c3; font-size:.68rem; font-weight:650;}
    .kpi-period {margin-left:37px; margin-top:8px; color:#8597a8; font-size:.66rem;}
    hr {border-color:rgba(89,153,198,.22)!important;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-color:rgba(89,153,198,.24)!important; border-radius:13px!important; background:linear-gradient(145deg,rgba(13,31,46,.84),rgba(7,21,33,.78));}
    .chart-title-row {display:flex; gap:9px; align-items:flex-start; margin-bottom:2px;}
    .chart-icon {width:25px; height:25px; display:inline-flex; align-items:center; justify-content:center; flex:0 0 25px;}
    .chart-icon svg {width:20px; height:20px; stroke:currentColor; fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round;}
    .chart-title {color:#f2f6fa; font-size:.92rem; font-weight:780; line-height:1.1;}
    .chart-subtitle {color:#8ea0b2; font-size:.68rem; margin-top:3px;}
    .chart-side-period {color:#8ea0b2; font-size:.67rem; margin-bottom:.18rem;}
    .chart-side-value {color:#f7fbff; font-size:1.22rem; font-weight:820; margin-bottom:.2rem;}
    .chart-side-delta {font-size:.74rem; font-weight:760; margin-bottom:.25rem;}
    .chart-side-vs {color:#91a4b7; font-size:.65rem;}
    .legend-row {display:flex; flex-wrap:wrap; gap:20px; align-items:center; margin-top:.6rem; color:#9eb0c1; font-size:.72rem;}
    .legend-item {display:flex; align-items:center; gap:7px;}
    .legend-dot {width:13px; height:13px; border-radius:50%; display:inline-block;}
    .legend-square {width:13px; height:13px; border-radius:3px; display:inline-block; background:rgba(203,67,149,.50); border:1px solid rgba(244,105,188,.45);}
    .source-foot {color:#8295a8; font-size:.68rem; text-align:right; margin-top:.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CARREGAMENTO
# ============================================================

@st.cache_data
def ler_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Converte apenas nomes que realmente representam datas.
    for candidato in [
        "date",
        "data",
        "data_fonte",
        "primeira_data",
        "ultima_data",
        "data_atualizacao_pipeline",
    ]:
        if candidato in df.columns:
            df[candidato] = pd.to_datetime(df[candidato], errors="coerce")

    return df


mensal = ler_csv(ARQ_MENSAL)
trimestral = ler_csv(ARQ_TRIMESTRAL)
snapshot = ler_csv(ARQ_SNAPSHOT)
status = ler_csv(ARQ_STATUS)
correlacoes = ler_csv(ARQ_CORRELACOES)
periodos = ler_csv(ARQ_PERIODOS)
previsoes = ler_csv(ARQ_PREVISOES)
dicionario = ler_csv(ARQ_DICIONARIO)


# ============================================================
# METADADOS / NOMES
# ============================================================

FALLBACK_NOMES = {
    "juros_cartao_pf_aa_pct": "Juros cartão PF",
    "spread_cartao_pf": "Spread cartão PF",
    "hhi_credito_0_1": "HHI crédito",
    "cr5_credito": "CR5 crédito",
    "share_fintechs": "Share fintechs",
    "share_bancos_tradicionais": "Share bancos tradicionais",
    "inadimplencia_90d_pct": "Inadimplência > 90 dias",
    "saldo_cartao_pf_total_mr": "Saldo cartão PF",
    "comprometimento_renda_pct": "Comprometimento de renda",
    "endividamento_familias_pct_renda": "Endividamento das famílias",
    "selic_am_pct": "Selic mensal",
    "taxa_desocupacao_trimestral_pct": "Taxa de desocupação",
    "taxa_subutilizacao_trimestral_pct": "Taxa de subutilização",
    "rendimento_medio_real_brl": "Rendimento médio real",
    "massa_rendimento_real_mr": "Massa de rendimento real",
    "ipca_mensal_pct": "IPCA mensal",
    "prop_rotativo": "Proporção do rotativo",
}


def nome_metrica(metrica: str) -> str:
    if config is not None:
        nomes = getattr(config, "NOMES_METRICAS", None)
        if isinstance(nomes, dict) and metrica in nomes:
            return str(nomes[metrica])
    return FALLBACK_NOMES.get(metrica, metrica.replace("_", " ").title())


# Séries que sabemos terem sido expandidas para uma grade mensal via ffill.
SERIES_TRIMESTRAIS_CARREGADAS = {
    "share_fintechs",
    "share_bancos_tradicionais",
    "share_bancos_medios",
    "share_cooperativas",
    "share_instituicoes_pagamento",
    "share_financeiras",
    "share_agencias_fomento",
    "share_mercado_capitais",
    "share_outros",
    "entrada_inst",
    "saida_inst",
    "check_soma_shares",
    "taxa_desocupacao_trimestral_pct",
    "taxa_subutilizacao_trimestral_pct",
    "rendimento_medio_real_brl",
    "massa_rendimento_real_mr",
}

KPIS_CAPA = [
    {"coluna": "juros_cartao_pf_aa_pct", "titulo": "Juros cartão PF", "tipo": "percentual", "delta": "pp"},
    {"coluna": "spread_cartao_pf", "titulo": "Spread cartão PF", "tipo": "percentual", "delta": "pp"},
    {"coluna": "hhi_credito_0_1", "titulo": "HHI crédito", "tipo": "indice", "delta": "absoluto"},
    {"coluna": "share_fintechs", "titulo": "Share fintechs", "tipo": "percentual_fracao", "delta": "pp_fracao"},
    {"coluna": "inadimplencia_90d_pct", "titulo": "Inadimplência > 90 dias", "tipo": "percentual", "delta": "pp"},
    {"coluna": "saldo_cartao_pf_total_mr", "titulo": "Saldo cartão PF", "tipo": "milhoes", "delta": "percentual"},
    {"coluna": "comprometimento_renda_pct", "titulo": "Comprometimento de renda", "tipo": "percentual", "delta": "pp"},
    {"coluna": "selic_am_pct", "titulo": "Selic mensal", "tipo": "percentual", "delta": "pp"},
    {"coluna": "taxa_desocupacao_trimestral_pct", "titulo": "Taxa de desocupação", "tipo": "percentual", "delta": "pp"},
    {"coluna": "rendimento_medio_real_brl", "titulo": "Rendimento médio real", "tipo": "brl", "delta": "percentual"},
]

# Frequência original exibida no dashboard.
SERIES_TRIMESTRAIS = SERIES_TRIMESTRAIS_CARREGADAS | {
    # IFData / concentração do mercado de crédito
    "hhi_credito_0_1",
    "hhi_credito_0_10000",
    "cr5_credito",
    "n_instituicoes_credito",

    # IFData / cartão de crédito: observações trimestrais preservadas sem ffill
    "total_cartao",
    "total_cartao_ativo",
    "ihhn_cartao_0_1",
    "ihhn_cartao_0_10000",

    # Derivadas de séries trimestrais: mantêm a frequência da série de origem
    "dlog_total_cartao",
    "delta_hhi_credito",
    "delta_hhi_cartao",
}

# Direção econômica usada apenas para colorir a variação dos KPIs.
# "alta_boa": aumento verde; "baixa_boa": queda verde.
DIRECAO_KPI = {
    "juros_cartao_pf_aa_pct": "baixa_boa",
    "spread_cartao_pf": "baixa_boa",
    "hhi_credito_0_1": "baixa_boa",
    "share_fintechs": "alta_boa",
    "inadimplencia_90d_pct": "baixa_boa",
    "saldo_cartao_pf_total_mr": "alta_boa",
    "comprometimento_renda_pct": "baixa_boa",
    "selic_am_pct": "baixa_boa",
    "taxa_desocupacao_trimestral_pct": "baixa_boa",
    "rendimento_medio_real_brl": "alta_boa",
}

CORES_METRICAS = {
    "juros_cartao_pf_aa_pct": "#ff6b6b",
    "spread_cartao_pf": "#ffa94d",
    "hhi_credito_0_1": "#845ef7",
    "cr5_credito": "#9775fa",
    "share_fintechs": "#20c997",
    "inadimplencia_90d_pct": "#f06595",
    "saldo_cartao_pf_total_mr": "#4dabf7",
    "comprometimento_renda_pct": "#e64980",
    "selic_am_pct": "#fcc419",
    "taxa_desocupacao_trimestral_pct": "#ff922b",
    "rendimento_medio_real_brl": "#51cf66",
    "ipca_mensal_pct": "#74c0fc",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def detectar_coluna_data(df: pd.DataFrame):
    for c in ["date", "data"]:
        if c in df.columns:
            return c
    return None


COL_DATA_MENSAL = detectar_coluna_data(mensal)
COL_DATA_TRIM = detectar_coluna_data(trimestral)


def formatar_numero(valor, tipo="numero"):
    if pd.isna(valor):
        return "—"
    valor = float(valor)
    if tipo == "percentual":
        return f"{valor:.2f}%".replace(".", ",")
    if tipo == "percentual_fracao":
        return f"{valor * 100:.2f}%".replace(".", ",")
    if tipo == "brl":
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if tipo == "milhoes":
        return f"{valor:,.0f} mi".replace(",", ".")
    if tipo == "indice":
        return f"{valor:.4f}".replace(".", ",")
    if tipo == "inteiro":
        return f"{valor:,.0f}".replace(",", ".")
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def preparar_serie(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    if df.empty or coluna not in df.columns:
        return pd.DataFrame()

    col_data = detectar_coluna_data(df)
    if col_data is None:
        return pd.DataFrame()

    aux = df[[col_data, coluna]].copy()
    aux[col_data] = pd.to_datetime(aux[col_data], errors="coerce")
    aux[coluna] = pd.to_numeric(aux[coluna], errors="coerce")
    return aux.dropna().sort_values(col_data)


def base_da_metrica(coluna: str):
    """Retorna a base mais adequada para a frequência original da métrica.

    Para séries trimestrais, prefere a base trimestral somente quando a
    coluna existe e contém ao menos uma observação numérica válida. Isso
    evita cards vazios quando uma coluna existe apenas estruturalmente em
    uma das bases.
    """
    if coluna in SERIES_TRIMESTRAIS:
        if (
            not trimestral.empty
            and coluna in trimestral.columns
            and pd.to_numeric(trimestral[coluna], errors="coerce").notna().any()
        ):
            return trimestral
    return mensal


def ultimo_e_anterior_da_base(coluna: str):
    base = base_da_metrica(coluna)
    aux = preparar_serie(base, coluna)
    if aux.empty:
        return np.nan, None, np.nan, None

    col_data = detectar_coluna_data(aux)
    atual = aux.iloc[-1]
    anterior = aux.iloc[-2] if len(aux) >= 2 else None

    return (
        atual[coluna],
        atual[col_data],
        anterior[coluna] if anterior is not None else np.nan,
        anterior[col_data] if anterior is not None else None,
    )


def buscar_snapshot(metrica: str):
    if snapshot.empty:
        return None

    possiveis = ["metrica", "indicador", "variavel"]
    col_metrica = next((c for c in possiveis if c in snapshot.columns), None)
    if col_metrica is None:
        return None

    aux = snapshot.loc[snapshot[col_metrica].astype(str) == str(metrica)]
    if aux.empty:
        return None
    return aux.iloc[-1]


def registro_kpi(metrica: str):
    reg = buscar_snapshot(metrica)
    if reg is not None:
        valor = pd.to_numeric(reg.get("valor", np.nan), errors="coerce")
        anterior = pd.to_numeric(reg.get("valor_anterior", np.nan), errors="coerce")
        var_abs = pd.to_numeric(reg.get("variacao_absoluta", np.nan), errors="coerce")
        var_pct = pd.to_numeric(reg.get("variacao_percentual", np.nan), errors="coerce")
        periodo = reg.get("periodo_fonte", reg.get("periodo", "—"))
        periodo_ant = reg.get("periodo_anterior", "—")

        # Se o snapshot não tiver campos completos, calcula a partir da base.
        if pd.isna(valor) or pd.isna(var_abs):
            v, p, va, pa = ultimo_e_anterior_da_base(metrica)
            if pd.isna(valor):
                valor = v
            if pd.isna(anterior):
                anterior = va
            if pd.isna(var_abs) and pd.notna(v) and pd.notna(va):
                var_abs = v - va
            if pd.isna(var_pct) and pd.notna(v) and pd.notna(va) and va != 0:
                var_pct = (v / va - 1) * 100
            if periodo is None or str(periodo) in {"nan", "NaT", "—"}:
                periodo = p
            if periodo_ant is None or str(periodo_ant) in {"nan", "NaT", "—"}:
                periodo_ant = pa

        return valor, periodo, anterior, periodo_ant, var_abs, var_pct

    v, p, va, pa = ultimo_e_anterior_da_base(metrica)
    var_abs = v - va if pd.notna(v) and pd.notna(va) else np.nan
    var_pct = (v / va - 1) * 100 if pd.notna(v) and pd.notna(va) and va != 0 else np.nan
    return v, p, va, pa, var_abs, var_pct


def formatar_periodo(periodo, metrica=None):
    if periodo is None or (isinstance(periodo, float) and pd.isna(periodo)):
        return "—"

    # Mantém labels prontos como 2026Q1.
    texto = str(periodo)
    if "Q" in texto and len(texto) <= 8:
        return texto

    try:
        dt = pd.to_datetime(periodo)
        if metrica in SERIES_TRIMESTRAIS:
            return str(dt.to_period("Q"))
        return dt.strftime("%m/%Y")
    except Exception:
        return texto


def formatar_delta(var_abs, var_pct, periodo_ant, regra):
    if pd.isna(var_abs):
        return "Sem comparação anterior"

    seta = "↑" if var_abs > 0 else "↓" if var_abs < 0 else "→"
    periodo_txt = formatar_periodo(periodo_ant)

    if regra == "pp":
        texto = f"{seta} {abs(var_abs):.2f} p.p. vs. {periodo_txt}"
    elif regra == "pp_fracao":
        texto = f"{seta} {abs(var_abs) * 100:.2f} p.p. vs. {periodo_txt}"
    elif regra == "percentual":
        if pd.notna(var_pct):
            texto = f"{seta} {abs(var_pct):.2f}% vs. {periodo_txt}"
        else:
            texto = f"{seta} {abs(var_abs):.2f} vs. {periodo_txt}"
    elif regra == "absoluto":
        texto = f"{seta} {abs(var_abs):.4f} vs. {periodo_txt}"
    else:
        texto = f"{seta} {abs(var_abs):.2f} vs. {periodo_txt}"

    return texto.replace(".", ",")


def classe_delta_semantica(coluna, var_abs):
    if pd.isna(var_abs) or float(var_abs) == 0:
        return "delta-neutral"
    regra = DIRECAO_KPI.get(coluna)
    if regra == "alta_boa":
        return "delta-good" if var_abs > 0 else "delta-bad"
    if regra == "baixa_boa":
        return "delta-good" if var_abs < 0 else "delta-bad"
    return "delta-neutral"


SERIES_ANUAIS = {"cartoes_emitidos_milhoes"}

ICONE_METRICA = {
    "juros_cartao_pf_aa_pct": '<svg viewBox="0 0 24 24"><path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20V7"/></svg>',
    "spread_cartao_pf": '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v5c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/><path d="M5 10v5c0 1.7 3.1 3 7 3s7-1.3 7-3v-5"/></svg>',
    "hhi_credito_0_1": '<svg viewBox="0 0 24 24"><path d="M12 2v10h10A10 10 0 1 1 12 2Z"/><path d="M15 2.5A10 10 0 0 1 21.5 9H15Z"/></svg>',
    "share_fintechs": '<svg viewBox="0 0 24 24"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "inadimplencia_90d_pct": '<svg viewBox="0 0 24 24"><path d="M10.3 2.9 1.8 17a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 2.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "saldo_cartao_pf_total_mr": '<svg viewBox="0 0 24 24"><rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20"/><path d="M6 15h4"/></svg>',
    "comprometimento_renda_pct": '<svg viewBox="0 0 24 24"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M19 8v6"/><path d="M16 11h6"/></svg>',
    "selic_am_pct": '<svg viewBox="0 0 24 24"><line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/></svg>',
    "taxa_desocupacao_trimestral_pct": '<svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M3 12h18"/><path d="M10 12v2h4v-2"/></svg>',
    "rendimento_medio_real_brl": '<svg viewBox="0 0 24 24"><path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20V7"/></svg>',
}

SUBTITULO_GRAFICO = {
    "juros_cartao_pf_aa_pct": "Taxa média de juros do cartão de crédito (PF)",
    "spread_cartao_pf": "Diferença entre juros do cartão PF e Selic anualizada",
    "hhi_credito_0_1": "Índice de concentração do mercado de crédito",
    "share_fintechs": "Participação das fintechs no mercado de crédito",
    "inadimplencia_90d_pct": "Percentual da carteira com atraso superior a 90 dias",
    "saldo_cartao_pf_total_mr": "Saldo total da carteira de cartão de crédito (PF)",
    "comprometimento_renda_pct": "Percentual da renda comprometida com serviço da dívida",
    "selic_am_pct": "Taxa Selic acumulada no mês",
    "taxa_desocupacao_trimestral_pct": "Taxa de desocupação da população economicamente ativa",
    "rendimento_medio_real_brl": "Rendimento médio real do trabalho",
}

def frequencia_metrica(coluna):
    if coluna in SERIES_ANUAIS: return "anual"
    if coluna in SERIES_TRIMESTRAIS: return "trimestral"
    return "mensal"

def classe_frequencia(coluna):
    return {"mensal":"monthly", "trimestral":"quarterly", "anual":"annual"}[frequencia_metrica(coluna)]

def render_kpi(titulo, coluna, tipo="numero", delta="absoluto"):
    valor, periodo, _, periodo_ant, var_abs, var_pct = registro_kpi(coluna)
    delta_texto = formatar_delta(var_abs, var_pct, periodo_ant, delta)
    delta_classe = classe_delta_semantica(coluna, var_abs)
    freq = frequencia_metrica(coluna); freq_css = classe_frequencia(coluna)
    icone = ICONE_METRICA.get(coluna, '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/></svg>')
    if " vs. " in delta_texto:
        delta_principal, vs_txt = delta_texto.split(" vs. ", 1); vs_txt = f"vs. {vs_txt}"
    else:
        delta_principal, vs_txt = delta_texto, ""
    periodo_texto = f"Última obs.: {formatar_periodo(periodo, coluna)} · {freq}"
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top"><div class="kpi-name-wrap"><span class="kpi-icon icon-{freq_css}">{icone}</span><span class="kpi-label">{titulo}</span></div><span class="freq-badge freq-{freq_css}">{freq.capitalize()}</span></div>
            <div class="kpi-value">{formatar_numero(valor, tipo)}</div>
            <div class="kpi-delta {delta_classe}">{delta_principal}</div>
            <div class="kpi-vs">{vs_txt}</div>
            <div class="kpi-period">{periodo_texto}</div>
        </div>""", unsafe_allow_html=True)


def adicionar_periodos_pandemia(fig):
    fig.add_vrect(x0="2020-01-01", x1="2021-12-31", fillcolor="rgba(201,64,151,.24)", line=dict(color="rgba(230,94,181,.30)", width=1), opacity=1, annotation_text="Pandemia<br>2020–2021", annotation_position="top left", annotation_font=dict(size=10, color="#d9c6d8"))
    return fig


def hex_para_rgba(hex_color, alpha=0.14):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def criar_fig_serie(
    df,
    coluna,
    titulo,
    ylabel="",
    altura=250,
    percentual_fracao=False,
    inicio=None,
    fim=None,
    cor_override=None,
):
    aux = preparar_serie(df, coluna)
    if aux.empty:
        return None
    col_data = detectar_coluna_data(aux)
    if inicio is not None:
        aux = aux[aux[col_data] >= pd.Timestamp(inicio)]
    if fim is not None:
        aux = aux[aux[col_data] <= pd.Timestamp(fim)]
    if aux.empty:
        return None

    y = coluna
    if percentual_fracao:
        y = "valor_exibicao"
        aux[y] = aux[coluna] * 100

    freq = frequencia_metrica(coluna)
    cor = cor_override or {
        "mensal": "#ff5d73",
        "trimestral": "#6385ff",
        "anual": "#f5c84c",
    }[freq]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aux[col_data], y=aux[y], mode="lines",
        line=dict(color=cor, width=2.6, shape="spline", smoothing=.45),
        fill="tozeroy", fillcolor=hex_para_rgba(cor, .18),
        hovertemplate="%{x|%b/%Y}<br><b>%{y:,.2f}</b><extra></extra>",
        showlegend=False,
    ))
    ultimo = aux.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[ultimo[col_data]], y=[ultimo[y]], mode="markers",
        marker=dict(color=cor, size=7), hoverinfo="skip", showlegend=False,
    ))
    fig = adicionar_periodos_pandemia(fig)
    fig.update_layout(
        height=altura, margin=dict(l=8, r=6, t=12, b=4), hovermode="x unified",
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a9bacb", size=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(112,143,168,.10)", zeroline=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="rgba(112,143,168,.11)", zeroline=False, title=ylabel),
    )
    return fig


def grafico_serie(df, coluna, titulo, ylabel="", altura=390, percentual_fracao=False):
    fig = criar_fig_serie(df,coluna,titulo,ylabel,altura,percentual_fracao)
    if fig is None:
        st.info(f"Sem dados disponíveis para {titulo}."); return
    st.markdown(f"**{titulo}**")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})


def render_grafico_card(coluna, titulo, ylabel, tipo, delta, percentual_fracao=False, inicio=None, fim=None):
    base = base_da_metrica(coluna); css = classe_frequencia(coluna)
    icone = ICONE_METRICA.get(coluna, '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/></svg>'); subtitulo = SUBTITULO_GRAFICO.get(coluna, "")
    valor, periodo, _, periodo_ant, var_abs, var_pct = registro_kpi(coluna)
    delta_texto = formatar_delta(var_abs,var_pct,periodo_ant,delta)
    delta_class = classe_delta_semantica(coluna,var_abs)
    fig = criar_fig_serie(base,coluna,titulo,ylabel,245,percentual_fracao,inicio,fim)
    with st.container(border=True):
        st.markdown(f'''<div class="chart-title-row"><span class="chart-icon icon-{css}">{icone}</span><div><div class="chart-title">{titulo}</div><div class="chart-subtitle">{subtitulo}</div></div></div>''', unsafe_allow_html=True)
        cplot,cinfo = st.columns([3.15,1],gap="small")
        with cplot:
            if fig is not None: st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
            else: st.info("Sem dados no período selecionado.")
        with cinfo:
            st.markdown(f'''<div style="padding-top:2.1rem"><div class="chart-side-period">{formatar_periodo(periodo,coluna)}</div><div class="chart-side-value">{formatar_numero(valor,tipo)}</div><div class="chart-side-delta {delta_class}">{delta_texto.split(' vs. ')[0]}</div><div class="chart-side-vs">● vs. {formatar_periodo(periodo_ant,coluna)}</div></div>''', unsafe_allow_html=True)



def grupo_metrica(metrica):
    if not dicionario.empty and {"metrica", "grupo"}.issubset(dicionario.columns):
        aux = dicionario.loc[dicionario["metrica"].astype(str) == str(metrica), "grupo"]
        if not aux.empty and pd.notna(aux.iloc[-1]):
            return str(aux.iloc[-1])
    return "Outros"


def rotulo_metrica_frequencia(metrica):
    return f"{nome_metrica(metrica)} · {frequencia_metrica(metrica).capitalize()}"


def percentual_fracao_metrica(metrica):
    return metrica.startswith("share_") or metrica in {"cr5_credito"}


# Colunas auxiliares de calendário/modelagem não são indicadores econômicos
# e não devem aparecer no seletor de métricas nem no "selecionar todos".
COLUNAS_TECNICAS = {
    "ano",
    "mes",
    "mês",
    "periodo_mes",
    "periodo mês",
    "trimestre",
    "periodo_trimestre",
    "periodo pandemia",
    "periodo_pandemia",
    "pandemia",
    "year",
    "month",
    "quarter",
    "trimestre_num",
}


def eh_coluna_tecnica(coluna):
    nome = str(coluna).strip().lower()
    return nome in COLUNAS_TECNICAS


def coluna_tem_dados_validos(df, coluna):
    if df.empty or coluna not in df.columns:
        return False
    return pd.to_numeric(df[coluna], errors="coerce").notna().any()


def metricas_validas(df):
    """Lista apenas indicadores numéricos com pelo menos um valor válido."""
    if df.empty:
        return []

    col_data = detectar_coluna_data(df)
    saida = []

    for coluna in df.columns:
        if coluna == col_data or eh_coluna_tecnica(coluna):
            continue
        if coluna_tem_dados_validos(df, coluna):
            saida.append(coluna)

    return saida


def render_grade_graficos(metricas, base_func, frequencia_forcada=None, key_prefix="grafico"):

    if not metricas:
        st.info("Selecione pelo menos um indicador para visualizar os gráficos.")
        return

    for i in range(0, len(metricas), 2):
        slots = st.columns(2, gap="small")
        for slot, metrica in zip(slots, metricas[i:i+2]):
            with slot:
                base = base_func(metrica)
                freq_real = frequencia_metrica(metrica)
                freq_exibida = frequencia_forcada or freq_real
                cor = {"mensal": "#ff5d73", "trimestral": "#6385ff", "anual": "#f5c84c"}[freq_exibida]

                fig = criar_fig_serie(
                    base,
                    metrica,
                    nome_metrica(metrica),
                    altura=310,
                    percentual_fracao=percentual_fracao_metrica(metrica),
                    cor_override=cor,
                )

                with st.container(border=True):
                    html = (
                        f'<div class="chart-title-row">'
                        f'<span class="chart-icon" style="color:{cor}">●</span>'
                        f'<div><div class="chart-title">{nome_metrica(metrica)}</div>'
                        f'<div class="chart-subtitle">{grupo_metrica(metrica)} · {freq_exibida.capitalize()}</div>'
                        f'</div></div>'
                    )
                    st.markdown(html, unsafe_allow_html=True)
                    if fig is None:
                        st.info("Sem dados disponíveis para este indicador.")
                    else:
                        st.plotly_chart(
                            fig,
                            width="stretch",
                            config={"displayModeBar": False},
                            key=f"{key_prefix}_{freq_exibida}_{metrica}_{i}",
                        )

def metricas_numericas(df):
    if df.empty:
        return []
    col_data = detectar_coluna_data(df)
    saida = []
    for c in df.columns:
        if c == col_data or eh_coluna_tecnica(c):
            continue
        convertido = pd.to_numeric(df[c], errors="coerce")
        if convertido.notna().sum() >= 3:
            saida.append(c)
    return saida



# ============================================================
# MONITORAMENTO PREDITIVO V13 — REGISTRO PERSISTENTE
# ============================================================

CAMPOS_V13 = [
    "id_previsao", "metrica", "periodo_previsto", "periodo_ultimo_dado",
    "modelo", "valor_previsto", "valor_realizado", "data_registro",
    "data_referencia_emissao", "tipo_previsao", "origem", "versao_modelo",
    "fonte_realizado", "data_publicacao", "data_conferencia", "observacoes",
]


def ler_historico_v13():
    if not ARQ_HISTORICO_V13.exists():
        return pd.DataFrame(columns=CAMPOS_V13)
    try:
        df = pd.read_csv(ARQ_HISTORICO_V13, dtype={"id_previsao": str})
        for campo in CAMPOS_V13:
            if campo not in df.columns:
                df[campo] = pd.NA
        return df[CAMPOS_V13]
    except Exception as exc:
        st.error(f"Falha ao ler histórico v13: {exc}")
        return pd.DataFrame(columns=CAMPOS_V13)


def salvar_historico_v13(df):
    """Não modifica registros anteriores; cria backup e usa substituição atômica."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ARQ_HISTORICO_V13.exists():
        backup = DATA_DIR / "backups_previsoes"
        backup.mkdir(exist_ok=True)
        momento = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shutil.copy2(ARQ_HISTORICO_V13, backup / f"historico_previsoes_v13_{momento}.csv")
    temp = ARQ_HISTORICO_V13.with_suffix(".csv.tmp")
    try:
        df.to_csv(temp, index=False)
        os.replace(temp, ARQ_HISTORICO_V13)
    finally:
        if temp.exists():
            temp.unlink()


def id_da_previsao(linha):
    # Identificador independente do caminho do arquivo.
    partes = [str(linha.get(k, "")) for k in [
        "metrica",
        "periodo_previsto",
        "periodo_ultimo_dado",
        "modelo",
        "valor_previsto",
        "versao_modelo",
    ]]
    return hashlib.sha256("|".join(partes).encode()).hexdigest()[:20]


def importar_snapshot_4e(arquivo, historico):
    if not arquivo.exists():
        return historico, 0

    origem_df = pd.read_csv(arquivo)

    obrigatorias = [
        "metrica",
        "periodo_previsto",
        "periodo_ultimo_dado",
        "modelo_selecionado",
        "previsao_selecionada",
    ]

    faltantes = [
        coluna
        for coluna in obrigatorias
        if coluna not in origem_df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas ausentes no Modelo 4E: {faltantes}"
        )

    def chave_semantica(registro):
        valor = pd.to_numeric(
            registro.get("valor_previsto"),
            errors="coerce"
        )

        if pd.isna(valor):
            return None

        return (
            str(registro.get("metrica", "")),
            str(registro.get("periodo_previsto", "")),
            str(registro.get("periodo_ultimo_dado", "")),
            str(registro.get("modelo", "")),
            float(valor),
        )

    existentes = set(
        historico["id_previsao"]
        .dropna()
        .astype(str)
    )

    chaves_existentes = {
        chave
        for _, registro in historico.iterrows()
        if (chave := chave_semantica(registro)) is not None
    }

    entradas = []
    data_registro = datetime.now(
        timezone.utc
    ).isoformat()

    for _, row in origem_df.iterrows():
        previsto = pd.to_numeric(
            row["previsao_selecionada"],
            errors="coerce"
        )

        if pd.isna(previsto):
            continue

        novo = dict.fromkeys(CAMPOS_V13, pd.NA)

        novo.update({
            "metrica": str(row["metrica"]),
            "periodo_previsto": str(row["periodo_previsto"]),
            "periodo_ultimo_dado": str(row["periodo_ultimo_dado"]),
            "modelo": str(row["modelo_selecionado"]),
            "valor_previsto": float(previsto),
            "data_registro": data_registro,
            "tipo_previsao": "importada_sem_emissao_comprovada",
            "origem": str(arquivo),
            "versao_modelo": "Modelo 4E — snapshot CSV",
            "observacoes": (
                "Data de geração original não comprovada. "
                "Não classificar como previsão prospectiva auditada."
            ),
        })

        chave = chave_semantica(novo)

        # Evita reimportar a mesma previsão,
        # independentemente do ID ou caminho da origem.
        if chave in chaves_existentes:
            continue

        novo["id_previsao"] = id_da_previsao(novo)

        if novo["id_previsao"] in existentes:
            continue

        entradas.append(novo)
        existentes.add(novo["id_previsao"])
        chaves_existentes.add(chave)

    if entradas:
        historico = pd.concat(
            [historico, pd.DataFrame(entradas)],
            ignore_index=True
        )
        salvar_historico_v13(historico)

    return historico, len(entradas)


def status_v13(linha):
    if str(linha.get("metrica")) in ("hhi_credito_0_1", "cr5_credito") and pd.isna(linha.get("data_conferencia")):
        return "🟠 Em auditoria temporal"
    if pd.notna(pd.to_numeric(linha.get("valor_realizado"), errors="coerce")):
        return "🟢 Conciliado (base local)"
    try:
        fim = pd.Period(str(linha["periodo_previsto"]), freq="Q").end_time.date()
        if datetime.now().date() <= fim:
            return "🔵 Em andamento"
    except Exception:
        pass
    return "🟡 Aguardando publicação"


def formatar_valor_v13(metrica, valor):
    n = pd.to_numeric(valor, errors="coerce")
    if pd.isna(n):
        return "—"
    if metrica == "hhi_credito_0_1":
        return f"{n:.4f}"
    if metrica == "cr5_credito":
        return f"{n*100:.2f}%"
    return f"{n:.2f}%"


def atualizar_realizados_automaticamente(historico, base_trimestral):
    """Concilia somente observações existentes no CSV trimestral atualizado.

    Não consulta a internet: a atualização da fonte ocorre no pipeline do projeto.
    HHI/CR5 ficam bloqueados até auditoria de horizonte e definição.
    """
    if historico.empty or base_trimestral.empty:
        return historico, 0
    coluna_data = detectar_coluna_data(base_trimestral)
    if not coluna_data:
        return historico, 0
    dados = base_trimestral.copy()
    datas = pd.to_datetime(dados[coluna_data], errors="coerce")
    dados["_periodo_q"] = datas.dt.to_period("Q").astype("string")
    dados = dados.dropna(subset=["_periodo_q"])
    if dados.empty:
        return historico, 0
    atualizado = historico.copy()
    qtd = 0
    for i, reg in atualizado.iterrows():
        met = str(reg.get("metrica", ""))
        trimestre = str(reg.get("periodo_previsto", ""))
        if met in {"hhi_credito_0_1", "cr5_credito"}:
            continue  # Em auditoria temporal: não preencher automaticamente.
        if met not in dados.columns:
            continue
        if pd.notna(pd.to_numeric(reg.get("valor_realizado"), errors="coerce")):
            continue  # Nunca substituir resultado já registrado.
        if not trimestre or trimestre.lower() == "nan":
            continue
        candidatos = dados.loc[dados["_periodo_q"] == trimestre, met]
        valores = pd.to_numeric(candidatos, errors="coerce").dropna()
        if len(valores) != 1:
            # Exige um valor trimestral inequívoco; não usa último mês arbitrariamente.
            continue
        valor = float(valores.iloc[0])
        atualizado.at[i, "valor_realizado"] = valor
        atualizado.at[i, "fonte_realizado"] = f"Base local consolidada: {ARQ_TRIMESTRAL.name} (pipeline BCB/IFData/IBGE)"
        atualizado.at[i, "data_publicacao"] = pd.NA  # Data oficial não inferível do CSV.
        atualizado.at[i, "data_conferencia"] = datetime.now(timezone.utc).isoformat()
        atualizado.at[i, "observacoes"] = (
            str(reg.get("observacoes", "") if pd.notna(reg.get("observacoes")) else "")
            + " | Realizado importado automaticamente da base trimestral local; publicação e definição devem ser auditadas."
        ).strip()
        qtd += 1
    if qtd:
        salvar_historico_v13(atualizado)
    return atualizado, qtd


def painel_historico_v13(arquivo_futuro):
    st.markdown("### Monitoramento histórico — Previsto × Realizado")
    st.caption(
        "As previsões são preservadas ao longo dos trimestres. Os valores realizados são conciliados "
        "automaticamente com a base trimestral quando o pipeline disponibiliza observações válidas. "
        "Não há digitação manual."
    )
    hist = ler_historico_v13()
    # ETAPA 174 — Proteção da integridade histórica.
    # O painel é somente de leitura: não importa previsões nem
    # concilia realizados automaticamente durante sua renderização.
    # Qualquer atualização do histórico exige processo auditado separado.
    st.caption(
        "Histórico em modo somente leitura. "
        "Importações e conciliações automáticas estão desativadas."
    )
    if hist.empty:
        st.info("Nenhuma previsão disponível no CSV do Modelo 4E. Execute primeiro o pipeline preditivo.")
        return
    hist["valor_previsto"] = pd.to_numeric(hist["valor_previsto"], errors="coerce")
    hist["valor_realizado"] = pd.to_numeric(hist["valor_realizado"], errors="coerce")

    # ETAPA 199 — Realizados observados apenas para visualização.
    # Não altera o CSV histórico nem comprova emissão prospectiva.
    hist["_realizado_observado_199"] = pd.NA

    if not trimestral.empty:
        _data_199 = detectar_coluna_data(trimestral)

        if _data_199:
            _base_199 = trimestral.copy()
            _base_199["_trimestre_199"] = (
                pd.to_datetime(_base_199[_data_199], errors="coerce")
                .dt.to_period("Q")
                .astype("string")
            )

            for _metrica_199 in [
                "juros_cartao_pf_aa_pct",
                "inadimplencia_90d_pct",
            ]:
                if _metrica_199 not in _base_199.columns:
                    continue

                _serie_199 = _base_199[
                    ["_trimestre_199", _metrica_199]
                ].copy()

                _serie_199[_metrica_199] = pd.to_numeric(
                    _serie_199[_metrica_199],
                    errors="coerce",
                )

                _serie_199 = _serie_199.dropna()

                # Aceitar somente um registro por trimestre.
                _contagens_199 = _serie_199.groupby(
                    "_trimestre_199"
                ).size()

                _validos_199 = _contagens_199[
                    _contagens_199 == 1
                ].index

                _serie_199 = _serie_199[
                    _serie_199["_trimestre_199"].isin(_validos_199)
                ]

                _mapa_199 = dict(
                    zip(
                        _serie_199["_trimestre_199"].astype(str),
                        _serie_199[_metrica_199],
                    )
                )

                _mascara_199 = (
                    hist["metrica"].eq(_metrica_199)
                    & hist["valor_realizado"].isna()
                )

                hist.loc[
                    _mascara_199,
                    "_realizado_observado_199",
                ] = (
                    hist.loc[_mascara_199, "periodo_previsto"]
                    .astype(str)
                    .map(_mapa_199)
                )

    hist["status"] = hist.apply(status_v13, axis=1)

    # Não calcular erros sem comprovação temporal da previsão.
    hist["erro_absoluto"] = pd.NA
    metricas = [m for m in ["juros_cartao_pf_aa_pct", "inadimplencia_90d_pct", "hhi_credito_0_1", "cr5_credito"] if m in hist["metrica"].values or m in trimestral.columns]
    metricas += [m for m in sorted(hist["metrica"].dropna().unique()) if m not in metricas]
    periodos = sorted(hist["periodo_previsto"].dropna().unique(), reverse=True)
    periodo = st.selectbox("Filtrar trimestre da tabela", ["Todos"] + periodos, key="v15_periodo")

    st.markdown("#### Histórico realizado e previsões por indicador")
    st.caption("Azul: série histórica observada na base trimestral. Laranja: previsões registradas pelo Modelo 4E. "
               "Os dados realizados anteriores ao período previsto também são exibidos, mesmo quando não existia previsão. "
               "Resultados futuros aparecem automaticamente após a atualização da base consolidada.")
    col_data = detectar_coluna_data(trimestral)
    dados = trimestral.copy()
    if col_data:
        dados["_quarter_v15"] = pd.to_datetime(dados[col_data], errors="coerce").dt.to_period("Q").astype("string")
    else:
        dados["_quarter_v15"] = pd.Series(dtype="string")
    for metrica in metricas:
        st.markdown(f"##### {nome_metrica(metrica)}")
        serie = hist.loc[hist["metrica"] == metrica].copy().sort_values(["periodo_previsto", "data_registro"])
        # Exibe apenas a primeira emissão no gráfico. Todas permanecem na tabela e no CSV.
        serie = serie.drop_duplicates("periodo_previsto", keep="first")
        if metrica in dados.columns:
            realizados = dados[["_quarter_v15", metrica]].copy()
            realizados[metrica] = pd.to_numeric(realizados[metrica], errors="coerce")
            realizados = realizados.dropna(subset=["_quarter_v15", metrica])
            # Se houver mais de um registro por trimestre, não inventar agregação.
            contagens = realizados.groupby("_quarter_v15").size()
            realizados = realizados[realizados["_quarter_v15"].isin(contagens[contagens == 1].index)]
            realizados = realizados.sort_values("_quarter_v15")
            eixo_real = realizados["_quarter_v15"].astype(str).tolist()
            valor_real = realizados[metrica].tolist()
        else:
            eixo_real, valor_real = [], []
        # Não incorporar no gráfico observações futuras quando a métrica está em auditoria temporal:
        # ainda assim mostrar a série histórica observada da base, sem chamar de conciliação.
        eixo_prev = serie["periodo_previsto"].astype(str).tolist()
        valor_prev = serie["valor_previsto"].tolist()
        if not eixo_real and not eixo_prev:
            st.info("Sem valores para este indicador.")
            continue
        # Eixo trimestral compartilhado, incluindo trimestres intermediários sem informação.
        qs = []
        for valor in eixo_real + eixo_prev:
            try:
                qs.append(pd.Period(valor, freq="Q"))
            except (ValueError, TypeError):
                continue
        eixo = [str(q) for q in pd.period_range(min(qs), max(qs), freq="Q")] if qs else sorted(set(eixo_real + eixo_prev))
        reais_map = dict(zip(eixo_real, valor_real))
        prev_map = dict(zip(eixo_prev, valor_prev))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eixo, y=[reais_map.get(q) for q in eixo], mode="lines+markers",
            name="Realizado (histórico)", line=dict(color="#52c8ff", width=3),
            marker=dict(size=5), connectgaps=False,
            hovertemplate="%{x}<br>Realizado: %{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=eixo, y=[prev_map.get(q) for q in eixo], mode="lines+markers",
            name="Previsto", line=dict(color="#ffab55", width=3, dash="dash"),
            marker=dict(size=9), connectgaps=False,
            hovertemplate="%{x}<br>Previsto: %{y:.4f}<extra></extra>",
        ))
        unidade = "Índice (0–1)" if metrica in ("cr5_credito", "hhi_credito_0_1") else "%"
        fig.update_layout(
            height=355, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
            xaxis_title="Trimestre", yaxis_title=unidade,
            legend=dict(orientation="h", y=1.14),
            margin=dict(l=20, r=20, t=55, b=40),
        )
        # Mostrar últimos trimestres por padrão, preservando a navegação/zoom para todo o histórico.
        if len(eixo) > 16:
            fig.update_xaxes(range=[len(eixo)-16.5, len(eixo)-0.5])
        st.plotly_chart(fig, width="stretch", key=f"v15_grafico_{metrica}")
        if metrica in {"hhi_credito_0_1", "cr5_credito"}:
            st.caption("A série observada é exibida para contexto; a conciliação das previsões permanece bloqueada até a auditoria temporal de HHI/CR5.")
        elif not any(pd.notna(v) for v in serie["valor_realizado"]):
            st.caption("As previsões ainda não possuem valor realizado conciliado. A linha azul representa apenas observações históricas disponíveis.")

    st.markdown("#### Histórico completo — tabela")
    vis = hist.copy().sort_values(["periodo_previsto", "metrica", "data_registro"], ascending=[False, True, True])
    if periodo != "Todos":
        vis = vis[vis["periodo_previsto"] == periodo]
    tabela = vis[["periodo_previsto", "metrica", "modelo", "valor_previsto", "valor_realizado", "erro_absoluto", "status", "tipo_previsao", "fonte_realizado"]].copy()

    # Exibição separada do registro histórico.
    _observado_199 = pd.to_numeric(
        vis["_realizado_observado_199"],
        errors="coerce",
    )

    _sem_realizado_199 = tabela["valor_realizado"].isna()
    _mostrar_199 = _sem_realizado_199 & _observado_199.notna()

    tabela.loc[
        _mostrar_199,
        "valor_realizado",
    ] = _observado_199.loc[_mostrar_199]

    tabela.loc[
        _mostrar_199,
        "status",
    ] = "Realizado disponível — previsão em auditoria temporal"

    # ETAPA 217 — A coluna pode ter sido inferida como float64
    # quando o histórico contém apenas valores ausentes.
    # Conversão restrita à tabela de apresentação.
    tabela["fonte_realizado"] = tabela["fonte_realizado"].astype("string")

    tabela.loc[
        _mostrar_199,
        "fonte_realizado",
    ] = "Base trimestral consolidada (somente visualização)"
    # ETAPA 214 — Erro absoluto descritivo, somente em memória.
    # Não altera os arquivos históricos e não valida emissão prospectiva.
    _previsto_214 = pd.to_numeric(
        tabela["valor_previsto"], errors="coerce"
    )
    _realizado_214 = pd.to_numeric(
        tabela["valor_realizado"], errors="coerce"
    )

    tabela["erro_absoluto"] = (
        _realizado_214 - _previsto_214
    ).abs()

    tabela["erro_absoluto"] = [
        (
            f"{e*100:.4f} p.p."
            if m == "cr5_credito"
            else f"{e:.6f}"
            if m == "hhi_credito_0_1"
            else f"{e:.6f} p.p."
        )
        if pd.notna(e) else "—"
        for m, e in zip(
            tabela["metrica"],
            tabela["erro_absoluto"],
        )
    ]

    for coluna in ["valor_previsto", "valor_realizado"]:
        tabela[coluna] = [
            formatar_valor_v13(m, v)
            for m, v in zip(
                tabela["metrica"],
                tabela[coluna],
            )
        ]
    # ETAPA 219 — Rótulos de apresentação, sem modificar o histórico.
    tabela["tipo_previsao"] = tabela["tipo_previsao"].replace({
        "importada_sem_emissao_comprovada":
            "Em auditoria — emissão original não comprovada",
        "reconstrucao_historica":
            "Reconstrução histórica",
        "prospectiva_pendente_auditoria":
            "Prospectiva — auditoria pendente",
        "prospectiva_auditada":
            "Prospectiva auditada",
    })

    tabela["fonte_realizado"] = (
        tabela["fonte_realizado"]
        .astype("string")
        .replace({"None": pd.NA, "nan": pd.NA, "": pd.NA})
        .fillna("Aguardando validação")
    )

    tabela.columns = ["Trimestre", "Indicador", "Modelo", "Previsto", "Realizado", "Erro absoluto descritivo", "Status", "Origem da previsão", "Fonte do realizado"]
    st.dataframe(tabela, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Exportar histórico completo (CSV)",
        hist.drop(columns=["status", "erro_absoluto"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="historico_previsoes_v13.csv", mime="text/csv", key="exportar_hist_v15",
    )
    # ETAPA 199 — Avaliação prospectiva bloqueada até auditoria
    # da data original de emissão das previsões.
    avaliados = hist.iloc[0:0].copy()
    if not avaliados.empty:
        avaliados["erro"] = avaliados["valor_realizado"] - avaliados["valor_previsto"]
        qualidade = avaliados.groupby(["metrica", "modelo"]).agg(
            MAE=("erro", lambda v: v.abs().mean()),
            RMSE=("erro", lambda v: np.sqrt(np.mean(np.square(v)))),
            Observacoes=("erro", "size"),
        ).reset_index()
        st.markdown("#### Erro acumulado por modelo")
        st.dataframe(qualidade.round(6), hide_index=True, width="stretch")
    else:
        st.info("Os erros absolutos descritivos são exibidos quando existem valores realizados disponíveis. MAE e RMSE prospectivos permanecem bloqueados até a validação temporal das previsões.")
    st.warning(
        "A sincronização lê os CSVs locais: para buscar publicações novas, execute ou automatize o pipeline de coleta. "
        "HHI e CR5 permanecem em auditoria temporal e não são conciliados automaticamente. "
        "Previsões sem data original comprovada não são evidência prospectiva auditada."
    )


# ============================================================
# CABEÇALHO E MENU
# ============================================================

pagina = st.sidebar.radio(
    "Navegação",
    [
        "Visão Geral",
        "Histórico",
        "Determinantes do Mercado",
        "Correlações",
        "Previsão Markov-Neural",
        "Validação Experimental v17",
        "Questões de Pesquisa",
        "Dicionário",
    ],
)

st.sidebar.markdown("---")
if not mensal.empty and COL_DATA_MENSAL:
    ultima_data = mensal[COL_DATA_MENSAL].max()
    if pd.notna(ultima_data):
        st.sidebar.caption(f"Base mensal até {ultima_data.strftime('%m/%Y')}")

if not trimestral.empty and COL_DATA_TRIM:
    ultima_tri = trimestral[COL_DATA_TRIM].max()
    if pd.notna(ultima_tri):
        st.sidebar.caption(f"Último trimestre completo: {ultima_tri.to_period('Q')}")


# Cabeçalho executivo
ultima_data_header = mensal[COL_DATA_MENSAL].max() if (not mensal.empty and COL_DATA_MENSAL) else None
periodo_header = ultima_data_header.strftime("%m/%Y") if pd.notna(ultima_data_header) else "—"
h1,h2 = st.columns([4.6,1.15], vertical_alignment="top")
with h1:
    st.markdown('<div class="dashboard-title">Mercado Brasileiro de Cartão de Crédito PF</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Dinâmica competitiva, condições econômicas, regimes de mercado e previsões</div>', unsafe_allow_html=True)
with h2:
    st.markdown(f'''<div class="update-wrap"><div class="update-box"><div class="update-label">Última atualização da base</div><div class="update-pill">▣ &nbsp; {periodo_header} (mensal)</div><div class="update-source">Fonte: BCB/SGS, IFData, IBGE/SIDRA</div></div></div>''', unsafe_allow_html=True)

# ============================================================
# 1. VISÃO GERAL
# ============================================================

if pagina == "Visão Geral":
    st.markdown('<div class="section-title">Estado atual do mercado</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Principais indicadores e variação em relação ao mês/trimestre anterior</div>', unsafe_allow_html=True)
    cols=st.columns(5,gap="small")
    for slot,kpi in zip(cols,KPIS_CAPA[:5]):
        with slot: render_kpi(kpi["titulo"],kpi["coluna"],kpi["tipo"],kpi["delta"])
    cols=st.columns(5,gap="small")
    for slot,kpi in zip(cols,KPIS_CAPA[5:]):
        with slot: render_kpi(kpi["titulo"],kpi["coluna"],kpi["tipo"],kpi["delta"])
    st.markdown("<hr>",unsafe_allow_html=True)
    e1,e2=st.columns([3.5,1.45],vertical_alignment="bottom")
    with e1:
        st.markdown('<div class="section-title">Evolução dos principais indicadores</div>', unsafe_allow_html=True)
    min_data=mensal[COL_DATA_MENSAL].min().date() if (not mensal.empty and COL_DATA_MENSAL) else None
    max_data=mensal[COL_DATA_MENSAL].max().date() if (not mensal.empty and COL_DATA_MENSAL) else None
    with e2:
        periodo_sel=st.date_input("Período", value=(min_data,max_data), min_value=min_data, max_value=max_data)
    if isinstance(periodo_sel,(list,tuple)) and len(periodo_sel)==2:
        inicio_sel,fim_sel=periodo_sel
    else:
        inicio_sel,fim_sel=min_data,max_data

    graficos_visao_geral = [
        ("juros_cartao_pf_aa_pct", "Juros cartão PF (aa)", "% a.a.", "percentual", "pp", False),
        ("spread_cartao_pf", "Spread cartão PF", "p.p.", "percentual", "pp", False),
        ("hhi_credito_0_1", "Concentração do mercado — HHI", "Índice", "indice", "absoluto", False),
        ("share_fintechs", "Participação das fintechs", "%", "percentual_fracao", "pp_fracao", True),
        ("inadimplencia_90d_pct", "Inadimplência > 90 dias", "%", "percentual", "pp", False),
        ("saldo_cartao_pf_total_mr", "Saldo total cartão PF", "R$ milhões", "milhoes", "percentual", False),
        ("comprometimento_renda_pct", "Comprometimento de renda", "%", "percentual", "pp", False),
        ("selic_am_pct", "Selic mensal", "%", "percentual", "pp", False),
        ("taxa_desocupacao_trimestral_pct", "Taxa de desocupação", "%", "percentual", "pp", False),
        ("rendimento_medio_real_brl", "Rendimento médio real", "R$", "brl", "percentual", False),
    ]

    for i in range(0, len(graficos_visao_geral), 3):
        slots = st.columns(3, gap="small")
        for slot, item in zip(slots, graficos_visao_geral[i:i+3]):
            with slot:
                render_grafico_card(*item, inicio=inicio_sel, fim=fim_sel)

    st.markdown('''<div class="legend-row"><div class="legend-item"><span class="legend-dot" style="background:#ff5d73"></span> Mensal (BCB/SGS)</div><div class="legend-item"><span class="legend-dot" style="background:#6385ff"></span> Trimestral (IFData / IBGE-SIDRA)</div><div class="legend-item"><span class="legend-dot" style="background:#f5c84c"></span> Anual</div><div class="legend-item"><span class="legend-square"></span> Período da pandemia (2020–2021)</div></div><div class="source-foot">Fonte: BCB/SGS, IFData, CVM, IBGE/SIDRA. Elaboração própria.</div>''', unsafe_allow_html=True)


# ============================================================
# 2. INDICADORES GERAIS
# ============================================================

elif pagina == "Histórico":
    st.markdown('<div class="section-title">Histórico dos Indicadores</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Visão detalhada das métricas na base original e da base trimestral preparada para os modelos.</div>',
        unsafe_allow_html=True,
    )

    aba_original, aba_trimestral = st.tabs([
        "Base detalhada",
        "Base Trimestral de Modelagem",
    ])

    with aba_original:
        st.caption(
            "Mostra as métricas na base detalhada, mantendo a periodicidade de origem de cada indicador. "
            "Mensal = coral, trimestral = azul e anual = amarelo."
        )

        if mensal.empty:
            st.warning("Base detalhada não disponível.")
        else:
            col_data = detectar_coluna_data(mensal)
            # Exibe apenas indicadores reais, com ao menos uma observação válida.
            # Campos auxiliares (Ano, Mês, Trimestre, Período Pandemia etc.)
            # ficam fora inclusive quando o usuário seleciona todos.
            metricas = metricas_validas(mensal)
            grupos_disp = sorted({grupo_metrica(m) for m in metricas})

            c1, c2 = st.columns([1.2, 1], vertical_alignment="bottom")
            with c1:
                grupo_original = st.selectbox(
                    "Grupo", ["Todos"] + grupos_disp, key="grupo_base_detalhada"
                )
            with c2:
                visual_original = st.radio(
                    "Visualização", ["Tabela", "Gráficos"], horizontal=True,
                    key="visual_base_detalhada"
                )

            if grupo_original != "Todos":
                metricas = [m for m in metricas if grupo_metrica(m) == grupo_original]

            selecionadas_original = st.multiselect(
                "Indicadores",
                metricas,
                default=metricas[:10],
                format_func=rotulo_metrica_frequencia,
                key="indicadores_base_detalhada",
            )

            st.markdown(
                '<div class="legend-row">'
                '<div class="legend-item"><span class="legend-dot" style="background:#ff5d73"></span> Mensal</div>'
                '<div class="legend-item"><span class="legend-dot" style="background:#6385ff"></span> Trimestral</div>'
                '<div class="legend-item"><span class="legend-dot" style="background:#f5c84c"></span> Anual</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if visual_original == "Tabela":
                cols = ([col_data] if col_data else []) + selecionadas_original
                if not selecionadas_original:
                    st.info("Selecione pelo menos um indicador.")
                else:
                    tabela_original = mensal[cols].copy()
                    if col_data:
                        tabela_original[col_data] = pd.to_datetime(
                            tabela_original[col_data], errors="coerce"
                        ).dt.strftime("%m/%Y")

                    configuracao_colunas = {}
                    if col_data:
                        configuracao_colunas[col_data] = st.column_config.TextColumn("Período")
                    for m in selecionadas_original:
                        configuracao_colunas[m] = st.column_config.NumberColumn(
                            rotulo_metrica_frequencia(m)
                        )

                    st.dataframe(
                        tabela_original,
                        width="stretch",
                        hide_index=True,
                        height=650,
                        column_config=configuracao_colunas,
                    )
                    st.caption(
                        "Esta é a base detalhada usada pelo dashboard. A identificação no cabeçalho mostra a periodicidade de origem de cada indicador."
                    )
            else:
                render_grade_graficos(
                    selecionadas_original,
                    base_func=base_da_metrica,
                    key_prefix="detalhada",
                )
                st.caption("Cada métrica é exibida em um gráfico próprio e na sua frequência original.")

    with aba_trimestral:
        st.caption(
            "Base trimestral preparada para as análises econométricas e preditivas. "
            "Cada modelo pode aplicar filtros adicionais conforme as variáveis utilizadas. "
            "Por isso, todas as séries desta aba são apresentadas em frequência trimestral."
        )

        if trimestral.empty:
            st.warning("Base trimestral não disponível.")
        else:
            col_data_tri = detectar_coluna_data(trimestral)
            # A base trimestral também exclui campos técnicos e colunas
            # completamente vazias do seletor de indicadores.
            metricas_tri = metricas_validas(trimestral)
            grupos_tri = sorted({grupo_metrica(m) for m in metricas_tri})

            c1, c2 = st.columns([1.2, 1], vertical_alignment="bottom")
            with c1:
                grupo_tri = st.selectbox(
                    "Grupo", ["Todos"] + grupos_tri, key="grupo_base_trimestral"
                )
            with c2:
                visual_tri = st.radio(
                    "Visualização", ["Tabela", "Gráficos"], horizontal=True,
                    key="visual_base_trimestral"
                )

            if grupo_tri != "Todos":
                metricas_tri = [m for m in metricas_tri if grupo_metrica(m) == grupo_tri]

            selecionadas_tri = st.multiselect(
                "Indicadores trimestrais",
                metricas_tri,
                default=metricas_tri[:10],
                format_func=lambda m: f"{nome_metrica(m)} · Trimestral",
                key="indicadores_base_trimestral",
            )

            st.markdown(
                '<div class="legend-row">'
                '<div class="legend-item"><span class="legend-dot" style="background:#6385ff"></span> Trimestral — base de modelagem</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if visual_tri == "Tabela":
                cols_tri = ([col_data_tri] if col_data_tri else []) + selecionadas_tri
                if not selecionadas_tri:
                    st.info("Selecione pelo menos um indicador.")
                else:
                    tabela_tri = trimestral[cols_tri].copy()
                    if col_data_tri:
                        datas_tri = pd.to_datetime(tabela_tri[col_data_tri], errors="coerce")
                        tabela_tri[col_data_tri] = datas_tri.dt.to_period("Q").astype(str)

                    config_tri = {}
                    if col_data_tri:
                        config_tri[col_data_tri] = st.column_config.TextColumn("Trimestre")
                    for m in selecionadas_tri:
                        config_tri[m] = st.column_config.NumberColumn(
                            f"{nome_metrica(m)} · Trimestral"
                        )

                    st.dataframe(
                        tabela_tri,
                        width="stretch",
                        hide_index=True,
                        height=650,
                        column_config=config_tri,
                    )
                    st.caption(
                        "Uma linha por trimestre. Esta é a base trimestral de modelagem; cada modelo pode aplicar filtros adicionais conforme as variáveis utilizadas."
                    )
            else:
                render_grade_graficos(
                    selecionadas_tri,
                    base_func=lambda _: trimestral,
                    frequencia_forcada="trimestral",
                    key_prefix="trimestral_modelagem",
                )
                st.caption("Cada métrica da Base Trimestral de Modelagem é exibida em um gráfico trimestral próprio. A amostra final pode variar entre os modelos conforme seus filtros e variáveis.")


# ============================================================
# 3. HISTÓRICO
# ============================================================

elif pagina == "Determinantes do Mercado":
    st.subheader("Determinantes do mercado de cartão de crédito")
    st.caption("Três dimensões complementares: estrutura competitiva, condições financeiras das famílias e contexto macroeconômico.")
    aba_competicao, aba_familias, aba_macro = st.tabs(["🏦 Competição", "👨‍👩‍👧 Famílias", "🌎 Macroeconomia"])
    with aba_competicao:
        st.subheader("Estrutura competitiva")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_kpi("HHI", "hhi_credito_0_1", "indice", "absoluto")
        with c2:
            render_kpi("CR5", "cr5_credito", "percentual_fracao", "pp_fracao")
        with c3:
            render_kpi("Share fintechs", "share_fintechs", "percentual_fracao", "pp_fracao")

        c1, c2 = st.columns(2)
        with c1:
            grafico_serie(trimestral, "hhi_credito_0_1", "HHI do mercado de crédito", "Índice")
        with c2:
            grafico_serie(trimestral, "cr5_credito", "CR5 do mercado de crédito", "Índice")

        shares = [
            "share_bancos_tradicionais",
            "share_bancos_medios",
            "share_fintechs",
            "share_cooperativas",
            "share_instituicoes_pagamento",
            "share_financeiras",
        ]

        # Série complementar exclusiva para visualização.
        # Não modifica as variáveis utilizadas nos modelos.
        arquivo_competicao_visual = (
            "outputs_competicao_auditada/"
            "shares_institucionais_dashboard.csv"
        )

        base_competicao_visual = trimestral
        data_competicao_visual = COL_DATA_TRIM

        if Path(arquivo_competicao_visual).exists():
            base_competicao_visual = pd.read_csv(
                arquivo_competicao_visual
            )
            base_competicao_visual["date"] = pd.to_datetime(
                base_competicao_visual["date"]
            )
            data_competicao_visual = "date"

        shares = [
            s for s in shares
            if s in base_competicao_visual.columns
        ]

        if shares and data_competicao_visual:
            aux = base_competicao_visual[
                [data_competicao_visual] + shares
            ].copy()
            for c in shares:
                aux[c] = pd.to_numeric(aux[c], errors="coerce") * 100
            aux = aux.melt(id_vars=data_competicao_visual, var_name="Instituição", value_name="Participação")
            aux["Instituição"] = aux["Instituição"].map({c: nome_metrica(c) for c in shares})
            fig = px.line(aux, x=data_competicao_visual, y="Participação", color="Instituição", title="Evolução trimestral da composição institucional")
            fig = adicionar_periodos_pandemia(fig)
            fig.update_layout(height=510, hovermode="x unified", xaxis_title="", yaxis_title="Participação (%)")
            st.plotly_chart(fig, width="stretch")

            st.caption(
                "Série institucional com 44 trimestres. "
                "Os períodos 2023Q1 e 2025Q4 foram recuperados "
                "da auditoria complementar do IFData. "
                "A série combina metodologias com diferenças identificadas; "
                "seu uso é exclusivamente descritivo, "
                "sem substituir os dados dos modelos econométricos. "
                "Os indicadores referem-se ao crédito total, "
                "não exclusivamente a cartões."
            )

    with aba_familias:
        st.subheader("Condições financeiras das famílias")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_kpi("Inadimplência", "inadimplencia_90d_pct", "percentual", "pp")
        with c2:
            render_kpi("Comprometimento de renda", "comprometimento_renda_pct", "percentual", "pp")
        with c3:
            render_kpi("Endividamento", "endividamento_familias_pct_renda", "percentual", "pp")

        c1, c2 = st.columns(2)
        with c1:
            grafico_serie(mensal, "inadimplencia_90d_pct", "Inadimplência > 90 dias", "%")
        with c2:
            grafico_serie(mensal, "comprometimento_renda_pct", "Comprometimento da renda", "%")
        grafico_serie(mensal, "endividamento_familias_pct_renda", "Endividamento das famílias", "%")

    with aba_macro:
        st.subheader("Ambiente macroeconômico — IBGE")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_kpi("Desocupação", "taxa_desocupacao_trimestral_pct", "percentual", "pp")
        with c2:
            render_kpi("Rendimento médio real", "rendimento_medio_real_brl", "brl", "percentual")
        with c3:
            render_kpi("IPCA mensal", "ipca_mensal_pct", "percentual", "pp")

        c1, c2 = st.columns(2)
        with c1:
            grafico_serie(trimestral, "taxa_desocupacao_trimestral_pct", "Taxa de desocupação", "%")
        with c2:
            grafico_serie(trimestral, "rendimento_medio_real_brl", "Rendimento médio real", "R$")
        grafico_serie(mensal, "ipca_mensal_pct", "IPCA mensal", "%")

        st.caption("As séries trimestrais do IBGE são exibidas na frequência trimestral; IPCA e IPCA-15 permanecem mensais.")

elif pagina == "Correlações":
    st.subheader("Matriz integrada de correlações — BCB, IFData e IBGE")
    st.caption("Séries trimestrais alinhadas por período. O número de observações (N) pode variar por par de indicadores; valores ausentes não são preenchidos artificialmente.")

    base_corr = trimestral.copy()
    if base_corr.empty:
        st.warning("Base trimestral não disponível. Atualize metricas_master_trimestral.csv.")
    else:
        data_corr = detectar_coluna_data(base_corr)
        if data_corr:
            base_corr[data_corr] = pd.to_datetime(base_corr[data_corr], errors="coerce")
            base_corr = base_corr.dropna(subset=[data_corr]).sort_values(data_corr)
            periodos = base_corr[data_corr].dt.to_period("Q")
            inicio_padrao, fim_padrao = periodos.min(), periodos.max()
            if pd.notna(inicio_padrao) and pd.notna(fim_padrao):
                intervalo = st.select_slider(
                    "Período da análise",
                    options=pd.period_range(inicio_padrao, fim_padrao, freq="Q").astype(str).tolist(),
                    value=(str(inicio_padrao), str(fim_padrao)),
                    key="intervalo_correlacoes_v13",
                )
                inicio, fim = pd.Period(intervalo[0], freq="Q"), pd.Period(intervalo[1], freq="Q")
                base_corr = base_corr.loc[periodos.between(inicio, fim)].copy()

        grupos_corr = {
            "BCB · Cartão e famílias": ["juros_cartao_pf_aa_pct", "spread_cartao_pf", "inadimplencia_90d_pct", "saldo_cartao_pf_total_mr", "comprometimento_renda_pct"],
            "IFData · Competição": ["hhi_credito_0_1", "cr5_credito", "share_fintechs", "n_instituicoes_credito"],
            "IBGE · Trabalho e renda": ["taxa_desocupacao_trimestral_pct", "taxa_desocupacao_movel_pct", "rendimento_medio_real_brl", "massa_rendimento_real_mr"],
            "Inflação · IBGE": ["ipca_mensal_pct", "ipca_alimentacao_pct", "ipca_transportes_pct", "ipca_saude_pct", "ipca_educacao_pct", "ipca15_mensal_pct"],
            "BCB · Macroeconomia": ["selic_am_pct"],
        }
        colunas_numericamente_validas = metricas_numericas(base_corr)
        opcoes = list(dict.fromkeys(
            col for cols in grupos_corr.values() for col in cols if col in colunas_numericamente_validas
        ))
        opcoes += [c for c in colunas_numericamente_validas if c not in opcoes]
        padrao = [
            "juros_cartao_pf_aa_pct", "spread_cartao_pf", "inadimplencia_90d_pct",
            "hhi_credito_0_1", "cr5_credito", "share_fintechs", "selic_am_pct",
            "taxa_desocupacao_trimestral_pct", "rendimento_medio_real_brl", "massa_rendimento_real_mr",
        ]
        padrao = [c for c in padrao if c in opcoes]
        st.caption("Selecione livremente indicadores das diferentes fontes. IPCA e IPCA-15 têm cobertura menor nesta base; confira a amostra (N) antes de interpretar.")
        selecionadas = st.multiselect(
            "Indicadores na matriz",
            options=opcoes,
            default=padrao if len(padrao) >= 2 else opcoes[:8],
            format_func=lambda c: f"{nome_metrica(c)}  ·  " + next((g.split(' · ')[0] for g, cols in grupos_corr.items() if c in cols), "Outro"),
            key="selecionadas_corr_integrada_v13",
        )
        if len(selecionadas) >= 2:
            valores = base_corr[selecionadas].apply(pd.to_numeric, errors="coerce")
            matriz_n = valores.notna().astype(int).T.dot(valores.notna().astype(int))
            corr = valores.corr(method="pearson", min_periods=3)
            rotulos = [nome_metrica(x) for x in selecionadas]
            texto = np.empty(corr.shape, dtype=object)
            hover = np.empty(corr.shape, dtype=object)
            for i, linha in enumerate(selecionadas):
                for j, coluna in enumerate(selecionadas):
                    n = int(matriz_n.loc[linha, coluna])
                    r = corr.loc[linha, coluna]
                    texto[i, j] = f"{r:.2f}" if pd.notna(r) else "—"
                    hover[i, j] = f"{nome_metrica(linha)} × {nome_metrica(coluna)}<br>r = {r:.3f}<br>N = {n}" if pd.notna(r) else f"N = {n}<br>Correlação indisponível"
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=rotulos, y=rotulos,
                zmin=-1, zmax=1, zmid=0,
                colorscale="RdBu", reversescale=True,
                text=texto, texttemplate="%{text}", textfont=dict(size=11),
                customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                colorbar=dict(title="Pearson r"),
            ))
            fig.update_layout(
                height=max(630, 47 * len(selecionadas) + 180),
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis=dict(tickangle=-45, automargin=True),
                yaxis=dict(autorange="reversed", automargin=True),
            )
            st.plotly_chart(fig, width="stretch")
            with st.expander("Ver amostra (N) por par de indicadores"):
                st.dataframe(matriz_n.rename(index=nome_metrica, columns=nome_metrica), width="stretch")
            with st.expander("Cobertura e fontes dos indicadores"):
                cobertura = pd.DataFrame({
                    "Indicador": [nome_metrica(c) for c in selecionadas],
                    "Fonte/grupo": [next((g for g, cols in grupos_corr.items() if c in cols), "Outro") for c in selecionadas],
                    "Observações válidas": [int(valores[c].notna().sum()) for c in selecionadas],
                })
                st.dataframe(cobertura, hide_index=True, width="stretch")
            st.caption("Correlação de Pearson calculada por pares completos (pairwise), sem preenchimento de ausências. Uma correlação alta não comprova causalidade e pode refletir tendências comuns.")
        else:
            st.info("Selecione pelo menos dois indicadores para gerar a matriz.")

        st.markdown("#### Relação entre duas métricas")
        if len(opcoes) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                xvar = st.selectbox("Eixo X", opcoes, index=0, format_func=nome_metrica)
            with c2:
                yvar = st.selectbox("Eixo Y", opcoes, index=1, format_func=nome_metrica)
            cols = ([data_corr] if data_corr else []) + list(dict.fromkeys([xvar, yvar]))
            aux = base_corr[cols].copy()
            aux[xvar] = pd.to_numeric(aux[xvar], errors="coerce")
            aux[yvar] = pd.to_numeric(aux[yvar], errors="coerce")
            aux = aux.dropna(subset=[xvar, yvar])
            if len(aux) >= 3 and xvar != yvar:
                if data_corr:
                    aux["Período"] = aux[data_corr].dt.to_period("Q").astype(str)
                fig = px.scatter(aux, x=xvar, y=yvar,
                    hover_name="Período" if "Período" in aux.columns else None,
                    labels={xvar: nome_metrica(xvar), yvar: nome_metrica(yvar)})
                fig.update_layout(height=520)
                st.plotly_chart(fig, width="stretch")
                r = aux[[xvar, yvar]].corr().iloc[0, 1]
                st.metric("Correlação de Pearson", f"{r:.3f}")
                st.caption(f"Observações pareadas: {len(aux)}")
            else:
                st.info("Escolha duas métricas diferentes com pelo menos três observações simultâneas.")

    st.caption("Esta matriz é exploratória. A amostra da estimação do Modelo 3C pode ser diferente da base trimestral exibida.")


# ============================================================
# 8. PRÉ / PANDEMIA / PÓS
# ============================================================

elif pagina == "Previsão Markov-Neural":
    st.subheader("Previsão Markov-Neural")
    st.markdown("### Identificação histórica dos regimes — Markov-Switching")
    st.caption("A visualização abaixo mostra a série de juros usada na análise; a classificação dos regimes depende dos resultados estimados pelo modelo.")
    if "juros_cartao_pf_aa_pct" in trimestral.columns:
        grafico_serie(trimestral, "juros_cartao_pf_aa_pct", "Evolução trimestral dos juros do cartão PF", "%")
    st.markdown("### Monitoramento preditivo — Previsto × Realizado")

    if not previsoes.empty:
        nomes_previsao = {
            "juros_cartao_pf_aa_pct": ("Juros do cartão PF", "% a.a."),
            "inadimplencia_90d_pct": ("Inadimplência acima de 90 dias", "%"),
            "hhi_credito_0_1": ("HHI — crédito institucional", ""),
            "cr5_credito": ("CR5 — crédito institucional", ""),
        }

        historico_painel = previsoes.copy()
        historico_painel["valor_previsto"] = pd.to_numeric(
            historico_painel["valor_previsto"], errors="coerce"
        )
        historico_painel["valor_realizado"] = pd.to_numeric(
            historico_painel["valor_realizado"], errors="coerce"
        )

        periodos = sorted(
            historico_painel["periodo_previsto"].dropna().astype(str).unique(),
            reverse=True
        )
        periodo_escolhido = st.selectbox(
            "Trimestre previsto", periodos, key="periodo_monitoramento"
        )

        painel_periodo = historico_painel.loc[
            historico_painel["periodo_previsto"].astype(str) == periodo_escolhido
        ]

        colunas = st.columns(2)

        for i, (_, linha) in enumerate(painel_periodo.iterrows()):
            metrica = str(linha["metrica"])
            nome, unidade = nomes_previsao.get(metrica, (metrica, ""))

            previsto = linha["valor_previsto"]
            realizado = linha["valor_realizado"]

            def formatar(valor):
                if pd.isna(valor):
                    return "Aguardando publicação"
                if metrica == "hhi_credito_0_1":
                    return f"{valor:.6f}"
                if metrica == "cr5_credito":
                    return f"{valor * 100:.2f}%"
                return f"{valor:.2f}{(' ' + unidade) if unidade else ''}"

            with colunas[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nome} — {periodo_escolhido}**")
                    st.metric("Previsto", formatar(previsto))
                    st.markdown(f"**Realizado:** {formatar(realizado)}")
                    if pd.notna(realizado) and pd.notna(previsto):
                        st.caption(f"Erro absoluto: {abs(realizado - previsto):.4f}")
                    st.caption(f"Modelo: {linha['modelo']}")

        st.markdown("#### Histórico de previsões")
        st.dataframe(historico_painel, width="stretch", hide_index=True)
        st.caption(
            "Previsões históricas recuperadas são identificadas no campo "
            "'tipo_previsao'. A data de modificação do arquivo não comprova "
            "isoladamente a data de emissão da previsão."
        )
    else:
        st.info("Nenhuma previsão registrada no histórico até o momento.")


# ============================================================
# 11. RESULTADOS DA PESQUISA
# ============================================================

elif pagina == "Validação Experimental v17":
    st.subheader("Validação experimental — Markov-Neural")
    st.caption("Avaliação retrospectiva walk-forward: 2022Q4–2026Q2, 15 previsões. Resultados de pesquisa, não projeções futuras nem evidência causal.")

    raiz = PROJECT_ROOT
    dir_variacao = raiz / "outputs_modelo4_variacao_juros"
    dir_boot = raiz / "outputs_modelo4_validacao_estatistica"
    dir_pib = raiz / "outputs_modelo4_pib"
    dir_robustez = raiz / "outputs_modelo3D_robustez"
    dir_politico = raiz / "outputs_modelo3D_politico_institucional" / "comparacao_modelos"

    previsoes_v16 = ler_csv(dir_variacao / "previsoes_walk_forward.csv")
    metricas_v16 = ler_csv(dir_variacao / "metricas.csv")
    bootstrap_v16 = ler_csv(dir_boot / "bootstrap_erros.csv")
    macro_v16 = ler_csv(dir_pib / "base_modelo4_pib_validacao.csv")
    robustez_v16 = ler_csv(dir_robustez / "resumo_robustez.csv")
    politico_v16 = ler_csv(dir_politico / "comparacao_3C_3D.csv")

    tab_prev, tab_erros, tab_macro, tab_regimes = st.tabs([
        "📈 Previsões", "📐 Erros e incerteza", "🌎 PIB e eleições", "🧪 Robustez Markov"
    ])

    with tab_prev:
        if previsoes_v16.empty:
            st.warning(f"Arquivo não encontrado: {dir_variacao / 'previsoes_walk_forward.csv'}")
        else:
            nomes = {
                "persistencia": "Persistência",
                "mlp_nivel": "MLP — nível",
                "hibrido_nivel": "Markov-Neural — nível",
                "mlp_variacao": "MLP — variação",
                "hibrido_variacao": "Markov-Neural — variação",
            }
            disponiveis = [x for x in nomes if x in previsoes_v16.columns]
            escolhas = st.multiselect(
                "Modelos comparados", disponiveis,
                default=[x for x in ["persistencia", "mlp_variacao", "hibrido_variacao"] if x in disponiveis],
                format_func=lambda x: nomes[x], key="v16_series"
            )
            if "alvo" in previsoes_v16 and "realizado" in previsoes_v16:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=previsoes_v16["alvo"].astype(str), y=pd.to_numeric(previsoes_v16["realizado"], errors="coerce"), name="Realizado", mode="lines+markers", line=dict(width=3)))
                for col in escolhas:
                    fig.add_trace(go.Scatter(x=previsoes_v16["alvo"].astype(str), y=pd.to_numeric(previsoes_v16[col], errors="coerce"), name=nomes[col], mode="lines+markers"))
                fig.update_layout(title="Juros do cartão PF — realizado × previsto", xaxis_title="Trimestre previsto", yaxis_title="Juros (% a.a.)", height=450, template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(previsoes_v16, use_container_width=True, hide_index=True)
            st.download_button("Baixar previsões", data=previsoes_v16.to_csv(index=False).encode("utf-8-sig"), file_name="previsoes_modelo4_v16.csv", mime="text/csv", key="v16_dl_prev")

    with tab_erros:
        if metricas_v16.empty:
            st.warning(f"Arquivo não encontrado: {dir_variacao / 'metricas.csv'}")
        else:
            st.markdown("#### Desempenho preditivo fora da amostra")
            st.dataframe(metricas_v16, use_container_width=True, hide_index=True)
            if {"modelo", "MAE", "RMSE"}.issubset(metricas_v16.columns):
                for metrica in ["MAE", "RMSE"]:
                    dados = metricas_v16.copy()
                    dados[metrica] = pd.to_numeric(dados[metrica], errors="coerce")
                    fig = px.bar(dados.sort_values(metrica), x="modelo", y=metrica, text_auto=".3f", title=f"{metrica} por modelo (menor é melhor)")
                    fig.update_layout(template="plotly_dark", height=340, xaxis_title="", yaxis_title=metrica)
                    st.plotly_chart(fig, use_container_width=True)
        if not previsoes_v16.empty and {"alvo", "realizado", "persistencia", "hibrido_variacao"}.issubset(previsoes_v16.columns):
            df_erros = previsoes_v16[["alvo", "realizado", "persistencia", "hibrido_variacao"]].copy()
            for col in ["realizado", "persistencia", "hibrido_variacao"]:
                df_erros[col] = pd.to_numeric(df_erros[col], errors="coerce")
            df_erros["ganho_pp"] = (df_erros["realizado"] - df_erros["persistencia"]).abs() - (df_erros["realizado"] - df_erros["hibrido_variacao"]).abs()
            fig = px.bar(df_erros, x="alvo", y="ganho_pp", title="Ganho no erro absoluto: persistência menos híbrido", labels={"ganho_pp": "Ganho (p.p.)", "alvo": "Trimestre"})
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Valores positivos favorecem o híbrido; valores negativos favorecem a persistência.")
        st.markdown("#### Bootstrap em blocos: diferença média de erros absolutos")
        if bootstrap_v16.empty:
            st.info(f"Resultado do bootstrap ainda não encontrado: {dir_boot / 'bootstrap_erros.csv'}")
        else:
            st.dataframe(bootstrap_v16, use_container_width=True, hide_index=True)
            st.info("Os intervalos de 95% obtidos incluem zero. A superioridade estatística do híbrido não foi demonstrada; com apenas 15 trimestres, o bootstrap é exploratório.")
        st.caption("As várias especificações foram avaliadas na mesma janela de teste. Os resultados podem refletir seleção de modelo e não devem ser interpretados como validação externa independente.")

    with tab_macro:
        if macro_v16.empty:
            st.warning(f"Arquivo não encontrado: {dir_pib / 'base_modelo4_pib_validacao.csv'}")
        else:
            macro_v16 = macro_v16.copy()
            if "quarter" in macro_v16.columns:
                macro_v16["quarter"] = macro_v16["quarter"].astype(str)
                campos = [c for c in ["pib_crescimento_tri_pct", "pib_disponivel_origem", "iiebr_lag1"] if c in macro_v16.columns]
                for campo in campos:
                    macro_v16[campo] = pd.to_numeric(macro_v16[campo], errors="coerce")
                if "pib_crescimento_tri_pct" in campos:
                    fig = px.line(macro_v16, x="quarter", y="pib_crescimento_tri_pct", markers=True, title="Crescimento trimestral real do PIB (IBGE)")
                    if "periodo_eleitoral" in macro_v16.columns:
                        for periodo in macro_v16.loc[pd.to_numeric(macro_v16["periodo_eleitoral"], errors="coerce") == 1, "quarter"]:
                            fig.add_vrect(x0=str(periodo), x1=str(periodo), line_width=2, line_color="orange")
                    fig.update_layout(template="plotly_dark", height=380, xaxis_title="Trimestre", yaxis_title="Variação (%)")
                    st.plotly_chart(fig, use_container_width=True)
                if "iiebr_lag1" in campos:
                    fig = px.line(macro_v16, x="quarter", y="iiebr_lag1", title="Incerteza econômica FGV — IIE-Br defasado")
                    fig.update_layout(template="plotly_dark", height=350, xaxis_title="Trimestre", yaxis_title="Índice")
                    st.plotly_chart(fig, use_container_width=True)
                if "periodo_eleitoral" in macro_v16.columns:
                    periodos_eleitorais = macro_v16.loc[pd.to_numeric(macro_v16["periodo_eleitoral"], errors="coerce") == 1, "quarter"].tolist()
                    st.caption("Trimestres classificados como eleitorais presidenciais (Q3/Q4): " + (", ".join(periodos_eleitorais) if periodos_eleitorais else "nenhum na base"))
                st.dataframe(macro_v16[["quarter"] + campos + (["periodo_eleitoral"] if "periodo_eleitoral" in macro_v16.columns else [])].tail(20), use_container_width=True, hide_index=True)
        st.caption("PIB e IIE-Br são séries históricas sujeitas a revisões e calendários de publicação. A variável eleitoral marca períodos, não mede efeitos causais.")

    with tab_regimes:
        st.markdown("#### Estabilidade de estimação — modelos com variáveis político-institucionais")
        if robustez_v16.empty:
            st.info(f"Arquivo não encontrado: {dir_robustez / 'resumo_robustez.csv'}")
        else:
            st.dataframe(robustez_v16, use_container_width=True, hide_index=True)
        st.markdown("#### Primeira comparação 3C × 3D")
        if politico_v16.empty:
            st.info(f"Arquivo não encontrado: {dir_politico / 'comparacao_3C_3D.csv'}")
        else:
            st.dataframe(politico_v16, use_container_width=True, hide_index=True)
        st.warning("AIC/BIC avaliam ajuste dentro da amostra, não acurácia preditiva. Houve avisos de probabilidades de transição durante estimações EM; a ocupação de alguns regimes é pequena. Não interpretar associação eleitoral como causalidade.")


elif pagina == "Questões de Pesquisa":
    st.subheader("Questões de Pesquisa")
    st.caption("Perguntas orientadoras da dissertação e evidências observáveis. Resultados conclusivos dependem da validação dos modelos.")
    with st.container(border=True):
        st.markdown("**Q1. Quais regimes competitivos existem no mercado brasileiro de cartão de crédito?**")
        st.write("Evidências: identificação dos regimes pelo Markov-Switching, probabilidades, perfis e matriz de transição, quando disponíveis.")
    with st.container(border=True):
        st.markdown("**Q2. Quais indicadores estão associados às mudanças entre os regimes?**")
        st.write("Evidências: estrutura competitiva, condições das famílias, contexto macroeconômico e análises estatísticas. Associação não implica causalidade.")
    with st.container(border=True):
        st.markdown("**Q3. O modelo híbrido Markov-Neural prevê os regimes melhor que os modelos de referência?**")
        st.write("Evidências: validação temporal e comparação com persistência, regressão logística e MLP. Não há conclusão de superioridade sem métricas comparáveis.")
    st.markdown("### Evidências e modelos da pesquisa")
    aba1, aba2, aba3, aba4 = st.tabs(["Modelo 1", "Modelo 2", "Modelo 3", "Modelo 4"])

    with aba1:
        st.markdown("### Modelo 1 — Dinâmica competitiva")
        st.caption(
            "Evolução da concentração do crédito e participação das fintechs. "
            "As regressões descrevem associações estatísticas, não efeitos causais."
        )

        c1, c2 = st.columns(2)
        with c1:
            grafico_serie(
                trimestral,
                "hhi_credito_0_1",
                "HHI do crédito",
            )
        with c2:
            grafico_serie(
                trimestral,
                "share_fintechs",
                "Participação das fintechs",
                "%",
                percentual_fracao=True,
            )

        st.markdown("#### Resultados econométricos")

        pasta_coeficientes = (
            PROJECT_ROOT / "DataSet"
        )

        especificacoes = {
            "Modelo 1A — Concentração e competição":
                "coeficientes_modelo_01_competicao.csv",
            "Modelo 1B — Variação da concentração":
                "coeficientes_modelo_01B_variacao_hhi.csv",
            "Modelo 1C — Dinâmica com defasagem":
                "coeficientes_modelo_01C_revisado.csv",
        }

        for titulo_modelo, nome_arquivo in especificacoes.items():
            st.markdown(f"##### {titulo_modelo}")
            caminho = pasta_coeficientes / nome_arquivo

            if not caminho.exists():
                st.warning(f"Arquivo não encontrado: {nome_arquivo}")
                continue

            resultados = pd.read_csv(caminho)

            colunas_necessarias = {
                "variavel", "coeficiente", "erro_padrao_hac",
                "estatistica_t", "p_valor",
                "ic_95_inferior", "ic_95_superior",
            }

            if not colunas_necessarias.issubset(resultados.columns):
                st.error(
                    f"Colunas estatísticas incompletas: {nome_arquivo}"
                )
                continue

            tabela = resultados.copy()
            tabela["significativo_5pct"] = (
                tabela["p_valor"] < 0.05
            ).map({True: "Sim", False: "Não"})

            tabela = tabela.rename(columns={
                "variavel": "Variável",
                "coeficiente": "Coeficiente",
                "erro_padrao_hac": "Erro-padrão HAC",
                "estatistica_t": "Estatística t",
                "p_valor": "Valor-p",
                "ic_95_inferior": "IC 95% inferior",
                "ic_95_superior": "IC 95% superior",
                "significativo_5pct": "Significativo (5%)",
            })

            colunas_numericas = tabela.select_dtypes(
                include="number"
            ).columns

            tabela[colunas_numericas] = (
                tabela[colunas_numericas].round(6)
            )

            st.dataframe(
                tabela,
                hide_index=True,
                width="stretch",
            )

            st.caption(
                "Erros-padrão HAC conforme arquivo original. "
                "Significância avaliada a 5%."
            )

        st.markdown("#### Interpretação dos resultados")
        st.info(
            "Nos Modelos 1A, 1B e 1C, a participação das fintechs "
            "apresenta associação negativa e estatisticamente "
            "significativa com a concentração do crédito. "
            "Esse resultado não estabelece causalidade."
        )

        st.warning(
            "O HHI analisado refere-se ao mercado de crédito "
            "das instituições consideradas, e não exclusivamente "
            "ao cartão de crédito. A classificação das fintechs "
            "também possui limitações documentadas na auditoria."
        )

        st.caption(
            "Os coeficientes foram recuperados dos arquivos existentes. "
            "R², tamanho da amostra e diagnósticos adicionais "
            "ainda precisam ser integrados a partir das saídas "
            "das estimações correspondentes."
        )

    with aba2:
        st.markdown("### Modelo 2 — Determinantes do spread")

        st.caption(
            "Regressões OLS com erros-padrão robustos HAC/Newey-West. "
            "Os resultados representam associações estatísticas, "
            "não necessariamente efeitos causais."
        )

        grafico_serie(
            mensal,
            "spread_cartao_pf",
            "Evolução do spread do cartão PF",
            "p.p.",
        )

        pasta_modelo2 = (
            PROJECT_ROOT
            / "outputs_modelo2"
        )

        especificacoes_2 = [
            "modelo2A",
            "modelo2B",
            "modelo2C",
            "modelo2D",
        ]

        import json

        resumos_2 = []
        disponiveis_2 = []

        for nome_modelo in especificacoes_2:
            pasta_especificacao = pasta_modelo2 / nome_modelo
            arquivo_resumo = pasta_especificacao / "resumo.json"
            arquivo_coef = pasta_especificacao / "coeficientes.csv"

            if not (arquivo_resumo.exists() and arquivo_coef.exists()):
                st.warning(
                    f"Resultados não encontrados para {nome_modelo}."
                )
                continue

            with open(arquivo_resumo, encoding="utf-8") as f:
                resumo = json.load(f)

            resumos_2.append({
                "Modelo": nome_modelo,
                "N": resumo.get("n_observacoes"),
                "Início": resumo.get("data_inicial"),
                "Fim": resumo.get("data_final"),
                "R²": resumo.get("r2"),
                "R² ajustado": resumo.get("r2_ajustado"),
                "AIC": resumo.get("aic"),
                "BIC": resumo.get("bic"),
                "Variável dependente": resumo.get(
                    "variavel_dependente"
                ),
            })

            disponiveis_2.append(nome_modelo)


        # =====================================================
        # MODELOS 2 — RESULTADOS OFICIAIS V1
        # =====================================================

        st.markdown("### Resultados oficiais — Modelos 2B, 2C e 2D")

        st.caption(
            "Base oficial V1 | Amostra trimestral comum: "
            "2015Q3–2025Q4 | 42 observações | "
            "Regressões OLS com erros-padrão HAC/Newey-West. "
            "Resultados associativos, sem interpretação causal."
        )

        pasta_oficial_2 = (
            Path(__file__).resolve().parent
            / "outputs_modelos2_oficial_v1"
        )

        resumo_oficial_2 = pasta_oficial_2 / "resumo_modelos.csv"
        coef_oficial_2 = pasta_oficial_2 / "coeficientes_hac.csv"
        ajustes_oficial_2 = pasta_oficial_2 / "ajustes_em_amostra.csv"

        if all(
            p.exists()
            for p in [
                resumo_oficial_2,
                coef_oficial_2,
                ajustes_oficial_2,
            ]
        ):
            resumo_2_oficial = pd.read_csv(resumo_oficial_2)
            coef_2_oficial = pd.read_csv(coef_oficial_2)
            ajustes_2_oficial = pd.read_csv(ajustes_oficial_2)

            st.markdown("#### Comparação oficial das especificações")

            tabela_resumo_2 = resumo_2_oficial.rename(columns={
                "modelo": "Modelo",
                "n": "N",
                "inicio": "Início",
                "fim": "Fim",
                "r2": "R²",
                "r2_ajustado": "R² ajustado",
                "aic": "AIC",
                "bic": "BIC",
                "durbin_watson": "Durbin-Watson",
            })

            st.dataframe(
                tabela_resumo_2.round(4),
                hide_index=True,
                use_container_width=True,
            )

            selecionado_oficial_2 = st.selectbox(
                "Especificação oficial",
                resumo_2_oficial["modelo"].tolist(),
                key="modelo2_oficial_v1_selecao",
            )

            st.markdown("#### Coeficientes robustos HAC")

            coef_selecionados_2 = coef_2_oficial[
                coef_2_oficial["modelo"]
                == selecionado_oficial_2
            ].copy()

            coef_selecionados_2["Significativo (5%)"] = (
                coef_selecionados_2["valor_p"] < 0.05
            ).map({True: "Sim", False: "Não"})

            st.dataframe(
                coef_selecionados_2.round(6),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("#### Valores observados × ajustados")

            ajustes_selecionados_2 = ajustes_2_oficial[
                ajustes_2_oficial["modelo"]
                == selecionado_oficial_2
            ].copy()

            ajustes_selecionados_2["date"] = pd.to_datetime(
                ajustes_selecionados_2["date"]
            )

            serie_ajustes_2 = ajustes_selecionados_2[
                ["date", "observado", "ajustado_em_amostra"]
            ].melt(
                id_vars="date",
                var_name="Série",
                value_name="Spread (p.p.)",
            )

            fig_ajustes_2 = px.line(
                serie_ajustes_2,
                x="date",
                y="Spread (p.p.)",
                color="Série",
                title=(
                    f"Modelo {selecionado_oficial_2} — "
                    "ajuste dentro da amostra"
                ),
            )

            st.plotly_chart(
                fig_ajustes_2,
                use_container_width=True,
            )

            st.caption(
                "Os valores ajustados são estimativas dentro "
                "da amostra. Não representam previsões "
                "fora da amostra."
            )

            st.info(
                "O Modelo 2A ainda não foi reestimado na "
                "base oficial V1. Os diagnósticos apresentados "
                "na seção histórica abaixo pertencem às "
                "estimações anteriores e não devem ser "
                "atribuídos aos modelos oficiais."
            )

        else:
            st.warning(
                "Arquivos oficiais dos Modelos 2B–2D "
                "não encontrados."
            )

        st.divider()

        st.markdown("### Resultados históricos — versões anteriores")

        st.caption(
            "Os resultados abaixo foram produzidos em "
            "versões anteriores da base. São mantidos "
            "para rastreabilidade metodológica e não "
            "devem ser confundidos com a estimação oficial V1."
        )

        st.markdown("#### Comparação das especificações")

        if resumos_2:
            comparacao_2 = pd.DataFrame(resumos_2)

            for coluna in ["R²", "R² ajustado", "AIC", "BIC"]:
                comparacao_2[coluna] = pd.to_numeric(
                    comparacao_2[coluna],
                    errors="coerce",
                ).round(4)

            st.dataframe(
                comparacao_2,
                hide_index=True,
                width="stretch",
            )

            st.caption(
                "AIC e BIC não devem ser usados para selecionar "
                "modelos com amostras ou variáveis dependentes "
                "não comparáveis."
            )


        st.markdown("#### Diagnósticos estatísticos — Modelo 2")

        tabelas_diagnosticos = []

        for nome_modelo in disponiveis_2:
            arquivo_diag = (
                pasta_modelo2
                / nome_modelo
                / "diagnosticos.csv"
            )

            if not arquivo_diag.exists():
                st.warning(
                    f"Diagnósticos indisponíveis: {nome_modelo}"
                )
                continue

            diag = pd.read_csv(arquivo_diag)

            colunas_esperadas = {
                "modelo",
                "durbin_watson",
                "breusch_pagan_p",
                "breusch_godfrey_1_p",
                "breusch_godfrey_2_p",
                "jarque_bera_p",
                "vif_maximo",
            }

            if not colunas_esperadas.issubset(diag.columns):
                st.error(
                    f"Diagnósticos incompletos: {nome_modelo}"
                )
                continue

            tabelas_diagnosticos.append(diag)

        if tabelas_diagnosticos:
            diagnosticos_completos = pd.concat(
                tabelas_diagnosticos,
                ignore_index=True,
            )

            diagnosticos_completos = (
                diagnosticos_completos.rename(columns={
                    "modelo": "Modelo",
                    "durbin_watson": "Durbin-Watson",
                    "breusch_pagan_p": "Breusch-Pagan (p)",
                    "breusch_godfrey_1_p": "BG 1 defasagem (p)",
                    "breusch_godfrey_2_p": "BG 2 defasagens (p)",
                    "jarque_bera_p": "Jarque-Bera (p)",
                    "vif_maximo": "VIF máximo",
                })
            )

            tabela_diag = diagnosticos_completos.round(6)

            colunas_p = [
                "Breusch-Pagan (p)",
                "BG 1 defasagem (p)",
                "BG 2 defasagens (p)",
                "Jarque-Bera (p)",
            ]

            for coluna in colunas_p:
                tabela_diag[coluna] = diagnosticos_completos[coluna].map(
                    lambda valor: (
                        "< 0,0001"
                        if pd.notna(valor) and valor < 0.0001
                        else f"{valor:.4f}".replace(".", ",")
                    )
                )

            st.dataframe(
                tabela_diag,
                hide_index=True,
                width="stretch",
            )

            st.caption(
                "Valores-p inferiores a 0,05 indicam rejeição "
                "da hipótese nula do respectivo teste a 5%. "
                "Breusch-Pagan: homocedasticidade; "
                "Breusch-Godfrey: ausência de autocorrelação; "
                "Jarque-Bera: normalidade dos resíduos. "
                "VIF elevado sugere multicolinearidade."
            )

            st.warning(
                "Diagnósticos referentes às estimações "
                "históricas. Não extrapolar estes resultados "
                "para a base oficial V1. Erros-padrão HAC "
                "não corrigem problemas de especificação."
            )

        st.markdown("#### Coeficientes e significância")

        if disponiveis_2:
            selecionado_2 = st.selectbox(
                "Selecione a especificação",
                disponiveis_2,
                key="modelo2_especificacao",
            )

            pasta_selecionada = pasta_modelo2 / selecionado_2

            coeficientes_2 = pd.read_csv(
                pasta_selecionada / "coeficientes.csv"
            )

            coeficientes_2["Significativo (5%)"] = (
                coeficientes_2["p_valor"] < 0.05
            ).map({True: "Sim", False: "Não"})

            tabela_coef_2 = coeficientes_2.rename(columns={
                "variavel": "Variável",
                "coeficiente": "Coeficiente",
                "erro_padrao_hac": "Erro-padrão HAC",
                "estatistica_t_ou_z": "Estatística t/z",
                "p_valor": "Valor-p",
                "ic_95_inferior": "IC 95% inferior",
                "ic_95_superior": "IC 95% superior",
            })

            tabela_coef_2 = tabela_coef_2.round(6)

            tabela_coef_2["Valor-p"] = (
                coeficientes_2["p_valor"].map(
                    lambda valor: (
                        "< 0,0001"
                        if pd.notna(valor) and valor < 0.0001
                        else f"{valor:.4f}".replace(".", ",")
                    )
                )
            )

            st.dataframe(
                tabela_coef_2,
                hide_index=True,
                width="stretch",
            )

            arquivo_texto = (
                pasta_selecionada / "resumo_estatistico.txt"
            )

            if arquivo_texto.exists():
                with st.expander(
                    "Consultar relatório estatístico completo"
                ):
                    st.code(
                        arquivo_texto.read_text(encoding="utf-8"),
                        language=None,
                    )

            with open(
                pasta_selecionada / "resumo.json",
                encoding="utf-8",
            ) as f:
                detalhes_2 = json.load(f)

            st.caption(
                f"Amostra: {detalhes_2.get('data_inicial')} "
                f"a {detalhes_2.get('data_final')} | "
                f"Observações: {detalhes_2.get('n_observacoes')} | "
                f"Covariância: {detalhes_2.get('cov_type')}"
            )

        st.warning(
            "Os diagnósticos de autocorrelação, "
            "heterocedasticidade e multicolinearidade "
            "precisam ser considerados na interpretação. "
            "Erros-padrão HAC não eliminam problemas "
            "de especificação dos modelos."
        )

    with aba3:
        st.markdown("### Modelo 3 — Regimes Markov-Switching")

        st.caption(
            "Resultados da versão auditada do Modelo 3C. "
            "Os regimes são estimados a partir dos juros do cartão PF. "
            "Sua interpretação competitiva depende da análise dos "
            "demais indicadores econômicos."
        )

        pasta_markov = (
            PROJECT_ROOT
            / "outputs_modelo3C_auditado"
        )

        arquivos_markov = {
            "base": "modelo3C_base_regimes_completa.csv",
            "perfil": "modelo3C_perfil_regimes_media.csv",
            "transicao": "modelo3C_matriz_transicao.csv",
            "duracao": "modelo3C_duracao_regimes.csv",
            "mudancas": "modelo3C_mudancas_regime.csv",
        }

        faltantes = [
            nome for nome in arquivos_markov.values()
            if not (pasta_markov / nome).exists()
        ]

        if faltantes:
            st.error(
                "Arquivos auditados não encontrados: "
                + ", ".join(faltantes)
            )
        else:
            dados_regimes = pd.read_csv(
                pasta_markov / arquivos_markov["base"]
            )

            dados_regimes["date"] = pd.to_datetime(
                dados_regimes["date"]
            )

            dados_regimes = dados_regimes.sort_values("date")

            nomes_regimes = {
                0: "Regime 0 — Intermediário",
                1: "Regime 1 — Elevado",
                2: "Regime 2 — Baixo",
            }

            cores_regimes = {
                0: "#E5A700",
                1: "#D64545",
                2: "#258B57",
            }

            st.markdown("#### Linha do tempo dos regimes")

            mostrar_governos = st.checkbox(
                "Exibir períodos dos governos presidenciais",
                value=False,
                key="markov_governos",
            )

            mostrar_eleicoes = st.checkbox(
                "Exibir eleições presidenciais",
                value=False,
                key="markov_eleicoes",
            )

            fig_regimes = px.line(
                dados_regimes,
                x="date",
                y="juros_cartao_pf_aa_pct",
                title="Juros do cartão PF e regimes estimados",
                labels={
                    "date": "Trimestre",
                    "juros_cartao_pf_aa_pct": "Juros (% a.a.)",
                },
            )

            fig_regimes.update_traces(
                line=dict(color="#727A87", width=1.5),
                name="Juros observados",
                showlegend=True,
            )

            if mostrar_governos:
                governos = [
                    ("Dilma Rousseff", "2012-07-01", "2016-05-12"),
                    ("Michel Temer", "2016-05-12", "2019-01-01"),
                    ("Jair Bolsonaro", "2019-01-01", "2023-01-01"),
                    ("Lula", "2023-01-01", "2027-01-01"),
                ]

                for indice, (governo, inicio, fim) in enumerate(governos):
                    fig_regimes.add_vrect(
                        x0=inicio,
                        x1=fim,
                        fillcolor=(
                            "#58769B" if indice % 2 == 0
                            else "#B19A71"
                        ),
                        opacity=0.10,
                        layer="below",
                        line_width=0,
                        annotation_text=governo,
                        annotation_position="top left",
                    )

            if mostrar_eleicoes:
                eleicoes = [
                    ("Eleição 2014", "2014-10-26"),
                    ("Eleição 2018", "2018-10-28"),
                    ("Eleição 2022", "2022-10-30"),
                ]

                for rotulo, data_eleicao in eleicoes:
                    fig_regimes.add_shape(
                        type="line",
                        x0=data_eleicao,
                        x1=data_eleicao,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(
                            color="#B6C5DA",
                            width=2,
                            dash="dot",
                        ),
                    )

                    fig_regimes.add_annotation(
                        x=data_eleicao,
                        y=1.08,
                        xref="x",
                        yref="paper",
                        text=rotulo,
                        showarrow=False,
                        textangle=-45,
                        font=dict(size=11),
                    )

            for regime, nome in nomes_regimes.items():
                grupo = dados_regimes.loc[
                    dados_regimes["regime_estimado"] == regime
                ]

                fig_regimes.add_scatter(
                    x=grupo["date"],
                    y=grupo["juros_cartao_pf_aa_pct"],
                    mode="markers",
                    marker=dict(
                        color=cores_regimes[regime],
                        size=9,
                    ),
                    name=nome,
                    customdata=grupo[["quarter"]],
                    hovertemplate=(
                        "Trimestre: %{customdata[0]}<br>"
                        "Juros: %{y:.2f}% a.a.<extra></extra>"
                    ),
                )

            fig_regimes.update_layout(
                height=510,
                legend_title="Classificação",
            )

            st.plotly_chart(
                fig_regimes,
                use_container_width=True,
            )

            st.caption(
                "Cores: verde = juros baixos; "
                "amarelo = intermediários; vermelho = elevados. "
                "As faixas dos governos são apenas referências "
                "cronológicas, não evidências de causalidade. "
                "A transição Dilma–Temer é indicada pelo início "
                "do exercício interino de Temer, em maio de 2016."
            )

            st.markdown("#### Probabilidades estimadas dos regimes")

            probabilidades = dados_regimes[
                [
                    "date",
                    "prob_regime_0",
                    "prob_regime_1",
                    "prob_regime_2",
                ]
            ].melt(
                id_vars="date",
                var_name="Regime",
                value_name="Probabilidade",
            )

            fig_prob = px.line(
                probabilidades,
                x="date",
                y="Probabilidade",
                color="Regime",
                color_discrete_map={
                    "prob_regime_0": cores_regimes[0],
                    "prob_regime_1": cores_regimes[1],
                    "prob_regime_2": cores_regimes[2],
                },
                labels={"date": "Trimestre"},
                title="Probabilidades suavizadas dos regimes",
            )

            fig_prob.update_layout(height=380)

            st.plotly_chart(
                fig_prob,
                use_container_width=True,
            )

            st.caption(
                "As probabilidades suavizadas incorporam "
                "informações de toda a amostra. "
                "São adequadas à análise histórica, "
                "mas não devem ser tratadas como previsões "
                "produzidas em tempo real."
            )


            st.markdown("#### Perspectivas 2027 — Projeção dos regimes")

            arquivo_projecoes = (
                pasta_markov / "modelo3C_projecao_regimes_2027.csv"
            )

            if arquivo_projecoes.exists():
                df_proj = pd.read_csv(arquivo_projecoes)

                series_proj = df_proj[
                    [
                        "periodo_previsto",
                        "prob_baixo",
                        "prob_intermediario",
                        "prob_elevado",
                    ]
                ].melt(
                    id_vars="periodo_previsto",
                    var_name="Regime",
                    value_name="Probabilidade",
                )

                fig_projecoes = px.line(
                    series_proj,
                    x="periodo_previsto",
                    y="Probabilidade",
                    color="Regime",
                    markers=True,
                    color_discrete_map={
                        "prob_baixo": "#258B57",
                        "prob_intermediario": "#E5A700",
                        "prob_elevado": "#D64545",
                    },
                    title="Probabilidades projetadas até 2027Q4",
                    labels={
                        "periodo_previsto": "Trimestre",
                        "Probabilidade": "Probabilidade",
                    },
                )

                fig_projecoes.update_layout(
                    height=420,
                    yaxis_tickformat=".0%",
                    yaxis_range=[0, 1],
                )

                st.plotly_chart(
                    fig_projecoes,
                    use_container_width=True,
                )

                tabela_proj = df_proj[
                    [
                        "periodo_previsto",
                        "prob_baixo",
                        "prob_intermediario",
                        "prob_elevado",
                        "regime_mais_provavel",
                    ]
                ].copy()

                for coluna in [
                    "prob_baixo",
                    "prob_intermediario",
                    "prob_elevado",
                ]:
                    tabela_proj[coluna] = (
                        tabela_proj[coluna] * 100
                    ).round(2)

                tabela_proj.columns = [
                    "Trimestre",
                    "Baixo (%)",
                    "Intermediário (%)",
                    "Elevado (%)",
                    "Regime predominante",
                ]

                st.dataframe(
                    tabela_proj,
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "Projeção Markov pura, com origem em 2026Q2. "
                    "Calculada pela propagação da matriz de transição "
                    "estimada na versão auditada. Não corresponde "
                    "a uma previsão Markov-Neural nem incorpora "
                    "resultados eleitorais futuros. "
                    "A capacidade preditiva multiperíodo ainda "
                    "precisa ser avaliada fora da amostra."
                )

            st.markdown("#### Perfil econômico dos regimes")

            perfil = pd.read_csv(
                pasta_markov / arquivos_markov["perfil"],
                index_col=0,
            )

            indicadores_perfil = [
                "juros_cartao_pf_aa_pct",
                "selic_am_pct",
                "inadimplencia_90d_pct",
                "hhi_credito_0_1",
                "cr5_credito",
                "share_fintechs",
                "share_bancos_tradicionais",
                "saldo_cartao_pf_total_mr",
            ]

            presentes = [
                x for x in indicadores_perfil
                if x in perfil.index
            ]

            st.dataframe(
                perfil.loc[presentes].round(4),
                use_container_width=True,
            )

            st.caption(
                "HHI e CR5 referem-se à concentração do crédito "
                "das instituições analisadas, não necessariamente "
                "à concentração específica do cartão de crédito. "
                "Participação de fintechs sujeita às limitações "
                "de classificação documentadas na auditoria."
            )

            col_transicao, col_duracao = st.columns(2)

            with col_transicao:
                st.markdown("#### Matriz de transição")

                matriz = pd.read_csv(
                    pasta_markov / arquivos_markov["transicao"],
                    index_col=0,
                )

                st.dataframe(
                    matriz.round(4),
                    use_container_width=True,
                )

                st.caption(
                    "Colunas = regime de origem; "
                    "linhas = regime de destino."
                )

            with col_duracao:
                st.markdown("#### Duração esperada")

                duracao = pd.read_csv(
                    pasta_markov / arquivos_markov["duracao"]
                )

                st.dataframe(
                    duracao.round(4),
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "Duração estimada em trimestres, "
                    "sob as probabilidades de transição do modelo."
                )

            st.markdown("#### Mudanças históricas de regime")

            mudancas = pd.read_csv(
                pasta_markov / arquivos_markov["mudancas"]
            )

            st.dataframe(
                mudancas.round(4),
                hide_index=True,
                use_container_width=True,
            )

    with aba4:
        st.markdown("### Modelo 4 — Avaliação preditiva")
        st.caption(
            "Resultados históricos dos experimentos. "
            "Não confundir validação retrospectiva com previsões "
            "registradas em tempo real."
        )

        raiz_projeto = PROJECT_ROOT

        arquivo_juros = (
            raiz_projeto
            / "outputs_modelo4_previsao_juros_hibrido"
            / "previsoes_walk_forward.csv"
        )

        arquivo_metricas = (
            raiz_projeto
            / "outputs_modelo4_final"
            / "modelo4E_validacao_metricas.csv"
        )

        arquivo_futuro = (
            raiz_projeto
            / "outputs_modelo4_final"
            / "modelo4E_previsao_metricas.csv"
        )

        arquivo_competicao_4e = (
            Path(__file__).resolve().parent
            / "outputs_modelo4E_competicao"
            / "previsoes_walk_forward.csv"
        )

        arquivo_metricas_competicao_4e = (
            Path(__file__).resolve().parent
            / "outputs_modelo4E_competicao"
            / "metricas.csv"
        )

        tab_juros, tab_4e, tab_futuro, tab_competicao = st.tabs([
            "Markov–Neural",
            "Validação das métricas",
            "Previsões do Modelo 4E",
            "Modelo 4E — Competição",
        ])

        with tab_juros:
            st.markdown("#### Previsão trimestral dos juros")

            if arquivo_juros.exists():
                dados_juros = pd.read_csv(arquivo_juros)

                modelos_juros = [
                    "persistencia",
                    "mlp_convencional",
                    "mlp_hibrido",
                ]

                resumo_juros = []

                for modelo in modelos_juros:
                    erro = (
                        dados_juros["realizado"]
                        - dados_juros[modelo]
                    )

                    resumo_juros.append({
                        "Modelo": modelo,
                        "MAE": erro.abs().mean(),
                        "RMSE": (erro.pow(2).mean()) ** 0.5,
                        "Previsões": len(dados_juros),
                    })

                st.dataframe(
                    pd.DataFrame(resumo_juros).round(4),
                    hide_index=True,
                    width="stretch",
                )

                grafico_juros = dados_juros[
                    [
                        "alvo",
                        "realizado",
                        "persistencia",
                        "mlp_convencional",
                        "mlp_hibrido",
                    ]
                ].melt(
                    id_vars="alvo",
                    var_name="Série",
                    value_name="Juros (% a.a.)",
                )

                fig_juros = px.line(
                    grafico_juros,
                    x="alvo",
                    y="Juros (% a.a.)",
                    color="Série",
                    markers=True,
                    title="Juros observados × previsões fora da amostra",
                )

                st.plotly_chart(
                    fig_juros,
                    use_container_width=True,
                )

                with st.expander("Tabela completa das previsões"):
                    st.dataframe(
                        dados_juros.round(4),
                        hide_index=True,
                        width="stretch",
                    )

                st.info(
                    "Neste experimento, a MLP híbrida reduziu "
                    "o erro da MLP convencional, mas não superou "
                    "a persistência. A amostra contém 15 previsões."
                )
            else:
                st.warning(
                    "Arquivo de validação Markov–Neural não encontrado."
                )

        with tab_4e:
            st.markdown("#### Modelo 4E — Validação das métricas")

            if arquivo_metricas.exists():
                dados_4e = pd.read_csv(arquivo_metricas)

                resumo_4e = []

                for metrica, grupo in dados_4e.groupby("metrica"):
                    for modelo in ["persistencia", "ridge", "mlp"]:
                        validos = grupo[
                            ["valor_real_t1", modelo]
                        ].dropna()

                        if validos.empty:
                            continue

                        erro = (
                            validos["valor_real_t1"]
                            - validos[modelo]
                        )

                        resumo_4e.append({
                            "Métrica": metrica,
                            "Modelo": modelo,
                            "MAE": erro.abs().mean(),
                            "RMSE": (erro.pow(2).mean()) ** 0.5,
                            "Observações": len(validos),
                        })

                st.dataframe(
                    pd.DataFrame(resumo_4e).round(6),
                    hide_index=True,
                    width="stretch",
                )

                with st.expander("Validação histórica completa"):
                    st.dataframe(
                        dados_4e.round(6),
                        hide_index=True,
                        width="stretch",
                    )

                st.caption(
                    "Os resultados do Modelo 4E são históricos. "
                    "A disponibilidade temporal das variáveis e "
                    "a seleção dos modelos ainda requerem auditoria."
                )
            else:
                st.warning(
                    "Arquivo de validação do Modelo 4E não encontrado."
                )

        with tab_futuro:
            st.markdown("#### Previsões produzidas pelo Modelo 4E")

            if arquivo_futuro.exists():
                dados_futuros = pd.read_csv(arquivo_futuro)

                colunas_futuras = [
                    "metrica",
                    "periodo_previsto",
                    "periodo_ultimo_dado",
                    "modelo_selecionado",
                    "previsao_selecionada",
                ]

                st.dataframe(
                    dados_futuros[colunas_futuras].round(6),
                    hide_index=True,
                    width="stretch",
                )

                st.caption(
                    "Estas são as previsões armazenadas no "
                    "arquivo do Modelo 4E. O monitoramento "
                    "prospectivo utiliza exclusivamente os "
                    "registros preservados no histórico do dashboard."
                )

                st.warning(
                    "As previsões de HHI e CR5 ainda estão "
                    "em auditoria de alinhamento temporal."
                )
            else:
                st.warning(
                    "Arquivo de previsões do Modelo 4E não encontrado."
                )


        painel_historico_v13(arquivo_futuro)

        with tab_competicao:
            st.markdown(
                "#### Modelo 4E — Contribuição incremental da competição"
            )

            st.caption(
                "Experimento V2: referência Markov compartilhada, "
                "previsões walk-forward e comparação com amostra alinhada."
            )

            pasta_v2 = PROJECT_ROOT / "outputs_modelo4E_competicao_v2"

            caminho_previsoes = pasta_v2 / "previsoes_walk_forward.csv"
            caminho_metricas = pasta_v2 / "metricas.csv"
            caminho_validacao = (
                pasta_v2
                / "validacao_estatistica"
                / "resumo_validacao.csv"
            )

            arquivos_v2 = [
                caminho_previsoes,
                caminho_metricas,
                caminho_validacao
            ]

            if all(caminho.exists() for caminho in arquivos_v2):

                previsoes_v2 = pd.read_csv(caminho_previsoes)
                metricas_v2 = pd.read_csv(caminho_metricas)
                validacao_v2 = pd.read_csv(caminho_validacao)

                st.markdown("##### Desempenho dos modelos")

                st.dataframe(
                    metricas_v2,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("##### Previsões versus valores realizados")

                nomes_modelos = {
                    "realizado": "Realizado",
                    "persistencia": "Persistência",
                    "mlp_convencional": "MLP convencional",
                    "mlp_hibrido": "Markov-Neural",
                    "mlp_hibrido_alinhado": "Markov-Neural alinhado",
                    "modelo4E_competicao": "Markov-Neural + competição"
                }

                colunas_grafico = [
                    coluna for coluna in nomes_modelos
                    if coluna in previsoes_v2.columns
                ]

                grafico_v2 = previsoes_v2.set_index("alvo")[
                    colunas_grafico
                ].rename(columns=nomes_modelos)

                st.line_chart(grafico_v2)

                st.markdown("##### Comparação por trimestre")

                st.dataframe(
                    previsoes_v2,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("##### Validação estatística")

                st.dataframe(
                    validacao_v2,
                    use_container_width=True,
                    hide_index=True
                )

                controle = metricas_v2.loc[
                    metricas_v2["modelo"] == "mlp_hibrido_alinhado"
                ]

                competicao = metricas_v2.loc[
                    metricas_v2["modelo"] == "modelo4E_competicao"
                ]

                if len(controle) == 1 and len(competicao) == 1:

                    mae_controle = float(controle["MAE"].iloc[0])
                    mae_competicao = float(competicao["MAE"].iloc[0])

                    variacao = (
                        (mae_competicao / mae_controle - 1) * 100
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "MAE — híbrido alinhado",
                        f"{mae_controle:.4f}"
                    )

                    c2.metric(
                        "MAE — competição",
                        f"{mae_competicao:.4f}"
                    )

                    c3.metric(
                        "Variação do MAE",
                        f"{variacao:+.2f}%"
                    )

                st.info(
                    "Na avaliação de 15 trimestres, a inclusão "
                    "dos indicadores de competição não demonstrou "
                    "melhoria estatisticamente significativa. "
                    "Consulte os resultados HAC e bootstrap "
                    "na tabela de validação."
                )

                st.caption(
                    "Resultados retrospectivos. A defasagem trimestral "
                    "não substitui uma auditoria das datas efetivas "
                    "de publicação dos indicadores de competição."
                )

            else:

                st.warning(
                    "Arquivos V2 do Modelo 4E não encontrados."
                )

                for caminho in arquivos_v2:
                    if not caminho.exists():
                        st.write(f"Ausente: {caminho}")

elif pagina == "Dicionário":
    st.subheader("Dicionário de indicadores")

    if dicionario.empty:
        st.warning("O arquivo data/dicionario_metricas.csv não foi encontrado. Rode novamente pipeline_mensal.py.")
    else:
        st.caption(f"{len(dicionario)} métricas cadastradas no arquivo do pipeline.")
        dic = dicionario.copy()

        # ETAPA 199 — Variáveis da validação experimental.
        # Não são adicionadas às bases de treinamento.
        _extras_199 = [
            {
                "metrica": "pib_crescimento_tri_pct",
                "nome_amigavel": "Crescimento trimestral real do PIB",
                "grupo": "Macroeconomia",
                "classificacao": "Validação experimental",
                "descricao": "Variação trimestral real do Produto Interno Bruto.",
                "fonte": "IBGE — base experimental do Modelo 4",
                "referencia_fonte": "outputs_modelo4_pib/base_modelo4_pib_validacao.csv",
                "construcao": "Série trimestral utilizada na validação experimental.",
                "unidade": "%",
                "frequencia": "Trimestral",
                "uso_modelos": "Validação experimental",
            },
            {
                "metrica": "pib_crescimento_lag1",
                "nome_amigavel": "Crescimento do PIB defasado",
                "grupo": "Macroeconomia",
                "classificacao": "Validação experimental",
                "descricao": "Crescimento trimestral real do PIB com defasagem de um trimestre.",
                "fonte": "IBGE — transformação experimental",
                "referencia_fonte": "outputs_modelo4_pib/base_modelo4_pib_validacao.csv",
                "construcao": "Defasagem trimestral da variável de crescimento do PIB.",
                "unidade": "%",
                "frequencia": "Trimestral",
                "uso_modelos": "Validação experimental",
            },
            {
                "metrica": "periodo_eleitoral",
                "nome_amigavel": "Período eleitoral presidencial",
                "grupo": "Política e instituições",
                "classificacao": "Validação experimental",
                "descricao": "Indicador binário dos trimestres classificados como eleitorais presidenciais.",
                "fonte": "Calendário eleitoral — classificação experimental",
                "referencia_fonte": "outputs_modelo4_pib/base_modelo4_pib_validacao.csv",
                "construcao": "Indicador binário: 1 para trimestre eleitoral, 0 para demais períodos.",
                "unidade": "0/1",
                "frequencia": "Trimestral",
                "uso_modelos": "Validação experimental",
            },
        ]

        _existentes_199 = (
            set(dic["metrica"].astype(str))
            if "metrica" in dic.columns
            else set()
        )

        _novos_199 = [
            item for item in _extras_199
            if item["metrica"] not in _existentes_199
        ]

        if _novos_199:
            dic = pd.concat(
                [dic, pd.DataFrame(_novos_199)],
                ignore_index=True,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            grupos = sorted(dic["grupo"].dropna().astype(str).unique().tolist()) if "grupo" in dic.columns else []
            grupo_sel = st.multiselect("Grupo", grupos)
        with c2:
            classificacoes = sorted(dic["classificacao"].dropna().astype(str).unique().tolist()) if "classificacao" in dic.columns else []
            class_sel = st.multiselect("Classificação", classificacoes)
        with c3:
            busca = st.text_input("Buscar métrica", placeholder="Ex.: HHI, fintechs, renda...")

        if grupo_sel and "grupo" in dic.columns:
            dic = dic[dic["grupo"].isin(grupo_sel)]
        if class_sel and "classificacao" in dic.columns:
            dic = dic[dic["classificacao"].isin(class_sel)]
        if busca:
            termo = busca.strip().lower()
            mascara = pd.Series(False, index=dic.index)
            for c in ["metrica", "nome_amigavel", "descricao", "fonte", "grupo"]:
                if c in dic.columns:
                    mascara |= dic[c].fillna("").astype(str).str.lower().str.contains(termo, regex=False)
            dic = dic[mascara]

        # Aceita nomes alternativos gerados por versões diferentes do pipeline.
        aliases = {
            "primeira_data": "primeira_observacao",
            "ultima_data": "ultima_observacao",
        }
        for origem, destino in aliases.items():
            if origem in dic.columns and destino not in dic.columns:
                dic[destino] = dic[origem]

        # ------------------------------------------------------------
        # Metadados de cobertura calculados diretamente da base atual.
        # Isso evita que o Dicionário exiba datas/frequências antigas após
        # uma atualização mensal, trimestral ou anual do pipeline.
        # ------------------------------------------------------------
        if "metrica" in dic.columns and not mensal.empty and COL_DATA_MENSAL:
            datas_base = pd.to_datetime(mensal[COL_DATA_MENSAL], errors="coerce")

            for idx, metrica in dic["metrica"].astype(str).items():
                # A frequência exibida é a frequência ORIGINAL da série,
                # não a frequência da grade mensal que a armazena.
                dic.at[idx, "frequencia"] = frequencia_metrica(metrica).capitalize()

                if metrica not in mensal.columns:
                    continue

                valores = pd.to_numeric(mensal[metrica], errors="coerce")
                mascara_valida = valores.notna() & datas_base.notna()
                if not mascara_valida.any():
                    dic.at[idx, "primeira_observacao"] = pd.NA
                    dic.at[idx, "ultima_observacao"] = pd.NA
                    dic.at[idx, "observacoes_validas"] = 0
                    continue

                datas_validas = datas_base.loc[mascara_valida]
                primeira = datas_validas.min()
                ultima = datas_validas.max()
                n_validas = int(mascara_valida.sum())
                freq = frequencia_metrica(metrica)

                if freq == "trimestral":
                    primeira_txt = str(primeira.to_period("Q"))
                    ultima_txt = str(ultima.to_period("Q"))
                    n_possiveis = len(pd.period_range(primeira.to_period("Q"), ultima.to_period("Q"), freq="Q"))
                elif freq == "anual":
                    primeira_txt = str(primeira.year)
                    ultima_txt = str(ultima.year)
                    n_possiveis = ultima.year - primeira.year + 1
                else:
                    primeira_txt = primeira.strftime("%Y-%m")
                    ultima_txt = ultima.strftime("%Y-%m")
                    n_possiveis = len(pd.period_range(primeira.to_period("M"), ultima.to_period("M"), freq="M"))

                dic.at[idx, "primeira_observacao"] = primeira_txt
                dic.at[idx, "ultima_observacao"] = ultima_txt
                dic.at[idx, "observacoes_validas"] = n_validas
                dic.at[idx, "observacoes_possiveis"] = int(n_possiveis)
                dic.at[idx, "cobertura_pct"] = round(100 * n_validas / n_possiveis, 1) if n_possiveis else pd.NA

        ordem_colunas = [
            "metrica",
            "nome_amigavel",
            "grupo",
            "classificacao",
            "descricao",
            "fonte",
            "referencia_fonte",
            "construcao",
            "unidade",
            "frequencia",
            "primeira_observacao",
            "ultima_observacao",
            "observacoes_validas",
            "observacoes_possiveis",
            "cobertura_pct",
            "uso_modelos",
        ]
        ordem_colunas = [c for c in ordem_colunas if c in dic.columns]
        exibir = dic[ordem_colunas].copy() if ordem_colunas else dic.copy()
        st.dataframe(exibir, width="stretch", hide_index=True, height=650)
        st.caption(
            "Frequência, primeira observação, última observação e cobertura são recalculadas a partir dos valores "
            "efetivamente disponíveis na base atual. Séries trimestrais permanecem apenas nos trimestres observados, "
            "sem forward fill; séries anuais também não são replicadas artificialmente nos demais meses."
        )
