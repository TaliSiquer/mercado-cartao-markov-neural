"""
pipeline_mensal.py
=========================================================
Pipeline automatizado do framework de acompanhamento
do mercado brasileiro de cartão de crédito PF.

ETAPA 1:
- lê dataset_model_ready.csv;
- valida a base;
- gera tabelão mensal;
- gera tabelão trimestral;
- classifica Pré / Pandemia / Pós;
- gera fotografia atual das métricas;
- gera tabela de status;
- calcula correlações;
- prepara histórico de previsões;
- registra log da execução.

Posteriormente serão integrados:
- coleta automática BCB;
- IFData;
- IBGE;
- Modelos 1 e 2;
- Markov;
- Modelo 4D;
- Modelo 4E.
=========================================================
"""

from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

import config
from coleta_fgv_iie import update as atualizar_iie_fgv, integrate as integrar_iie_fgv


warnings.filterwarnings("ignore")


# =========================================================
# 1. CONFIGURAÇÕES LOCAIS
# =========================================================

ARQUIVO_CORRELACOES = (
    config.DATA_DIR / "correlacoes_dashboard.csv"
)

ARQUIVO_COMPARACAO_PERIODOS = (
    config.DATA_DIR / "comparacao_pre_pandemia_pos.csv"
)

ARQUIVO_LOG_PIPELINE = (
    config.LOGS_DIR / "pipeline_mensal.log"
)


# =========================================================
# 2. LOG
# =========================================================

def log(mensagem):
    """
    Exibe uma mensagem no terminal e grava no arquivo de log.
    """

    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    texto = f"[{agora}] {mensagem}"

    print(texto)

    config.LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        ARQUIVO_LOG_PIPELINE,
        "a",
        encoding="utf-8"
    ) as arquivo:
        arquivo.write(texto + "\n")


# =========================================================
# 3. CLASSIFICAÇÃO PRÉ / PANDEMIA / PÓS
# =========================================================

def classificar_periodo_pandemia(data):
    """
    Classifica cada observação em:

    - Pré-pandemia
    - Pandemia
    - Pós-pandemia
    """

    if pd.isna(data):
        return np.nan

    data = pd.Timestamp(data)

    if data <= pd.Timestamp(config.PRE_PANDEMIA_FIM):
        return "Pré-pandemia"

    elif (
        pd.Timestamp(config.PANDEMIA_INICIO)
        <= data
        <= pd.Timestamp(config.PANDEMIA_FIM)
    ):
        return "Pandemia"

    else:
        return "Pós-pandemia"


# =========================================================
# 4. LEITURA DA BASE
# =========================================================

def carregar_base():
    """
    Carrega o dataset principal.
    """

    arquivo = config.ARQUIVO_BASE_ORIGINAL

    if not arquivo.exists():
        raise FileNotFoundError(
            "\nArquivo não encontrado:\n"
            f"{arquivo}\n\n"
            "Coloque dataset_model_ready.csv "
            "na pasta principal do projeto."
        )

    log(
        f"Lendo base: {arquivo.name}"
    )

    df = pd.read_csv(arquivo)

    if config.COLUNA_DATA not in df.columns:
        raise ValueError(
            f"A coluna '{config.COLUNA_DATA}' "
            "não existe na base."
        )

    df[config.COLUNA_DATA] = pd.to_datetime(
        df[config.COLUNA_DATA],
        errors="coerce"
    )

    df = df.dropna(
        subset=[config.COLUNA_DATA]
    )

    df = df.sort_values(
        config.COLUNA_DATA
    ).reset_index(drop=True)

    log(
        f"Base carregada: "
        f"{len(df)} observações e "
        f"{len(df.columns)} colunas."
    )

    log(
        "Período da base: "
        f"{df[config.COLUNA_DATA].min():%Y-%m} "
        "até "
        f"{df[config.COLUNA_DATA].max():%Y-%m}"
    )

    return df


# =========================================================
# 5. CONVERSÃO DAS COLUNAS NUMÉRICAS
# =========================================================

def preparar_tipos(df):
    """
    Tenta converter para numérico todas as colunas,
    exceto a coluna de data.
    """

    df = df.copy()

    for coluna in df.columns:

        if coluna == config.COLUNA_DATA:
            continue

        # Não força conversão de variáveis textuais
        if df[coluna].dtype == "object":

            convertido = pd.to_numeric(
                df[coluna],
                errors="coerce"
            )

            # Só substitui se existir alguma informação
            # numérica válida.
            if convertido.notna().sum() > 0:
                df[coluna] = convertido

    return df


# =========================================================
# 6. TABELÃO MENSAL
# =========================================================

def criar_master_mensal(df):
    """
    Cria a base mensal de acompanhamento.

    NÃO faz forward fill de HHI, CR5 ou outras
    métricas trimestrais.
    """

    mensal = df.copy()

    mensal["ano"] = (
        mensal[config.COLUNA_DATA].dt.year
    )

    mensal["mes"] = (
        mensal[config.COLUNA_DATA].dt.month
    )

    mensal["periodo_mes"] = (
        mensal[config.COLUNA_DATA]
        .dt.to_period("M")
        .astype(str)
    )

    mensal["trimestre"] = (
        mensal[config.COLUNA_DATA]
        .dt.to_period("Q")
        .astype(str)
    )

    mensal["periodo_pandemia"] = (
        mensal[config.COLUNA_DATA]
        .apply(classificar_periodo_pandemia)
    )

    mensal.to_csv(
        config.ARQUIVO_MASTER_MENSAL,
        index=False
    )

    log(
        "Tabelão mensal salvo: "
        f"{config.ARQUIVO_MASTER_MENSAL.name}"
    )

    return mensal


# =========================================================
# 7. TABELÃO TRIMESTRAL
# =========================================================

def criar_master_trimestral(df):
    """
    Cria a base trimestral usada nas análises e modelos.

    Regras:
    - utiliza apenas trimestres completos;
    - não faz forward fill;
    - não faz backward fill;
    - preserva valores ausentes.
    """

    df_tri = df.copy()

    # Identifica o trimestre de cada observação
    df_tri["_quarter"] = (
        df_tri[config.COLUNA_DATA]
        .dt.to_period("Q")
    )

    # =====================================================
    # Considerar somente trimestres completos
    # =====================================================

    trimestre_atual = (
        pd.Timestamp.today()
        .to_period("Q")
    )

    ultimo_trimestre_completo = (
        trimestre_atual - 1
    )

    trimestres_excluidos = sorted(
        df_tri.loc[
            df_tri["_quarter"] > ultimo_trimestre_completo,
            "_quarter"
        ]
        .astype(str)
        .unique()
    )

    if trimestres_excluidos:
        log(
            "Trimestres incompletos excluídos "
            "da base trimestral: "
            + ", ".join(trimestres_excluidos)
        )

    df_tri = df_tri.loc[
        df_tri["_quarter"] <= ultimo_trimestre_completo
    ].copy()

    # VERIFICACAO_COBERTURA_TRIMESTRAL
    # Exige três meses distintos para os indicadores
    # mensais utilizados no monitoramento preditivo.

    metricas_cobertura = [
        "juros_cartao_pf_aa_pct",
        "inadimplencia_90d_pct",
    ]

    df_tri["_mes_cobertura"] = (
        df_tri[config.COLUNA_DATA].dt.to_period("M")
    )

    for metrica in metricas_cobertura:
        if metrica not in df_tri.columns:
            raise ValueError(
                f"Indicador obrigatório ausente: {metrica}"
            )

    trimestres_validos = None

    for metrica in metricas_cobertura:
        cobertura = (
            df_tri.loc[df_tri[metrica].notna()]
            .groupby("_quarter")["_mes_cobertura"]
            .nunique()
        )

        validos = set(cobertura[cobertura == 3].index)

        trimestres_validos = (
            validos if trimestres_validos is None
            else trimestres_validos & validos
        )

    excluidos_cobertura = sorted(
        set(df_tri["_quarter"]) - trimestres_validos
    )

    if excluidos_cobertura:
        log(
            "Trimestres excluídos por cobertura mensal "
            "insuficiente: "
            + ", ".join(map(str, excluidos_cobertura))
        )

    df_tri = df_tri.loc[
        df_tri["_quarter"].isin(trimestres_validos)
    ].drop(columns=["_mes_cobertura"]).copy()

    # =====================================================
    # Colunas numéricas
    # =====================================================

    colunas_numericas = (
        df_tri
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    # =====================================================
    # Agregação trimestral
    # =====================================================

    trimestral = (
        df_tri
        .groupby("_quarter")[colunas_numericas]
        .mean()
        .reset_index()
    )

    # Data representativa:
    # último dia do trimestre
    trimestral[config.COLUNA_DATA] = (
        trimestral["_quarter"]
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )

    trimestral["ano"] = (
        trimestral["_quarter"].dt.year
    )

    trimestral["trimestre_num"] = (
        trimestral["_quarter"].dt.quarter
    )

    trimestral["trimestre"] = (
        trimestral["_quarter"]
        .astype(str)
    )

    # =====================================================
    # Pré / Pandemia / Pós
    # =====================================================

    trimestral["periodo_pandemia"] = (
        trimestral[config.COLUNA_DATA]
        .apply(classificar_periodo_pandemia)
    )

    trimestral = trimestral.drop(
        columns=["_quarter"]
    )

    # Coloca a data na primeira coluna
    cols = [
        config.COLUNA_DATA
    ] + [
        c
        for c in trimestral.columns
        if c != config.COLUNA_DATA
    ]

    trimestral = trimestral[cols]

    # =====================================================
    # Salvar
    # =====================================================

    trimestral.to_csv(
        config.ARQUIVO_MASTER_TRIMESTRAL,
        index=False,
        encoding="utf-8-sig"
    )

    log(
        "Tabelão trimestral salvo: "
        f"{config.ARQUIVO_MASTER_TRIMESTRAL.name}"
    )

    log(
        "Total de trimestres completos: "
        f"{len(trimestral)}"
    )

    log(
        "Último trimestre completo considerado: "
        f"{ultimo_trimestre_completo}"
    )

    return trimestral

# =========================================================
# 8. DESCOBRIR ÚLTIMO DADO DE CADA MÉTRICA
# =========================================================

def buscar_ultima_observacao(df, metrica):
    """
    Busca o último valor realmente observado
    de uma métrica.

    Não preenche períodos posteriores.
    """

    if metrica not in df.columns:
        return None

    temp = (
        df[
            [
                config.COLUNA_DATA,
                metrica
            ]
        ]
        .dropna(subset=[metrica])
        .sort_values(config.COLUNA_DATA)
    )

    if temp.empty:
        return None

    ultima = temp.iloc[-1]

    return {
        "valor": ultima[metrica],
        "data": ultima[config.COLUNA_DATA]
    }


# =========================================================
# 9. FOTOGRAFIA ATUAL DO DASHBOARD
# =========================================================

def criar_metricas_dashboard(df):
    """
    Cria tabela com o último valor disponível
    de cada indicador.

    Importante:
    HHI/CR5 mantêm o período real de sua fonte.
    """

    registros = []

    hoje = pd.Timestamp.today()

    for metrica in config.METRICAS_DASHBOARD:

        if metrica not in df.columns:
            continue

        resultado = buscar_ultima_observacao(
            df,
            metrica
        )

        if resultado is None:
            continue

        data_fonte = pd.Timestamp(
            resultado["data"]
        )

        frequencia = (
            config.frequencia_metrica(metrica)
        )

        if frequencia == "trimestral":
            periodo_fonte = str(
                data_fonte.to_period("Q")
            )
        else:
            periodo_fonte = str(
                data_fonte.to_period("M")
            )

        registros.append(
            {
                "metrica": metrica,
                "nome_metrica":
                    config.nome_metrica(metrica),

                "valor":
                    resultado["valor"],

                "periodo_fonte":
                    periodo_fonte,

                "data_fonte":
                    data_fonte.date(),

                "frequencia":
                    frequencia,

                "formato":
                    config.formato_metrica(
                        metrica
                    ),

                "data_atualizacao_pipeline":
                    hoje.strftime("%Y-%m-%d"),
            }
        )

    dashboard = pd.DataFrame(
        registros
    )

    dashboard.to_csv(
        config.ARQUIVO_METRICAS_DASHBOARD,
        index=False
    )

    log(
        "Fotografia atual do dashboard salva: "
        f"{config.ARQUIVO_METRICAS_DASHBOARD.name}"
    )

    return dashboard


# =========================================================
# 10. STATUS DE ATUALIZAÇÃO
# =========================================================

def criar_status_atualizacao(df):
    """
    Mostra até qual período cada métrica possui dado.
    """

    registros = []

    data_maxima_base = (
        df[config.COLUNA_DATA].max()
    )

    for metrica in config.METRICAS_DASHBOARD:

        if metrica not in df.columns:

            registros.append(
                {
                    "metrica": metrica,
                    "nome_metrica":
                        config.nome_metrica(metrica),

                    "ultimo_periodo":
                        None,

                    "ultimo_valor":
                        None,

                    "frequencia":
                        config.frequencia_metrica(
                            metrica
                        ),

                    "status":
                        "Coluna não encontrada",
                }
            )

            continue

        resultado = buscar_ultima_observacao(
            df,
            metrica
        )

        if resultado is None:
            continue

        data = pd.Timestamp(
            resultado["data"]
        )

        frequencia = (
            config.frequencia_metrica(
                metrica
            )
        )

        if frequencia == "trimestral":

            periodo = str(
                data.to_period("Q")
            )

            atraso = (
                data_maxima_base.to_period("Q")
                - data.to_period("Q")
            ).n

        else:

            periodo = str(
                data.to_period("M")
            )

            atraso = (
                data_maxima_base.to_period("M")
                - data.to_period("M")
            ).n

        if atraso <= 0:
            status = "Atualizado"

        elif atraso == 1:
            status = "1 período de defasagem"

        else:
            status = (
                f"{atraso} períodos de defasagem"
            )

        registros.append(
            {
                "metrica":
                    metrica,

                "nome_metrica":
                    config.nome_metrica(
                        metrica
                    ),

                "ultimo_periodo":
                    periodo,

                "ultimo_valor":
                    resultado["valor"],

                "frequencia":
                    frequencia,

                "status":
                    status,
            }
        )

    status_df = pd.DataFrame(
        registros
    )

    status_df.to_csv(
        config.ARQUIVO_STATUS_ATUALIZACAO,
        index=False
    )

    log(
        "Status de atualização salvo: "
        f"{config.ARQUIVO_STATUS_ATUALIZACAO.name}"
    )

    return status_df


# =========================================================
# 11. CORRELAÇÕES
# =========================================================

def criar_correlacoes(df_trimestral):
    """
    Calcula matriz de correlação usando apenas
    variáveis disponíveis.

    Usa frequência trimestral para evitar misturar
    séries mensais e trimestrais diretamente.
    """

    colunas = [
        c
        for c in config.METRICAS_CORRELACAO
        if c in df_trimestral.columns
    ]

    if len(colunas) < 2:
        log(
            "Correlação não gerada: "
            "menos de duas métricas disponíveis."
        )

        return pd.DataFrame()

    matriz = (
        df_trimestral[colunas]
        .corr(method="pearson")
    )

    matriz.to_csv(
        ARQUIVO_CORRELACOES
    )

    log(
        "Matriz de correlações salva: "
        f"{ARQUIVO_CORRELACOES.name}"
    )

    return matriz


# =========================================================
# 12. PRÉ / PANDEMIA / PÓS
# =========================================================

def criar_comparacao_periodos(
    df_trimestral
):
    """
    Calcula médias das principais métricas
    por período:

    - Pré-pandemia
    - Pandemia
    - Pós-pandemia
    """

    metricas = [
        c
        for c in config.METRICAS_DASHBOARD
        if c in df_trimestral.columns
    ]

    if not metricas:
        return pd.DataFrame()

    comparacao = (
        df_trimestral
        .groupby(
            "periodo_pandemia",
            observed=True
        )[metricas]
        .mean()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "metrica"
            }
        )
    )

    comparacao.insert(
        1,
        "nome_metrica",
        comparacao["metrica"].map(
            config.NOMES_METRICAS
        ).fillna(
            comparacao["metrica"]
        )
    )

    ordem = [
        "Pré-pandemia",
        "Pandemia",
        "Pós-pandemia",
    ]

    for coluna in ordem:
        if coluna not in comparacao.columns:
            comparacao[coluna] = np.nan

    comparacao = comparacao[
        [
            "metrica",
            "nome_metrica",
            *ordem
        ]
    ]

    # Variação pré -> pós
    comparacao[
        "variacao_pre_pos"
    ] = (
        comparacao["Pós-pandemia"]
        - comparacao["Pré-pandemia"]
    )

    comparacao.to_csv(
        ARQUIVO_COMPARACAO_PERIODOS,
        index=False
    )

    log(
        "Comparação Pré/Pandemia/Pós salva: "
        f"{ARQUIVO_COMPARACAO_PERIODOS.name}"
    )

    return comparacao


# =========================================================
# 13. HISTÓRICO DE PREVISÕES
# =========================================================

def preparar_historico_previsoes():
    """
    Cria o histórico de previsões caso ainda
    não exista.

    O arquivo nunca deve ser apagado automaticamente.
    """

    arquivo = (
        config.ARQUIVO_HISTORICO_PREVISOES
    )

    if arquivo.exists():

        historico = pd.read_csv(
            arquivo
        )

        log(
            "Histórico de previsões existente: "
            f"{len(historico)} registros."
        )

        return historico

    historico = pd.DataFrame(
        columns=config.COLUNAS_HISTORICO_PREVISOES
    )

    historico.to_csv(
        arquivo,
        index=False
    )

    log(
        "Histórico de previsões criado."
    )

    return historico


# =========================================================
# 14. VERIFICAÇÃO DE NOVO TRIMESTRE
# =========================================================

def verificar_ultimo_trimestre(
    df_trimestral
):
    """
    Identifica o último trimestre disponível.

    Neste primeiro estágio apenas informa.
    Depois essa função decidirá se os modelos
    precisam ser executados novamente.
    """

    if df_trimestral.empty:
        return None

    ultimo = (
        df_trimestral[
            config.COLUNA_DATA
        ]
        .max()
        .to_period("Q")
    )

    log(
        f"Último trimestre presente na base: "
        f"{ultimo}"
    )

    return ultimo


# =========================================================
# 15. RESUMO DA EXECUÇÃO
# =========================================================

def mostrar_resumo(
    mensal,
    trimestral,
    dashboard,
    status,
    comparacao
):
    """
    Exibe resumo final no terminal.
    """

    print("\n")
    print("=" * 70)
    print("PIPELINE CONCLUÍDO")
    print("=" * 70)

    print(
        f"\nBase mensal: "
        f"{len(mensal)} observações"
    )

    print(
        f"Base trimestral: "
        f"{len(trimestral)} observações"
    )

    print(
        f"Métricas do dashboard: "
        f"{len(dashboard)}"
    )

    print("\nÚLTIMOS DADOS DISPONÍVEIS")
    print("-" * 70)

    if not dashboard.empty:

        colunas = [
            "nome_metrica",
            "valor",
            "periodo_fonte",
        ]

        print(
            dashboard[colunas]
            .to_string(index=False)
        )

    print("\nSTATUS")
    print("-" * 70)

    if not status.empty:

        print(
            status[
                [
                    "nome_metrica",
                    "ultimo_periodo",
                    "status"
                ]
            ]
            .to_string(index=False)
        )

    if not comparacao.empty:

        print("\nPRÉ / PANDEMIA / PÓS")
        print("-" * 70)

        print(
            comparacao[
                [
                    "nome_metrica",
                    "Pré-pandemia",
                    "Pandemia",
                    "Pós-pandemia",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print("\nArquivos gerados em:")
    print(config.DATA_DIR)

    print("\n" + "=" * 70)


# =========================================================
# 16. PIPELINE PRINCIPAL
# =========================================================

def executar_pipeline():

    inicio = datetime.now()

    log("=" * 70)
    log("INÍCIO DO PIPELINE MENSAL")
    log("=" * 70)

    try:

        # -------------------------------------------------
        # ETAPA 1 — base
        # -------------------------------------------------

        df = carregar_base()

        df = preparar_tipos(df)

        # -------------------------------------------------
        # ETAPA 2 — tabelão mensal
        # -------------------------------------------------

        mensal = criar_master_mensal(df)

        # -------------------------------------------------
        # ETAPA 3 — tabelão trimestral
        # -------------------------------------------------

        trimestral = criar_master_trimestral(
            df
        )

        # -------------------------------------------------
        # ETAPA 3B — IIE-Br: coletar da FGV ou usar cache
        # -------------------------------------------------
        iie = atualizar_iie_fgv(config.DATA_DIR, logger=log)
        if iie is not None:
            mensal, trimestral = integrar_iie_fgv(mensal, trimestral, iie)
            mensal.to_csv(config.ARQUIVO_MASTER_MENSAL, index=False, encoding="utf-8-sig")
            trimestral.to_csv(config.ARQUIVO_MASTER_TRIMESTRAL, index=False, encoding="utf-8-sig")
            log("FGV: IIE-Br incorporado aos tabelões mensal e trimestral")

        # -------------------------------------------------
        # ETAPA 4 — fotografia atual
        # -------------------------------------------------

        dashboard = (
            criar_metricas_dashboard(df)
        )

        # -------------------------------------------------
        # ETAPA 5 — status
        # -------------------------------------------------

        status = (
            criar_status_atualizacao(df)
        )

        # -------------------------------------------------
        # ETAPA 6 — correlações
        # -------------------------------------------------

        criar_correlacoes(
            trimestral
        )

        # -------------------------------------------------
        # ETAPA 7 — pandemia
        # -------------------------------------------------

        comparacao = (
            criar_comparacao_periodos(
                trimestral
            )
        )

        # -------------------------------------------------
        # ETAPA 8 — previsões
        # -------------------------------------------------

        preparar_historico_previsoes()

        # -------------------------------------------------
        # ETAPA 8B — atualizar realizados das previsões
        # -------------------------------------------------

        import subprocess
        import sys

        atualizador = (
            Path(__file__).resolve().parent.parent
            / "atualizar_realizados_dashboard.py"
        )

        processo_realizados = subprocess.run(
            [sys.executable, str(atualizador)],
            cwd=atualizador.parent,
            capture_output=True,
            text=True,
        )

        if processo_realizados.stdout:
            log(processo_realizados.stdout)

        if processo_realizados.returncode != 0:
            log(processo_realizados.stderr)
            raise RuntimeError(
                "Falha na atualização dos valores realizados."
            )

        # -------------------------------------------------
        # ETAPA 8C — sincronizar realizados do histórico v13
        # -------------------------------------------------

        atualizador_v13 = (
            Path(__file__).resolve().parent.parent
            / "sincronizar_realizados_v13.py"
        )

        log(
            "Iniciando sincronização dos realizados "
            "do histórico v13."
        )

        processo_v13 = subprocess.run(
            [sys.executable, str(atualizador_v13)],
            cwd=atualizador_v13.parent,
            capture_output=True,
            text=True,
        )

        if processo_v13.stdout:
            log(processo_v13.stdout)

        if processo_v13.returncode != 0:
            if processo_v13.stderr:
                log(processo_v13.stderr)

            raise RuntimeError(
                "Falha na sincronização dos realizados v13."
            )

        log(
            "Sincronização dos realizados v13 concluída."
        )

        # -------------------------------------------------
        # ETAPA 9 — trimestre
        # -------------------------------------------------

        verificar_ultimo_trimestre(
            trimestral
        )

        # -------------------------------------------------
        # RESUMO
        # -------------------------------------------------

        mostrar_resumo(
            mensal,
            trimestral,
            dashboard,
            status,
            comparacao
        )

        tempo = (
            datetime.now() - inicio
        ).total_seconds()

        log(
            f"Pipeline concluído com sucesso "
            f"em {tempo:.2f} segundos."
        )

        log("=" * 70)

    except Exception as erro:

        log(
            f"ERRO NO PIPELINE: "
            f"{type(erro).__name__}: {erro}"
        )

        log("=" * 70)

        raise


# =========================================================
# 17. EXECUÇÃO
# =========================================================

if __name__ == "__main__":
    executar_pipeline()