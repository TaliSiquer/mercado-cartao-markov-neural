from pathlib import Path
from datetime import datetime
import subprocess
import sys
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PASTA_LOGS = BASE_DIR / "logs_pipeline"
PASTA_LOGS.mkdir(exist_ok=True)

AGORA = datetime.now()

ARQUIVO_LOG = (
    PASTA_LOGS
    / f"pipeline_{AGORA.strftime('%Y-%m-%d_%H-%M-%S')}.log"
)


# ============================================================
# ETAPAS DA PIPELINE
# ============================================================

ETAPAS = [{'nome': 'Atualização Banco Central / IFData / CVM',
  'script': 'DataSet/bacen.py',
  'obrigatoria': True},
 {'nome': 'Atualização IBGE', 'script': 'DataSet/ibge.py', 'obrigatoria': True},
 {'nome': 'Preparação da base para modelagem', 'script': 'model_prep.py', 'obrigatoria': True},
 {'nome': 'Integração BCB + IBGE', 'script': 'merge_datasets.py', 'obrigatoria': True},
 {'nome': 'Campos calculados', 'script': 'campos_calculados.py', 'obrigatoria': True},
 {'nome': 'Atualização das bases mensais e trimestrais',
  'script': 'Mercado_de_Cartao/pipeline_mensal.py',
  'obrigatoria': True},
 {'nome': 'Modelo 2A', 'script': 'modelo2A.py', 'obrigatoria': False},
 {'nome': 'Modelo 2B', 'script': 'modelo2B.py', 'obrigatoria': False},
 {'nome': 'Modelo 2C', 'script': 'modelo2C.py', 'obrigatoria': False},
 {'nome': 'Modelo 2D', 'script': 'modelo2D.py', 'obrigatoria': False},
 {'nome': 'Modelo 3C — Markov Switching', 'script': 'modelo3c.py', 'obrigatoria': True},
 {'nome': 'Modelo 4 Final — Validação + Previsão',
  'script': 'modelo4_final.py',
  'obrigatoria': True},
 {'nome': 'Modelo 4E — Previsão das métricas',
  'script': 'modelo4E_metricas.py',
  'obrigatoria': True},
 {'nome': 'Arquivamento das evidências — Modelo 4E',
  'script': 'arquivar_evidencias_modelo4E.py',
  'obrigatoria': True},
 {'nome': 'Registro auditável — Modelo 4E',
  'script': 'registrar_execucao_modelo4E.py',
  'argumentos': ['--registrar'],
  'obrigatoria': True}]


# ============================================================
# FUNÇÕES DE LOG
# ============================================================

def registrar(texto=""):
    horario = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if texto:
        linha = f"[{horario}] {texto}"
    else:
        linha = f"[{horario}]"

    print(linha)

    with open(ARQUIVO_LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def registrar_saida(texto):
    if not texto:
        return

    with open(ARQUIVO_LOG, "a", encoding="utf-8") as f:
        f.write(texto)

        if not texto.endswith("\n"):
            f.write("\n")


def separador():
    registrar("=" * 100)


# ============================================================
# EXECUÇÃO DE SCRIPT
# ============================================================

def executar_script(etapa):
    nome = etapa["nome"]
    script_relativo = etapa["script"]
    obrigatoria = etapa.get("obrigatoria", False)

    script = BASE_DIR / script_relativo

    separador()
    registrar(nome)
    separador()

    if not script.exists():
        mensagem = f"Arquivo não encontrado: {script_relativo}"

        if obrigatoria:
            registrar(f"ERRO: {mensagem}")
            return False

        registrar(f"AVISO: {mensagem} — etapa ignorada.")
        return True

    registrar(f"Executando: {script_relativo}")

    inicio = datetime.now()

    processo = subprocess.run(
        [sys.executable, str(script), *etapa.get("argumentos", [])],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    fim = datetime.now()

    registrar_saida(processo.stdout)

    if processo.stderr:
        registrar_saida("\n--- STDERR ---\n")
        registrar_saida(processo.stderr)

    if processo.returncode == 0:
        registrar(f"CONCLUÍDO: {script_relativo}")
        registrar(f"Duração: {fim - inicio}")
        return True

    registrar(
        f"ERRO em {script_relativo} "
        f"(código de saída {processo.returncode})"
    )

    if obrigatoria:
        registrar(
            "Pipeline interrompida por erro em etapa obrigatória."
        )
        return False

    registrar(
        "Etapa opcional falhou. Pipeline continuará."
    )

    return True


# ============================================================
# VERIFICAÇÃO DAS BASES
# ============================================================

def verificar_bases():
    separador()
    registrar("VERIFICAÇÃO DAS BASES")
    separador()

    arquivos = [
        "bcb_creditcard_competition_dataset_v4.csv",
        "dataset_model_ready.csv",
        "ibge_macro_dataset.csv",
        "dataset_final_dissertacao.csv",
        "outputs_modelo3C/modelo3C_base_para_neural.csv",
        "outputs_modelo3C/modelo3C_base_para_neural_treino.csv",
        "outputs_modelo4_final/modelo4D_previsao.csv",
        "outputs_modelo4_final/modelo4D_probabilidades.csv",
        "outputs_modelo4_final/modelo4D_mapeamento_regimes.csv",
    ]

    for arquivo_relativo in arquivos:
        arquivo = BASE_DIR / arquivo_relativo

        if arquivo.exists():
            modificacao = datetime.fromtimestamp(
                arquivo.stat().st_mtime
            ).strftime("%d/%m/%Y %H:%M:%S")

            registrar(f"OK  {arquivo_relativo}")
            registrar(
                f"    Última modificação: {modificacao}"
            )

        else:
            registrar(
                f"FALTA  {arquivo_relativo}"
            )


# ============================================================
# COBERTURA TEMPORAL BCB
# ============================================================

def verificar_periodo_bcb():
    separador()
    registrar("COBERTURA TEMPORAL DA BASE BCB")
    separador()

    arquivo = (
        BASE_DIR
        / "bcb_creditcard_competition_dataset_v4.csv"
    )

    if not arquivo.exists():
        registrar("Base BCB v4 não encontrada.")
        return

    try:
        df = pd.read_csv(arquivo)

        if "date" not in df.columns:
            registrar(
                "Coluna 'date' não encontrada."
            )
            return

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = (
            df
            .dropna(subset=["date"])
            .sort_values("date")
        )

        registrar(
            f"Primeiro período: "
            f"{df['date'].min().strftime('%Y-%m')}"
        )

        registrar(
            f"Último período existente: "
            f"{df['date'].max().strftime('%Y-%m')}"
        )

        registrar("")
        registrar(
            "Último período válido por variável:"
        )

        variaveis = [
            "selic_am_pct",
            "juros_cartao_pf_aa_pct",
            "inadimplencia_90d_pct",
            "saldo_cartao_pf_total_mr",
            "comprometimento_renda_pct",
            "hhi_credito_0_1",
            "cr5_credito",
        ]

        for coluna in variaveis:
            if coluna not in df.columns:
                registrar(
                    f"{coluna}: coluna não encontrada"
                )
                continue

            datas_validas = df.loc[
                df[coluna].notna(),
                "date",
            ]

            if len(datas_validas) == 0:
                registrar(
                    f"{coluna}: nenhum dado"
                )

            else:
                registrar(
                    f"{coluna}: "
                    f"{datas_validas.max().strftime('%Y-%m')}"
                )

    except Exception as erro:
        registrar(
            f"Erro ao verificar base BCB: {erro}"
        )


# ============================================================
# VERIFICAÇÃO DA BASE TRIMESTRAL
# ============================================================

def obter_status_trimestre():

    arquivo = BASE_DIR / "dataset_model_ready.csv"

    if not arquivo.exists():
        return None

    try:
        dados = pd.read_csv(arquivo)

        if "quarter" in dados.columns:

            quarters = pd.PeriodIndex(
                dados["quarter"],
                freq="Q"
            )

        elif "date" in dados.columns:

            datas = pd.to_datetime(
                dados["date"],
                errors="coerce"
            )

            datas = datas.dropna()

            if len(datas) == 0:
                return None

            quarters = datas.dt.to_period("Q")

        else:
            return None

        ultimo_periodo = quarters.max()

        hoje = pd.Timestamp.now().normalize()

        fim_ultimo_periodo = (
            ultimo_periodo
            .end_time
            .normalize()
        )

        if hoje >= fim_ultimo_periodo:

            status = "COMPLETO"

            ultimo_completo = (
                ultimo_periodo
            )

        else:

            status = "PARCIAL / EM ANDAMENTO"

            ultimo_completo = (
                ultimo_periodo - 1
            )

        return {
            "ultimo_periodo":
                ultimo_periodo,

            "fim_ultimo_periodo":
                fim_ultimo_periodo,

            "status":
                status,

            "ultimo_completo":
                ultimo_completo,
        }

    except Exception:
        return None


def verificar_base_trimestral():

    arquivo = BASE_DIR / "dataset_model_ready.csv"

    separador()
    registrar("COBERTURA DA BASE TRIMESTRAL")
    separador()

    if not arquivo.exists():

        registrar(
            "Base trimestral não encontrada: "
            f"{arquivo.name}"
        )

        return

    try:

        dados = pd.read_csv(arquivo)

        # ----------------------------------------------------
        # Identificar o período trimestral
        # ----------------------------------------------------

        if "quarter" in dados.columns:

            quarters = pd.PeriodIndex(
                dados["quarter"],
                freq="Q"
            )

        elif "date" in dados.columns:

            datas = pd.to_datetime(
                dados["date"],
                errors="coerce"
            )

            if datas.isna().all():

                registrar(
                    "Não foi possível interpretar "
                    "a coluna 'date'."
                )

                return

            quarters = (
                datas
                .dropna()
                .dt
                .to_period("Q")
            )

        else:

            registrar(
                "Nenhuma coluna temporal reconhecida "
                "foi encontrada."
            )

            registrar(
                "Esperado: 'quarter' ou 'date'."
            )

            return

        primeiro_periodo = quarters.min()
        ultimo_periodo = quarters.max()

        registrar(
            f"Primeiro período: "
            f"{primeiro_periodo}"
        )

        registrar(
            f"Último período disponível: "
            f"{ultimo_periodo}"
        )

        registrar(
            f"Número de observações: "
            f"{len(dados)}"
        )

        status = obter_status_trimestre()

        if status is None:

            registrar(
                "Não foi possível determinar "
                "o status do trimestre."
            )

            return

        hoje = pd.Timestamp.now().normalize()

        registrar("")

        registrar(
            f"Data de referência da execução: "
            f"{hoje.strftime('%d/%m/%Y')}"
        )

        registrar(
            f"Fim do trimestre disponível: "
            f"{status['fim_ultimo_periodo'].strftime('%d/%m/%Y')}"
        )

        registrar("")

        registrar(
            f"Status do último trimestre: "
            f"{status['status']}"
        )

        registrar(
            f"Último trimestre completo: "
            f"{status['ultimo_completo']}"
        )

        if status["status"] != "COMPLETO":

            registrar("")

            registrar(
                "ATENÇÃO: o último trimestre disponível "
                "ainda não terminou."
            )

            registrar(
                "Resultados que utilizam esse trimestre "
                "devem ser interpretados como "
                "prospectivos / parciais."
            )

    except Exception as erro:

        registrar(
            f"Erro ao verificar base trimestral: "
            f"{erro}"
        )


# ============================================================
# VERIFICAÇÃO DA PREVISÃO
# ============================================================

def verificar_previsao():

    separador()
    registrar("PREVISÃO MAIS RECENTE")
    separador()

    arquivo_previsao = (
        BASE_DIR
        / "outputs_modelo4_final"
        / "modelo4D_previsao.csv"
    )

    arquivo_probabilidades = (
        BASE_DIR
        / "outputs_modelo4_final"
        / "modelo4D_probabilidades.csv"
    )

    if not arquivo_previsao.exists():

        registrar(
            "Arquivo modelo4D_previsao.csv "
            "não encontrado."
        )

        return

    try:

        df = pd.read_csv(
            arquivo_previsao
        )

        if df.empty:

            registrar(
                "Arquivo de previsão está vazio."
            )

            return

        ultima = df.iloc[-1]

        status = obter_status_trimestre()

        registrar(
            f"Regime atual: "
            f"{ultima['periodo_atual']} — "
            f"{ultima['regime_atual_economico']}"
        )

        # ----------------------------------------------------
        # Status temporal
        # ----------------------------------------------------

        if status is not None:

            registrar(
                f"Status da informação: "
                f"{status['status']}"
            )

            registrar(
                f"Último trimestre completo: "
                f"{status['ultimo_completo']}"
            )

        registrar("")

        if (
            status is not None
            and status["status"] != "COMPLETO"
        ):

            registrar(
                "TIPO DE RESULTADO: "
                "PREVISÃO OPERACIONAL COM "
                "TRIMESTRE PARCIAL"
            )

        else:

            registrar(
                "TIPO DE RESULTADO: "
                "PREVISÃO COM TRIMESTRE COMPLETO"
            )

        registrar("")

        registrar(
            f"Previsão para: "
            f"{ultima['periodo_previsto']}"
        )

        registrar("")

        registrar(
            f"Persistência: "
            f"{ultima['persistencia_economico']}"
        )

        registrar(
            f"Regressão Logística: "
            f"{ultima['logistica_economico']}"
        )

        registrar(
            f"Rede Neural MLP: "
            f"{ultima['neural_economico']}"
        )

        consenso = ultima["consenso"]

        if isinstance(consenso, str):

            consenso = (
                consenso
                .strip()
                .lower()
                == "true"
            )

        registrar("")

        if consenso:

            registrar(
                f"CONSENSO: SIM — "
                f"{ultima['neural_economico']}"
            )

        else:

            registrar(
                "CONSENSO: NÃO"
            )

        # ----------------------------------------------------
        # Probabilidades
        # ----------------------------------------------------

        if arquivo_probabilidades.exists():

            probs = pd.read_csv(
                arquivo_probabilidades
            )

            if not probs.empty:

                registrar("")
                registrar("PROBABILIDADES:")
                registrar("")

                for _, linha in probs.iterrows():

                    regime = (
                        linha[
                            "regime_economico"
                        ]
                    )

                    prob_log = (
                        linha[
                            "prob_logistica_pct"
                        ]
                    )

                    prob_neural = (
                        linha[
                            "prob_neural_pct"
                        ]
                    )

                    registrar(
                        f"{regime}: "
                        f"Logística = "
                        f"{prob_log:.2f}% | "
                        f"Neural = "
                        f"{prob_neural:.2f}%"
                    )

        registrar("")

        if (
            status is not None
            and status["status"] != "COMPLETO"
        ):

            registrar(
                f"Resultado prospectivo operacional: "
                f"{ultima['periodo_atual']} → "
                f"{ultima['periodo_previsto']}"
            )

            registrar(
                "Observação: o período de origem "
                "ainda está em andamento."
            )

        else:

            registrar(
                f"Resultado prospectivo: "
                f"{ultima['periodo_atual']} → "
                f"{ultima['periodo_previsto']}"
            )

    except Exception as erro:

        registrar(
            f"Erro ao verificar previsão: "
            f"{erro}"
        )


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():

    inicio_pipeline = datetime.now()

    separador()
    registrar(
        "PIPELINE — FRAMEWORK HÍBRIDO MARKOV-NEURAL"
    )
    separador()

    registrar(
        f"Projeto: {BASE_DIR}"
    )

    registrar(
        f"Python: {sys.executable}"
    )

    registrar(
        "Execução iniciada em: "
        + inicio_pipeline.strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    )

    registrar("")

    # Verificação inicial
    verificar_periodo_bcb()

    total = len(ETAPAS)
    processadas = 0

    for numero, etapa in enumerate(
        ETAPAS,
        start=1
    ):

        registrar("")
        registrar(
            f"ETAPA {numero}/{total}"
        )
        registrar("")

        sucesso = executar_script(
            etapa
        )

        if not sucesso:

            registrar("")

            separador()

            registrar(
                "PIPELINE INTERROMPIDA"
            )

            separador()

            registrar(
                f"Etapa problemática: "
                f"{etapa['nome']}"
            )

            registrar(
                f"Log: {ARQUIVO_LOG}"
            )

            raise SystemExit(1)

        processadas += 1

    registrar("")
    verificar_bases()

    registrar("")
    verificar_periodo_bcb()

    registrar("")
    verificar_base_trimestral()

    registrar("")
    verificar_previsao()

    fim_pipeline = datetime.now()

    registrar("")

    separador()

    registrar(
        "PIPELINE FINALIZADA"
    )

    separador()

    registrar(
        f"Etapas processadas: "
        f"{processadas}/{total}"
    )

    registrar(
        f"Duração total: "
        f"{fim_pipeline - inicio_pipeline}"
    )

    registrar(
        f"Log completo: "
        f"{ARQUIVO_LOG}"
    )

    separador()


if __name__ == "__main__":
    main()