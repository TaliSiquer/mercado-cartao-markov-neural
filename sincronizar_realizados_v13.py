from pathlib import Path
from datetime import datetime, timezone
import argparse
import os
import shutil
import tempfile

import pandas as pd


BASE = Path(__file__).resolve().parent

DATA = BASE / "Mercado_de_Cartao" / "data"

HISTORICO = DATA / "historico_previsoes_v13.csv"

MENSAL = DATA / "metricas_master_mensal.csv"

BACKUPS = DATA / "backups_previsoes"

METRICAS_MENSAIS = {
    "juros_cartao_pf_aa_pct",
    "inadimplencia_90d_pct",
}

METRICAS_IFDATA = {
    "hhi_credito_0_1",
    "cr5_credito",
}


def executar(dry_run=False):

    print("=" * 75)
    print("SINCRONIZAÇÃO DE REALIZADOS — HISTÓRICO V13")
    print("=" * 75)

    for arquivo in [HISTORICO, MENSAL]:
        if not arquivo.is_file():
            raise FileNotFoundError(arquivo)

    historico = pd.read_csv(HISTORICO)

    mensal = pd.read_csv(MENSAL)

    obrigatorias_historico = {
        "metrica",
        "periodo_previsto",
        "valor_previsto",
        "valor_realizado",
    }

    obrigatorias_mensal = {
        "date",
        *METRICAS_MENSAIS,
    }

    faltantes_h = (
        obrigatorias_historico - set(historico.columns)
    )

    faltantes_m = (
        obrigatorias_mensal - set(mensal.columns)
    )

    if faltantes_h or faltantes_m:
        raise RuntimeError(
            f"Colunas ausentes: histórico={faltantes_h}, "
            f"mensal={faltantes_m}"
        )

    mensal["date"] = pd.to_datetime(
        mensal["date"],
        errors="coerce"
    )

    mensal["mes"] = (
        mensal["date"].dt.to_period("M")
    )

    mensal["trimestre"] = (
        mensal["date"].dt.to_period("Q")
    )

    atualizados = 0

    for indice, linha in historico.iterrows():

        metrica = str(linha["metrica"])

        periodo = str(linha["periodo_previsto"])

        existente = pd.to_numeric(
            linha["valor_realizado"],
            errors="coerce"
        )

        if pd.notna(existente):
            print(
                f"PRESERVADO: {metrica} | {periodo}"
            )
            continue

        if metrica in METRICAS_IFDATA:
            print(
                f"AGUARDANDO AUDITORIA IFDATA: "
                f"{metrica} | {periodo}"
            )
            continue

        if metrica not in METRICAS_MENSAIS:
            print(
                f"MÉTRICA NÃO CONFIGURADA: "
                f"{metrica}"
            )
            continue

        try:
            trimestre = pd.Period(
                periodo,
                freq="Q"
            )
        except Exception:
            print(
                f"PERÍODO INVÁLIDO: {periodo}"
            )
            continue

        meses_esperados = set(
            pd.period_range(
                trimestre.start_time,
                periods=3,
                freq="M"
            )
        )

        dados = mensal.loc[
            mensal["trimestre"] == trimestre,
            ["mes", metrica]
        ].copy()

        dados[metrica] = pd.to_numeric(
            dados[metrica],
            errors="coerce"
        )

        dados = dados.dropna(
            subset=["mes", metrica]
        )

        # Duplicidade mensal impede conciliação
        # automática, mesmo quando os valores coincidem.
        if dados["mes"].duplicated().any():
            print(
                f"BLOQUEADO — MESES DUPLICADOS: "
                f"{metrica} | {periodo}"
            )
            continue

        meses_encontrados = set(
            dados["mes"]
        )

        if meses_encontrados != meses_esperados:
            print(
                f"AGUARDANDO: {metrica} | "
                f"{periodo} | "
                f"{len(meses_encontrados)}/3 meses"
            )
            continue

        realizado = float(
            dados[metrica].mean()
        )

        historico.at[
            indice,
            "valor_realizado"
        ] = realizado

        if "fonte_realizado" in historico.columns:
            historico.at[
                indice,
                "fonte_realizado"
            ] = (
                "metricas_master_mensal.csv | "
                "média dos três meses do trimestre"
            )

        if "data_conferencia" in historico.columns:
            historico.at[
                indice,
                "data_conferencia"
            ] = datetime.now(
                timezone.utc
            ).isoformat()


        if "observacoes" in historico.columns:
            anterior = linha.get(
                "observacoes",
                ""
            )

            anterior = (
                str(anterior)
                if pd.notna(anterior)
                else ""
            )

            complemento = (
                "Realizado conciliado pela média "
                "de três observações mensais. "
                "Data oficial de publicação não auditada."
            )

            historico.at[
                indice,
                "observacoes"
            ] = (
                anterior + " | " + complemento
            ).strip(" |")

        atualizados += 1

        print(
            f"REALIZADO IDENTIFICADO: "
            f"{metrica} | {periodo} | "
            f"{realizado:.6f}"
        )

    print("\nNovos realizados:", atualizados)

    if dry_run:
        print(
            "MODO SIMULAÇÃO: nenhum arquivo modificado."
        )
        return

    if atualizados == 0:
        print(
            "Histórico preservado: "
            "nenhum realizado novo disponível."
        )
        return

    BACKUPS.mkdir(
        parents=True,
        exist_ok=True
    )

    momento = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    backup = BACKUPS / (
        f"historico_v13_antes_realizados_{momento}.csv"
    )

    shutil.copy2(
        HISTORICO,
        backup
    )

    fd, temporario = tempfile.mkstemp(
        prefix=".realizados_v13_",
        suffix=".csv",
        dir=DATA
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8"
        ) as arquivo:
            historico.to_csv(
                arquivo,
                index=False
            )

        conferencia = pd.read_csv(
            temporario
        )

        if len(conferencia) != len(historico):
            raise RuntimeError(
                "Quantidade de registros alterada."
            )

        os.replace(
            temporario,
            HISTORICO
        )

    finally:
        if os.path.exists(temporario):
            os.unlink(temporario)

    print("\nBackup:", backup)
    print("Histórico atualizado:", HISTORICO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    argumentos = parser.parse_args()

    executar(
        dry_run=argumentos.dry_run
    )
