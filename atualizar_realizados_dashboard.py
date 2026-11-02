import shutil
from datetime import datetime
from pathlib import Path
import argparse
import os
import pandas as pd

BASE = Path("Mercado_de_Cartao/data")
HISTORICO = BASE / "historico_previsoes.csv"
MENSAL = BASE / "metricas_master_mensal.csv"

METRICAS_MENSAIS = {
    "juros_cartao_pf_aa_pct",
    "inadimplencia_90d_pct",
}

def atualizar(dry_run=False):
    historico = pd.read_csv(HISTORICO)
    mensal = pd.read_csv(MENSAL)

    mensal["date"] = pd.to_datetime(
        mensal["date"], errors="coerce"
    )

    mensal["quarter"] = (
        mensal["date"].dt.to_period("Q").astype(str)
    )

    mensal["month"] = (
        mensal["date"].dt.to_period("M").astype(str)
    )

    novos_realizados = 0

    for i, linha in historico.iterrows():
        metrica = str(linha["metrica"])
        periodo = str(linha["periodo_previsto"])

        if metrica not in METRICAS_MENSAIS:
            continue

        if pd.notna(linha["valor_realizado"]):
            print(f"PRESERVADO: {metrica} | {periodo}")
            continue

        dados = mensal.loc[
            mensal["quarter"] == periodo,
            ["month", metrica]
        ].dropna()

        if dados["month"].duplicated().any():
            print(
                f"BLOQUEADO — MESES DUPLICADOS: "
                f"{metrica} | {periodo}"
            )
            continue

        meses_esperados = {
            str(m)
            for m in pd.period_range(
                pd.Period(periodo, freq="Q").start_time,
                periods=3,
                freq="M"
            )
        }

        meses_encontrados = set(dados["month"])

        if meses_encontrados != meses_esperados:
            print(
                f"AGUARDANDO: {metrica} | {periodo} | "
                f"{len(meses_encontrados)}/3 meses"
            )
            continue

        realizado = float(dados[metrica].mean())
        previsto = float(linha["valor_previsto"])

        historico.at[i, "valor_realizado"] = realizado
        historico.at[i, "erro"] = abs(realizado - previsto)

        novos_realizados += 1

        print(
            f"ATUALIZADO: {metrica} | {periodo} | "
            f"Previsto={previsto:.4f} | "
            f"Realizado={realizado:.4f}"
        )

    if dry_run:
        print("\nSIMULAÇÃO: nenhum arquivo foi modificado.")
    elif novos_realizados:
        backup = HISTORICO.with_name(
            HISTORICO.stem
            + "_backup_antes_realizados_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + HISTORICO.suffix
        )
        shutil.copy2(HISTORICO, backup)

        temporario = HISTORICO.with_suffix(".tmp")
        historico.to_csv(temporario, index=False)
        os.replace(temporario, HISTORICO)
        print("Backup:", backup)
        print("\nHistórico atualizado com segurança.")
    else:
        print("\nNenhum realizado novo disponível.")

    print("Novos realizados:", novos_realizados)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true"
    )
    args = parser.parse_args()
    atualizar(dry_run=args.dry_run)
