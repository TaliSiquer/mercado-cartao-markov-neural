from pathlib import Path
from datetime import datetime
import os
import pandas as pd

BASE = Path("Mercado_de_Cartao/data")
HIST = BASE / "historico_previsoes.csv"
FONTE = Path("outputs_modelo4_final/modelo4E_previsao_metricas.csv")
REALIZADOS = BASE / "metricas_master_trimestral.csv"

METRICAS_AUTORIZADAS = {
    "juros_cartao_pf_aa_pct",
    "inadimplencia_90d_pct",
}

def main():
    if not all(p.exists() for p in (HIST, FONTE, REALIZADOS)):
        raise FileNotFoundError("Um dos arquivos necessários não existe.")

    historico = pd.read_csv(HIST)
    previsoes = pd.read_csv(FONTE)
    realizados = pd.read_csv(REALIZADOS)

    realizados["date"] = pd.to_datetime(realizados["date"])
    realizados["quarter"] = realizados["date"].dt.to_period("Q").astype(str)

    data_arquivo = datetime.fromtimestamp(FONTE.stat().st_mtime)
    novos = []

    for _, p in previsoes.iterrows():
        metrica = str(p["metrica"])

        if metrica not in METRICAS_AUTORIZADAS:
            print(f"AGUARDANDO AUDITORIA: {metrica}")
            continue

        periodo = str(p["periodo_previsto"])
        origem = str(p["periodo_atual"])
        periodo_fonte = str(p["periodo_ultimo_dado"])

        # A data do arquivo deve anteceder o fim do trimestre previsto.
        fim_periodo = pd.Period(periodo, freq="Q").end_time
        if data_arquivo > fim_periodo:
            print(f"IGNORADO: previsão posterior ao período {periodo}")
            continue

        # Evita inserir duas vezes a mesma previsão recuperada.
        duplicado = (
            (historico["metrica"].astype(str) == metrica)
            & (historico["periodo_previsto"].astype(str) == periodo)
            & (
                historico["tipo_previsao"].astype(str)
                == "historica_recuperada"
            )
        ).any()

        if duplicado:
            print(f"JÁ REGISTRADO: {metrica} | {periodo}")
            continue

        valor_realizado = float("nan")

        observacoes = realizados.loc[
            realizados["quarter"] == periodo, metrica
        ].dropna()

        if not observacoes.empty:
            valor_realizado = float(observacoes.iloc[-1])

        previsto = float(p["previsao_selecionada"])

        erro = (
            abs(previsto - valor_realizado)
            if pd.notna(valor_realizado)
            else float("nan")
        )

        novos.append({
            "data_execucao": data_arquivo.isoformat(timespec="seconds"),
            "origem_previsao": origem,
            "periodo_previsto": periodo,
            "tipo_previsao": "historica_recuperada",
            "metrica": metrica,
            "valor_previsto": previsto,
            "modelo": str(p["modelo_selecionado"]),
            "ultimo_valor_disponivel": float(p["valor_atual"]),
            "periodo_fonte": periodo_fonte,
            "valor_realizado": valor_realizado,
            "erro": erro,
        })

    if novos:
        historico = pd.concat(
            [historico, pd.DataFrame(novos)],
            ignore_index=True
        )

        # Gravação atômica: evita histórico parcialmente escrito.
        temporario = HIST.with_suffix(".tmp")
        historico.to_csv(temporario, index=False)
        os.replace(temporario, HIST)

    print("\nINTEGRAÇÃO CONCLUÍDA")
    print("Novos registros:", len(novos))
    print("Total no histórico:", len(historico))

    colunas = [
        "periodo_previsto",
        "metrica",
        "valor_previsto",
        "valor_realizado",
        "modelo",
        "tipo_previsao",
    ]

    print("\nHISTÓRICO:")
    print(historico[colunas].tail(10).to_string(index=False))

    print(
        "\nATENÇÃO: data_execucao foi recuperada da modificação "
        "do arquivo e não comprova, isoladamente, a data original "
        "da previsão."
    )

if __name__ == "__main__":
    main()
