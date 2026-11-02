import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def carregar_csv(caminho):
    arquivo = ROOT / caminho

    if not arquivo.exists():
        raise FileNotFoundError(arquivo)

    return pd.read_csv(arquivo)


def normalizar_mes(df, coluna="date"):
    df = df.copy()

    df[coluna] = (
        pd.to_datetime(df[coluna], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    return df


def carregar_institucionais():

    shares = carregar_csv(
        "outputs_competicao_auditada/"
        "shares_institucionais_dashboard.csv"
    )

    shares = normalizar_mes(shares)

    colunas_shares = [
        coluna for coluna in shares.columns
        if coluna.startswith("share_")
    ]

    shares = shares[
        ["date"] + colunas_shares + ["origem_serie"]
    ].copy()

    if shares["date"].duplicated().any():
        raise ValueError(
            "Datas duplicadas na base institucional."
        )

    return shares


def carregar_entrada_saida():

    df = carregar_csv(
        "DataSet/entrada_saida_instituicoes.csv"
    )

    df["date"] = pd.to_datetime(
        df["AnoMes"].astype(str),
        format="%Y%m",
        errors="coerce"
    )

    df = normalizar_mes(df)

    df = df[
        ["date", "entrada_inst", "saida_inst"]
    ].copy()

    if df["date"].duplicated().any():
        raise ValueError(
            "Datas duplicadas na base de entrada e saída."
        )

    return df


def main():

    bcb = normalizar_mes(
        carregar_csv(
            "bcb_creditcard_competition_dataset_v4.csv"
        )
    )

    ibge = normalizar_mes(
        carregar_csv("ibge_macro_dataset.csv")
    )

    shares = carregar_institucionais()
    entrada_saida = carregar_entrada_saida()

    if bcb["date"].duplicated().any():
        raise ValueError("Datas duplicadas no BCB.")

    if ibge["date"].duplicated().any():
        raise ValueError("Datas duplicadas no IBGE.")

    df = bcb.merge(
        ibge,
        on="date",
        how="left",
        validate="one_to_one"
    )

    df = df.merge(
        shares,
        on="date",
        how="left",
        validate="one_to_one"
    )

    df = df.merge(
        entrada_saida,
        on="date",
        how="left",
        validate="one_to_one"
    )

    duplicadas = [
        coluna for coluna in df.columns
        if coluna.endswith(("_x", "_y"))
    ]

    if duplicadas:
        raise ValueError(
            f"Colunas duplicadas: {duplicadas}"
        )

    destino = ROOT / "dataset_final_dissertacao.csv"

    df.to_csv(
        destino,
        index=False,
        encoding="utf-8-sig"
    )

    print("Dataset final:", df.shape)
    print("Participações institucionais integradas.")
    print("Entrada e saída de instituições integradas.")
    print("Arquivo salvo:", destino)


if __name__ == "__main__":
    main()
