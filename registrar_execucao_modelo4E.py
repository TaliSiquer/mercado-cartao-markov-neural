from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import fcntl
import tempfile
import shutil
import uuid

import pandas as pd


BASE = Path(__file__).resolve().parent
PASTA_EVIDENCIAS = BASE / "evidencias_modelo4E"

SNAPSHOT = (
    BASE / "outputs_modelo4_final"
    / "modelo4E_previsao_metricas.csv"
)

MANIFESTO = (
    BASE / "outputs_modelo4_final"
    / "modelo4E_manifesto_execucao.json"
)

MODELO = BASE / "modelo4E_metricas.py"
DADOS = BASE / "dataset_model_ready.csv"

REGISTRO = (
    BASE / "Mercado_de_Cartao"
    / "data"
    / "registro_execucoes_modelo4E.csv"
)

COLUNAS = [
    "id_execucao",
    "data_execucao",
    "metrica",
    "periodo_ultimo_dado",
    "periodo_previsto",
    "modelo_selecionado",
    "valor_previsto",
    "tipo_previsao",
    "hash_codigo",
    "hash_dados",
    "hash_snapshot",
]

CHAVE = [
    "id_execucao",
    "metrica",
    "periodo_previsto",
]


def sha256(caminho):
    digest = hashlib.sha256()

    with open(caminho, "rb") as entrada:
        for bloco in iter(
            lambda: entrada.read(1024 * 1024),
            b""
        ):
            digest.update(bloco)

    return digest.hexdigest()


def validar_manifesto():
    if not MANIFESTO.is_file():
        raise FileNotFoundError(
            "Manifesto de trabalho inexistente."
        )

    with MANIFESTO.open("r", encoding="utf-8") as entrada:
        manifesto_trabalho = json.load(entrada)

    obrigatorios = [
        "id_execucao",
        "data_execucao_utc",
        "hash_codigo",
        "hash_dados",
        "hash_snapshot",
        "quantidade_previsoes",
        "periodos_previstos",
    ]

    faltantes = [
        campo for campo in obrigatorios
        if campo not in manifesto_trabalho
    ]

    if faltantes:
        raise ValueError(
            f"Manifesto incompleto: {faltantes}"
        )

    id_execucao = str(
        uuid.UUID(str(manifesto_trabalho["id_execucao"]))
    )

    pasta = PASTA_EVIDENCIAS / id_execucao

    if not pasta.is_dir():
        raise FileNotFoundError(
            f"Evidências arquivadas não encontradas: {pasta}"
        )

    manifesto_arquivado = pasta / "manifesto.json"

    if not manifesto_arquivado.is_file():
        raise FileNotFoundError(
            "Manifesto arquivado inexistente."
        )

    if sha256(MANIFESTO) != sha256(manifesto_arquivado):
        raise ValueError(
            "Manifesto de trabalho diferente do arquivado. "
            "Registro bloqueado."
        )

    with manifesto_arquivado.open(
        "r", encoding="utf-8"
    ) as entrada:
        manifesto = json.load(entrada)

    data = datetime.fromisoformat(
        str(manifesto["data_execucao_utc"])
    )

    if data.tzinfo is None:
        raise ValueError(
            "Data do manifesto precisa conter fuso horário."
        )

    if data.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise ValueError(
            "Data do manifesto está no futuro."
        )

    arquivos = {
        "hash_codigo": pasta / "codigo_modelo.py",
        "hash_dados": pasta / "dataset_entrada.csv",
        "hash_snapshot": pasta / "previsoes.csv",
    }

    for campo, caminho in arquivos.items():
        if not caminho.is_file():
            raise FileNotFoundError(caminho)

        if sha256(caminho) != str(manifesto[campo]):
            raise ValueError(
                f"Hash incompatível: {campo}. "
                "Registro bloqueado."
            )

    return manifesto, arquivos["hash_snapshot"]


def preparar_registros():
    manifesto, snapshot_arquivado = validar_manifesto()

    df = pd.read_csv(snapshot_arquivado)

    # Compatibilidade com o snapshot original do Modelo 4E.
    # A normalização ocorre somente em memória.
    if "previsao_selecionada" in df.columns:
        if "valor_previsto" in df.columns:
            raise ValueError(
                "Snapshot ambíguo: contém previsao_selecionada "
                "e valor_previsto."
            )
        df = df.rename(
            columns={"previsao_selecionada": "valor_previsto"}
        )


    obrigatorias = [
        "metrica",
        "periodo_ultimo_dado",
        "periodo_previsto",
        "modelo_selecionado",
        "valor_previsto",
    ]

    faltantes = [
        campo for campo in obrigatorias
        if campo not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Snapshot incompleto: {faltantes}"
        )

    if len(df) != int(
        manifesto["quantidade_previsoes"]
    ):
        raise ValueError(
            "Quantidade de previsões diferente do manifesto."
        )

    periodos = sorted(
        df["periodo_previsto"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if periodos != sorted(
        str(p) for p in manifesto["periodos_previstos"]
    ):
        raise ValueError(
            "Períodos previstos incompatíveis com o manifesto."
        )

    if df[obrigatorias].isna().any().any():
        raise ValueError(
            "Snapshot contém campos obrigatórios vazios."
        )

    df["valor_previsto"] = pd.to_numeric(
        df["valor_previsto"],
        errors="raise"
    )

    if not pd.Series(
        df["valor_previsto"]
    ).map(
        lambda valor: pd.notna(valor)
        and abs(valor) != float("inf")
    ).all():
        raise ValueError(
            "Snapshot contém valores previstos inválidos."
        )

    data_execucao = datetime.fromisoformat(
        manifesto["data_execucao_utc"]
    )

    trimestre_execucao = pd.Period(
        data_execucao.date(),
        freq="Q"
    )

    registros = []

    for _, linha in df.iterrows():
        periodo = pd.Period(
            str(linha["periodo_previsto"]),
            freq="Q"
        )

        tipo = (
            "prospectiva_pendente_auditoria"
            if periodo > trimestre_execucao
            else "reconstrucao_historica"
        )

        registros.append({
            "id_execucao": manifesto["id_execucao"],
            "data_execucao": manifesto["data_execucao_utc"],
            "metrica": linha["metrica"],
            "periodo_ultimo_dado": linha["periodo_ultimo_dado"],
            "periodo_previsto": linha["periodo_previsto"],
            "modelo_selecionado": linha["modelo_selecionado"],
            "valor_previsto": linha["valor_previsto"],
            "tipo_previsao": tipo,
            "hash_codigo": manifesto["hash_codigo"],
            "hash_dados": manifesto["hash_dados"],
            "hash_snapshot": manifesto["hash_snapshot"],
        })

    novos = pd.DataFrame(
        registros,
        columns=COLUNAS
    )

    if novos.duplicated(CHAVE).any():
        raise ValueError(
            "Snapshot contém previsões duplicadas "
            "para a mesma métrica e período."
        )

    return novos


def gravar_registros(novos):
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    caminho_lock = REGISTRO.parent / '.registro_modelo4E.lock'

    with caminho_lock.open('a+b') as arquivo_lock:
        fcntl.flock(arquivo_lock.fileno(), fcntl.LOCK_EX)
        try:
            return _gravar_registros_sem_lock(novos)
        finally:
            fcntl.flock(arquivo_lock.fileno(), fcntl.LOCK_UN)


def _gravar_registros_sem_lock(novos):
    REGISTRO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if REGISTRO.exists():
        historico = pd.read_csv(REGISTRO)

        faltantes = [
            coluna for coluna in COLUNAS
            if coluna not in historico.columns
        ]

        if faltantes:
            raise ValueError(
                "Registro existente incompatível: "
                f"{faltantes}. Migração necessária."
            )

        if historico.duplicated(CHAVE).any():
            raise ValueError(
                "Duplicidades existentes no histórico. "
                "Gravação bloqueada."
            )

        existentes = set(
            map(
                tuple,
                historico[CHAVE].astype(str).to_numpy()
            )
        )

        mascara = [
            tuple(valores) not in existentes
            for valores in novos[CHAVE]
            .astype(str)
            .to_numpy()
        ]

        pendentes = novos.loc[mascara].copy()

        if pendentes.empty:
            print(
                "Todas as previsões desta execução "
                "já estão registradas."
            )
            return

        resultado = pd.concat(
            [historico, pendentes],
            ignore_index=True
        )

    else:
        pendentes = novos.copy()
        resultado = pendentes.copy()

    if resultado.duplicated(CHAVE).any():
        raise ValueError(
            "Duplicidade detectada antes da gravação."
        )

    pasta_backup = (
        REGISTRO.parent / "backups_previsoes"
    )

    pasta_backup.mkdir(
        parents=True,
        exist_ok=True
    )

    if REGISTRO.exists():
        nome_backup = (
            "registro_execucoes_modelo4E_"
            + datetime.now(timezone.utc).strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            + ".csv"
        )

        backup = pasta_backup / nome_backup

        shutil.copy2(
            REGISTRO,
            backup
        )

        print("Backup:", backup)

    fd, temporario = tempfile.mkstemp(
        prefix="registro_modelo4E_",
        suffix=".csv.tmp",
        dir=str(REGISTRO.parent)
    )

    try:
        with os.fdopen(
            fd, "w", encoding="utf-8"
        ) as saida:
            resultado.to_csv(
                saida,
                index=False
            )

            saida.flush()
            os.fsync(saida.fileno())

        os.replace(
            temporario,
            REGISTRO
        )

    finally:
        if os.path.exists(temporario):
            os.unlink(temporario)

    print("Registro atualizado:", REGISTRO)
    print("Novas previsões:", len(pendentes))
    print("Total de registros:", len(resultado))


def main():
    parser = argparse.ArgumentParser()

    grupo = parser.add_mutually_exclusive_group(
        required=True
    )

    grupo.add_argument(
        "--dry-run",
        action="store_true"
    )

    grupo.add_argument(
        "--registrar",
        action="store_true"
    )

    args = parser.parse_args()

    novos = preparar_registros()

    print("=" * 72)
    print("REGISTRADOR AUDITÁVEL — MODELO 4E")
    print("=" * 72)

    print(
        novos[
            [
                "metrica",
                "periodo_previsto",
                "modelo_selecionado",
                "valor_previsto",
                "tipo_previsao",
            ]
        ].to_string(index=False)
    )

    if args.dry_run:
        print(
            "\\nSIMULAÇÃO: nenhuma gravação realizada."
        )
        return

    gravar_registros(novos)


if __name__ == "__main__":
    main()
