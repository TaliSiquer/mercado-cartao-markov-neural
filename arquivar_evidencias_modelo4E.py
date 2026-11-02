from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import uuid
import os
import sys

BASE = Path(__file__).resolve().parent

PASTA_EVIDENCIAS = BASE / "evidencias_modelo4E"

MANIFESTO = (
    BASE
    / "outputs_modelo4_final"
    / "modelo4E_manifesto_execucao.json"
)

ARQUIVOS = {
    "hash_codigo": (
        BASE / "modelo4E_metricas.py",
        "codigo_modelo.py",
    ),
    "hash_dados": (
        BASE / "dataset_model_ready.csv",
        "dataset_entrada.csv",
    ),
    "hash_snapshot": (
        BASE
        / "outputs_modelo4_final"
        / "modelo4E_previsao_metricas.csv",
        "previsoes.csv",
    ),
}


def sha256(caminho):
    digest = hashlib.sha256()

    with caminho.open("rb") as entrada:
        for bloco in iter(
            lambda: entrada.read(1024 * 1024),
            b""
        ):
            digest.update(bloco)

    return digest.hexdigest()


def validar_arquivos(manifesto, pasta=None):
    for campo, (origem, nome) in ARQUIVOS.items():
        caminho = pasta / nome if pasta else origem

        if not caminho.is_file():
            raise FileNotFoundError(
                f"Arquivo ausente: {caminho}"
            )

        esperado = str(manifesto[campo])
        calculado = sha256(caminho)

        if calculado != esperado:
            raise ValueError(
                f"Hash incompatível: {campo} — {caminho}"
            )


def validar_pasta_existente(destino, manifesto):
    manifesto_arquivado = destino / "manifesto.json"

    if not manifesto_arquivado.is_file():
        raise ValueError(
            "Pasta existente sem manifesto. "
            "Arquivamento bloqueado."
        )

    if sha256(manifesto_arquivado) != sha256(MANIFESTO):
        raise ValueError(
            "Manifesto existente diferente do atual. "
            "Arquivamento bloqueado."
        )

    validar_arquivos(manifesto, destino)

    print("Evidências já arquivadas e íntegras.")
    print("Nenhum arquivo foi sobrescrito.")


def arquivar():
    if not MANIFESTO.is_file():
        raise FileNotFoundError(
            "Manifesto do Modelo 4E não encontrado."
        )

    manifesto = json.loads(
        MANIFESTO.read_text(encoding="utf-8")
    )

    obrigatorios = [
        "id_execucao",
        "hash_codigo",
        "hash_dados",
        "hash_snapshot",
    ]

    for campo in obrigatorios:
        if campo not in manifesto:
            raise ValueError(
                f"Campo obrigatório ausente: {campo}"
            )

    id_execucao = str(
        uuid.UUID(str(manifesto["id_execucao"]))
    )

    destino = PASTA_EVIDENCIAS / id_execucao

    # Reconhecer evidências anteriores sem sobrescrever.
    if destino.exists():
        if not destino.is_dir():
            raise ValueError(
                "Destino existente não é uma pasta."
            )

        validar_pasta_existente(
            destino,
            manifesto
        )
        return

    # Verificar origens antes de iniciar o arquivamento.
    validar_arquivos(manifesto)

    PASTA_EVIDENCIAS.mkdir(
        parents=True,
        exist_ok=True
    )

    temporaria = Path(
        tempfile.mkdtemp(
            prefix=f".{id_execucao}_",
            dir=PASTA_EVIDENCIAS,
        )
    )

    try:
        for campo, (origem, nome) in ARQUIVOS.items():
            shutil.copy2(
                origem,
                temporaria / nome
            )

        shutil.copy2(
            MANIFESTO,
            temporaria / "manifesto.json"
        )

        # Verificar o manifesto copiado.
        if sha256(
            temporaria / "manifesto.json"
        ) != sha256(MANIFESTO):
            raise ValueError(
                "Manifesto alterado durante a cópia."
            )

        # Verificar novamente as três evidências.
        validar_arquivos(
            manifesto,
            temporaria
        )

        # Publicar a pasta completa em uma única operação.
        # O diretório temporário está no mesmo sistema de arquivos.
        if destino.exists():
            raise FileExistsError(
                f"Destino já existe: {destino}"
            )

        os.rename(
            temporaria,
            destino
        )

        print("ARQUIVAMENTO CONCLUÍDO")
        print("Execução:", id_execucao)
        print("Destino:", destino)
        print("Arquivos arquivados: 4")

    finally:
        if temporaria.exists():
            shutil.rmtree(temporaria)


if __name__ == "__main__":
    import fcntl

    try:
        # Trava exclusiva para arquivadores deste projeto.
        # O arquivo de trava não contém evidências científicas.
        trava = BASE / ".arquivar_evidencias_modelo4E.lock"

        with trava.open("a+b") as controle:
            fcntl.flock(
                controle.fileno(),
                fcntl.LOCK_EX
            )

            try:
                arquivar()
            finally:
                fcntl.flock(
                    controle.fileno(),
                    fcntl.LOCK_UN
                )

    except Exception as erro:
        print(
            f"ARQUIVAMENTO BLOQUEADO: {erro}",
            file=sys.stderr
        )
        sys.exit(1)
