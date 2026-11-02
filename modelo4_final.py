# =============================================================================
# MODELO 4D — PREVISÃO PROSPECTIVA DO PRÓXIMO REGIME
# Framework Híbrido Markov-Neural
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

ARQUIVO = "outputs_modelo3C/modelo3C_base_para_neural.csv"

PASTA_SAIDA = "outputs_modelo4_final"

ALVO = "regime_proximo"

os.makedirs(PASTA_SAIDA, exist_ok=True)


# =============================================================================
# MAPEAMENTO ECONÔMICO DOS REGIMES
# =============================================================================
#
# IMPORTANTE:
#
# A numeração dos estados do Markov é estatística e pode mudar
# entre execuções.
#
# Por isso, NÃO utilizamos mais um mapa fixo como:
#
# 0 = INTERMEDIÁRIO
# 1 = ALTO
# 2 = BAIXO
#
# O significado econômico será construído automaticamente
# após a leitura da base:
#
# menor média de juros  -> BAIXO
# média intermediária   -> INTERMEDIÁRIO
# maior média de juros  -> ALTO
# =============================================================================

MAPA_REGIMES = {}


# Mesmas features usadas no Modelo 4 final

FEATURES_CANDIDATAS = [

    "juros_cartao_pf_aa_pct",
    "selic_am_pct",
    "inadimplencia_90d_pct",

    "hhi_credito_0_1",
    "cr5_credito",
    "share_fintechs",

    "comprometimento_renda_pct",

    "ibge_taxa_desocupacao_trimestral_pct",
    "ibge_rendimento_medio_real_brl",
]


print("=" * 100)
print("MODELO 4D — PREVISÃO PROSPECTIVA DO PRÓXIMO REGIME")
print("=" * 100)


# =============================================================================
# 2. CARREGAR BASE
# =============================================================================

if not os.path.exists(ARQUIVO):

    raise FileNotFoundError(
        f"\nArquivo não encontrado:\n{ARQUIVO}\n"
        "Execute primeiro o modelo3C.py."
    )


dados = pd.read_csv(ARQUIVO)


print("\nBASE RECEBIDA DO MODELO 3C")
print("-" * 100)

print(f"Linhas: {len(dados)}")
print(f"Colunas: {len(dados.columns)}")


# =============================================================================
# 3. ORGANIZAR TEMPO
# =============================================================================

if "date" in dados.columns:

    dados["date"] = pd.to_datetime(
        dados["date"]
    )


if "quarter" in dados.columns:

    dados["quarter"] = pd.PeriodIndex(
        dados["quarter"],
        freq="Q"
    )

    dados = (
        dados
        .sort_values("quarter")
        .reset_index(drop=True)
    )

elif "date" in dados.columns:

    dados = (
        dados
        .sort_values("date")
        .reset_index(drop=True)
    )


# =============================================================================
# 4. IDENTIFICAR FEATURES DISPONÍVEIS
# =============================================================================

features = [
    col
    for col in FEATURES_CANDIDATAS
    if col in dados.columns
]


print("\nFEATURES UTILIZADAS")
print("-" * 100)

for col in features:

    print(
        f"{col}: "
        f"{dados[col].notna().sum()} válidos"
    )


if len(features) == 0:

    raise ValueError(
        "Nenhuma feature foi encontrada."
    )


# =============================================================================
# 5. CONVERTER FEATURES PARA NUMÉRICO
# =============================================================================

for col in features:

    dados[col] = pd.to_numeric(
        dados[col],
        errors="coerce"
    )


# =============================================================================
# 6. TRATAMENTO DE DADOS AUSENTES
# =============================================================================
#
# Não realizamos forward fill nas features.
#
# Variáveis trimestrais, como HHI e CR5, permanecem ausentes nos
# períodos em que ainda não foram efetivamente observadas/publicadas.
#
# A amostra de treinamento será formada somente por observações com
# dados disponíveis para todas as features selecionadas.
# =============================================================================


# =============================================================================
# 7. CRIAR O ALVO, SE NECESSÁRIO
# =============================================================================
#
# A base completa pode ou não trazer regime_proximo.
#
# Se não trouxer, construímos:
#
# regime_proximo(t) = regime_estimado(t+1)
# =============================================================================

if "regime_estimado" not in dados.columns:

    raise ValueError(
        "A coluna 'regime_estimado' não foi encontrada."
    )


dados["regime_estimado"] = pd.to_numeric(
    dados["regime_estimado"],
    errors="coerce"
)


# =============================================================================
# 7.1 MAPEAMENTO ECONÔMICO AUTOMÁTICO DOS REGIMES
# =============================================================================
#
# Os números dos estados do Markov são apenas rótulos estatísticos.
# Eles podem mudar entre execuções.
#
# Portanto, o significado econômico é definido pela média da variável
# utilizada para identificar os regimes no Modelo 3C.
#
# menor média de juros -> BAIXO
# média intermediária  -> INTERMEDIÁRIO
# maior média de juros -> ALTO
# =============================================================================

VARIAVEL_REGIME = "juros_cartao_pf_aa_pct"


if VARIAVEL_REGIME not in dados.columns:

    raise ValueError(
        f"A variável '{VARIAVEL_REGIME}' "
        "não foi encontrada na base do Modelo 3C."
    )


dados[VARIAVEL_REGIME] = pd.to_numeric(
    dados[VARIAVEL_REGIME],
    errors="coerce"
)


medias_regimes = (
    dados
    .dropna(
        subset=[
            "regime_estimado",
            VARIAVEL_REGIME
        ]
    )
    .groupby(
        "regime_estimado"
    )[VARIAVEL_REGIME]
    .mean()
    .sort_values()
)


if len(medias_regimes) != 3:

    raise ValueError(
        "Não foi possível identificar exatamente "
        "3 regimes para o mapeamento econômico. "
        f"Regimes encontrados: {len(medias_regimes)}"
    )


regime_baixo = int(
    medias_regimes.index[0]
)

regime_intermediario = int(
    medias_regimes.index[1]
)

regime_alto = int(
    medias_regimes.index[2]
)


MAPA_REGIMES = {

    regime_baixo:
        "BAIXO",

    regime_intermediario:
        "INTERMEDIÁRIO",

    regime_alto:
        "ALTO"
}


print(
    "\nMAPEAMENTO ECONÔMICO AUTOMÁTICO DOS REGIMES"
)

print("-" * 100)


for regime, media in medias_regimes.items():

    regime = int(regime)

    print(
        f"Classe {regime}: "
        f"{MAPA_REGIMES[regime]} "
        f"| média de juros = {media:.4f}"
    )


# =============================================================================
# 7.2 CRIAR / VALIDAR ALVO
# =============================================================================

if ALVO not in dados.columns:

    dados[ALVO] = (
        dados["regime_estimado"]
        .shift(-1)
    )

else:

    dados[ALVO] = pd.to_numeric(
        dados[ALVO],
        errors="coerce"
    )


# =============================================================================
# 8. IDENTIFICAR ÚLTIMO PERÍODO
# =============================================================================

ultima_linha = dados.iloc[[-1]].copy()


if "quarter" in dados.columns:

    periodo_atual = (
        ultima_linha["quarter"].iloc[0]
    )

    periodo_previsto = (
        periodo_atual + 1
    )

else:

    periodo_atual = "último período"

    periodo_previsto = "t+1"


print("\nPERÍODO PROSPECTIVO")
print("-" * 100)

print(
    f"Informações utilizadas até: "
    f"{periodo_atual}"
)

print(
    f"Previsão para: "
    f"{periodo_previsto}"
)


# =============================================================================
# 9. BASE DE TREINAMENTO
# =============================================================================

base_treino = (
    dados
    .dropna(
        subset=features + [ALVO]
    )
    .copy()
)


base_treino[ALVO] = (
    base_treino[ALVO]
    .astype(int)
)


print("\nBASE FINAL DE TREINAMENTO")
print("-" * 100)

print(
    f"Observações: "
    f"{len(base_treino)}"
)

print(
    f"Features: "
    f"{len(features)}"
)


print("\nDistribuição do alvo:")

print(
    base_treino[ALVO]
    .value_counts()
    .sort_index()
)


# =============================================================================
# 10. CONSTRUIR VETOR PROSPECTIVO COM INFORMAÇÃO DISPONÍVEL
# =============================================================================
#
# Para a validação histórica, NÃO realizamos imputação temporal.
#
# Para a previsão prospectiva, entretanto, algumas variáveis estruturais
# trimestrais podem ainda não estar publicadas para o trimestre corrente
# (por exemplo, HHI e CR5).
#
# Nesse caso, utiliza-se exclusivamente a última observação efetivamente
# disponível até o período atual.
#
# Essa regra é aplicada SOMENTE ao vetor prospectivo e não altera a
# base histórica de treinamento.
# =============================================================================

X_futuro = ultima_linha[
    features
].copy()


fontes_features_futuro = []


for col in features:

    valor_atual = X_futuro[col].iloc[0]

    if pd.notna(valor_atual):

        periodo_fonte = periodo_atual

        valor_usado = valor_atual

        origem = "período atual"

    else:

        historico_disponivel = (
            dados.loc[
                dados[col].notna(),
                [col] + (
                    ["quarter"]
                    if "quarter" in dados.columns
                    else ["date"]
                )
            ]
            .copy()
        )

        if historico_disponivel.empty:

            raise ValueError(
                f"A feature '{col}' não possui nenhuma "
                "observação disponível na base."
            )

        ultima_disponivel = (
            historico_disponivel.iloc[-1]
        )

        valor_usado = (
            ultima_disponivel[col]
        )

        if "quarter" in historico_disponivel.columns:

            periodo_fonte = (
                ultima_disponivel["quarter"]
            )

        else:

            periodo_fonte = (
                ultima_disponivel["date"]
            )

        X_futuro.loc[
            X_futuro.index[0],
            col
        ] = valor_usado

        origem = "última observação disponível"


    fontes_features_futuro.append(
        {
            "feature": col,
            "valor_utilizado": valor_usado,
            "periodo_fonte": str(periodo_fonte),
            "origem": origem
        }
    )


fontes_features_futuro = pd.DataFrame(
    fontes_features_futuro
)


print(
    "\nINFORMAÇÕES UTILIZADAS NA PREVISÃO PROSPECTIVA"
)

print("-" * 100)

print(
    fontes_features_futuro
    .to_string(index=False)
)


if X_futuro.isna().any().any():

    faltantes = (
        X_futuro.columns[
            X_futuro.isna().any()
        ]
        .tolist()
    )

    raise ValueError(
        "Ainda há valores ausentes no vetor prospectivo: "
        f"{faltantes}"
    )


fontes_features_futuro.to_csv(
    os.path.join(
        PASTA_SAIDA,
        "modelo4D_fontes_features_previsao.csv"
    ),
    index=False
)


# =============================================================================
# 11. MATRIZES
# =============================================================================

X = base_treino[
    features
].copy()

y = base_treino[
    ALVO
].copy()



# =============================================================================
# 11B. VALIDAÇÃO HISTÓRICA WALK-FORWARD
# =============================================================================
#
# Objetivo:
# avaliar a capacidade de previsão fora da amostra respeitando a ordem
# temporal dos dados.
#
# Em cada iteração:
#   1. utiliza-se somente o passado para treinamento;
#   2. o StandardScaler é ajustado somente no conjunto de treino;
#   3. Logística e MLP são reestimadas;
#   4. prevê-se exclusivamente a observação seguinte;
#   5. compara-se a previsão com regime_proximo observado ex-post.
#
# A persistência usa regime_estimado(t) como previsão de regime(t+1).
#
# Não há embaralhamento, imputação temporal ou uso de informação futura.
# =============================================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

print("\n" + "=" * 100)
print("VALIDAÇÃO HISTÓRICA WALK-FORWARD")
print("=" * 100)

MIN_TREINO_WF = 16

base_wf = (
    base_treino
    .copy()
    .reset_index(drop=True)
)

# -------------------------------------------------------------------------
# Determinar primeiro ponto em que:
# - há pelo menos MIN_TREINO_WF observações anteriores;
# - todas as classes existentes na base completa já apareceram no treino.
# Isso evita avaliar um classificador multirregime antes de ele ter visto
# historicamente todos os regimes da amostra.
# -------------------------------------------------------------------------

classes_globais = sorted(
    base_wf[ALVO]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

inicio_wf = None

for i in range(
    MIN_TREINO_WF,
    len(base_wf)
):
    classes_treino_i = set(
        base_wf
        .iloc[:i][ALVO]
        .astype(int)
        .unique()
        .tolist()
    )

    if set(classes_globais).issubset(
        classes_treino_i
    ):
        inicio_wf = i
        break

if inicio_wf is None:
    raise ValueError(
        "Não foi possível iniciar o walk-forward com "
        "todas as classes presentes no treinamento."
    )

resultados_wf = []

for i in range(
    inicio_wf,
    len(base_wf)
):

    treino_wf = (
        base_wf
        .iloc[:i]
        .copy()
    )

    teste_wf = (
        base_wf
        .iloc[[i]]
        .copy()
    )

    X_train_wf = treino_wf[
        features
    ].copy()

    y_train_wf = treino_wf[
        ALVO
    ].astype(int).copy()

    X_test_wf = teste_wf[
        features
    ].copy()

    y_real_wf = int(
        teste_wf[ALVO].iloc[0]
    )

    # ---------------------------------------------------------------------
    # Padronização SEM vazamento:
    # scaler ajustado apenas no passado.
    # ---------------------------------------------------------------------

    scaler_wf = StandardScaler()

    X_train_scaled_wf = (
        scaler_wf
        .fit_transform(
            X_train_wf
        )
    )

    X_test_scaled_wf = (
        scaler_wf
        .transform(
            X_test_wf
        )
    )

    # ---------------------------------------------------------------------
    # Regressão Logística
    # mesmos hiperparâmetros do modelo prospectivo.
    # ---------------------------------------------------------------------

    logit_wf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )

    logit_wf.fit(
        X_train_scaled_wf,
        y_train_wf
    )

    pred_logit_wf = int(
        logit_wf
        .predict(
            X_test_scaled_wf
        )[0]
    )

    # ---------------------------------------------------------------------
    # MLP
    # mesmos hiperparâmetros do modelo prospectivo.
    # ---------------------------------------------------------------------

    mlp_wf = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation="tanh",
        solver="lbfgs",
        alpha=2.0,
        max_iter=5000,
        random_state=42
    )

    mlp_wf.fit(
        X_train_scaled_wf,
        y_train_wf
    )

    pred_mlp_wf = int(
        mlp_wf
        .predict(
            X_test_scaled_wf
        )[0]
    )

    # ---------------------------------------------------------------------
    # Persistência:
    # regime atual é utilizado como previsão do próximo.
    # ---------------------------------------------------------------------

    regime_atual_wf = int(
        teste_wf[
            "regime_estimado"
        ].iloc[0]
    )

    pred_persist_wf = (
        regime_atual_wf
    )

    mudou_regime_wf = (
        y_real_wf
        != regime_atual_wf
    )

    # Identificação temporal
    if "quarter" in teste_wf.columns:
        periodo_wf = str(
            teste_wf[
                "quarter"
            ].iloc[0]
        )

    elif "date" in teste_wf.columns:
        periodo_wf = str(
            teste_wf[
                "date"
            ].iloc[0]
        )

    else:
        periodo_wf = str(i)

    resultados_wf.append(
        {
            "periodo_origem": periodo_wf,
            "n_treino": len(treino_wf),
            "regime_atual": regime_atual_wf,
            "regime_real_proximo": y_real_wf,
            "pred_persistencia": pred_persist_wf,
            "pred_logistica": pred_logit_wf,
            "pred_mlp": pred_mlp_wf,
            "mudanca_regime": mudou_regime_wf,
        }
    )


resultados_wf = pd.DataFrame(
    resultados_wf
)

if resultados_wf.empty:
    raise ValueError(
        "A validação walk-forward não produziu previsões."
    )


# =============================================================================
# MÉTRICAS AGREGADAS
# =============================================================================

y_real_wf = (
    resultados_wf[
        "regime_real_proximo"
    ]
    .astype(int)
)

modelos_wf = {
    "Persistência": "pred_persistencia",
    "Regressão Logística": "pred_logistica",
    "Rede Neural MLP": "pred_mlp",
}

metricas_wf = []

for nome_modelo, coluna_pred in modelos_wf.items():

    y_pred_wf = (
        resultados_wf[
            coluna_pred
        ]
        .astype(int)
    )

    acc = accuracy_score(
        y_real_wf,
        y_pred_wf
    )

    bal_acc = balanced_accuracy_score(
        y_real_wf,
        y_pred_wf
    )

    macro_f1 = f1_score(
        y_real_wf,
        y_pred_wf,
        average="macro",
        zero_division=0
    )

    mask_mudanca = (
        resultados_wf[
            "mudanca_regime"
        ]
    )

    n_mudancas = int(
        mask_mudanca.sum()
    )

    if n_mudancas > 0:

        acc_mudancas = accuracy_score(
            y_real_wf[
                mask_mudanca
            ],
            y_pred_wf[
                mask_mudanca
            ]
        )

    else:

        acc_mudancas = float("nan")

    mask_permanencia = (
        ~resultados_wf[
            "mudanca_regime"
        ]
    )

    n_permanencias = int(
        mask_permanencia.sum()
    )

    if n_permanencias > 0:

        acc_permanencias = accuracy_score(
            y_real_wf[
                mask_permanencia
            ],
            y_pred_wf[
                mask_permanencia
            ]
        )

    else:

        acc_permanencias = float("nan")

    metricas_wf.append(
        {
            "modelo": nome_modelo,
            "n_previsoes": len(resultados_wf),
            "acuracia": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "n_mudancas_regime": n_mudancas,
            "acuracia_mudancas": acc_mudancas,
            "n_permanencias_regime": n_permanencias,
            "acuracia_permanencias": acc_permanencias,
        }
    )


metricas_wf = pd.DataFrame(
    metricas_wf
)


# =============================================================================
# MATRIZES DE CONFUSÃO
# =============================================================================

for nome_modelo, coluna_pred in modelos_wf.items():

    y_pred_wf = (
        resultados_wf[
            coluna_pred
        ]
        .astype(int)
    )

    matriz = confusion_matrix(
        y_real_wf,
        y_pred_wf,
        labels=classes_globais
    )

    matriz_df = pd.DataFrame(
        matriz,
        index=[
            f"real_{c}"
            for c in classes_globais
        ],
        columns=[
            f"pred_{c}"
            for c in classes_globais
        ]
    )

    nome_arquivo = (
        nome_modelo
        .lower()
        .replace(" ", "_")
        .replace("ã", "a")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("é", "e")
        .replace("ê", "e")
    )

    matriz_df.to_csv(
        os.path.join(
            PASTA_SAIDA,
            f"modelo4D_matriz_confusao_{nome_arquivo}.csv"
        )
    )


# =============================================================================
# SAÍDA DA VALIDAÇÃO
# =============================================================================

resultados_wf.to_csv(
    os.path.join(
        PASTA_SAIDA,
        "modelo4D_walkforward_previsoes.csv"
    ),
    index=False
)

metricas_wf.to_csv(
    os.path.join(
        PASTA_SAIDA,
        "modelo4D_walkforward_metricas.csv"
    ),
    index=False
)


print(
    f"\nPrimeiro teste fora da amostra: "
    f"índice {inicio_wf}"
)

print(
    f"Observações iniciais de treinamento: "
    f"{inicio_wf}"
)

print(
    f"Previsões fora da amostra: "
    f"{len(resultados_wf)}"
)

print(
    f"Mudanças de regime na janela de teste: "
    f"{int(resultados_wf['mudanca_regime'].sum())}"
)

print(
    f"Permanências de regime na janela de teste: "
    f"{int((~resultados_wf['mudanca_regime']).sum())}"
)


print("\nMÉTRICAS FORA DA AMOSTRA")
print("-" * 100)

metricas_exibicao = (
    metricas_wf
    .copy()
)

for col in [
    "acuracia",
    "balanced_accuracy",
    "macro_f1",
    "acuracia_mudancas",
    "acuracia_permanencias",
]:
    metricas_exibicao[col] = (
        metricas_exibicao[col]
        * 100
    )

print(
    metricas_exibicao
    .to_string(
        index=False,
        formatters={
            "acuracia": lambda x: f"{x:.2f}%",
            "balanced_accuracy": lambda x: f"{x:.2f}%",
            "macro_f1": lambda x: f"{x:.2f}%",
            "acuracia_mudancas": lambda x: (
                "NA"
                if pd.isna(x)
                else f"{x:.2f}%"
            ),
            "acuracia_permanencias": lambda x: (
                "NA"
                if pd.isna(x)
                else f"{x:.2f}%"
            ),
        }
    )
)


print("\nINTERPRETAÇÃO DA VALIDAÇÃO")
print("-" * 100)

print(
    "As métricas acima são fora da amostra e respeitam "
    "a ordem temporal."
)

print(
    "Acurácia mede a proporção total de previsões corretas."
)

print(
    "Balanced Accuracy e Macro-F1 são importantes porque "
    "os regimes possuem frequências diferentes."
)

print(
    "Acurácia nas mudanças mede especificamente a capacidade "
    "de antecipar transições de regime."
)

print(
    "A persistência é o benchmark que assume que o próximo "
    "regime será igual ao regime atual."
)

print(
    "\nArquivos da validação:"
)

print(
    " - outputs_modelo4_final/"
    "modelo4D_walkforward_previsoes.csv"
)

print(
    " - outputs_modelo4_final/"
    "modelo4D_walkforward_metricas.csv"
)

print(
    " - outputs_modelo4_final/"
    "modelo4D_matriz_confusao_*.csv"
)



# =============================================================================
# 12. PADRONIZAÇÃO
# =============================================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

X_futuro_scaled = scaler.transform(
    X_futuro
)


# =============================================================================
# 13. REGRESSÃO LOGÍSTICA
# =============================================================================

modelo_logit = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)


modelo_logit.fit(
    X_scaled,
    y
)


pred_logit = int(
    modelo_logit.predict(
        X_futuro_scaled
    )[0]
)


prob_logit = (
    modelo_logit
    .predict_proba(
        X_futuro_scaled
    )[0]
)


# =============================================================================
# 14. REDE NEURAL MLP
# =============================================================================

modelo_mlp = MLPClassifier(

    hidden_layer_sizes=(4,),

    activation="tanh",

    solver="lbfgs",

    alpha=2.0,

    max_iter=5000,

    random_state=42
)


modelo_mlp.fit(
    X_scaled,
    y
)


pred_neural = int(
    modelo_mlp.predict(
        X_futuro_scaled
    )[0]
)


prob_neural = (
    modelo_mlp
    .predict_proba(
        X_futuro_scaled
    )[0]
)


# =============================================================================
# 15. BENCHMARK DE PERSISTÊNCIA
# =============================================================================

regime_atual = int(
    ultima_linha[
        "regime_estimado"
    ].iloc[0]
)


pred_persistencia = (
    regime_atual
)


# =============================================================================
# 16. PROBABILIDADES
# =============================================================================

classes = sorted(
    set(modelo_logit.classes_)
    | set(modelo_mlp.classes_)
)


linhas_prob = []


for classe in classes:

    classe = int(classe)

    if classe in modelo_logit.classes_:

        indice_logit = (
            list(
                modelo_logit.classes_
            )
            .index(classe)
        )

        p_logit = (
            prob_logit[
                indice_logit
            ]
        )

    else:

        p_logit = np.nan


    if classe in modelo_mlp.classes_:

        indice_mlp = (
            list(
                modelo_mlp.classes_
            )
            .index(classe)
        )

        p_neural = (
            prob_neural[
                indice_mlp
            ]
        )

    else:

        p_neural = np.nan


    linhas_prob.append(
        {
            "regime":
                classe,

            "regime_economico":
                MAPA_REGIMES.get(
                    classe,
                    "DESCONHECIDO"
                ),

            "prob_logistica_pct":
                p_logit * 100,

            "prob_neural_pct":
                p_neural * 100
        }
    )


probabilidades = pd.DataFrame(
    linhas_prob
)


# =============================================================================
# 17. RESULTADO PROSPECTIVO
# =============================================================================

print("\n" + "=" * 100)
print("RESULTADO — PRÓXIMO TRIMESTRE")
print("=" * 100)


print(
    f"\nRegime atual em "
    f"{periodo_atual}: "
    f"{MAPA_REGIMES.get(regime_atual)} "
    f"(classe {regime_atual})"
)


print(
    f"\nPrevisão para "
    f"{periodo_previsto}:"
)


print(
    "\nBenchmark de persistência : "
    f"{MAPA_REGIMES.get(pred_persistencia)}"
)


print(
    "Regressão Logística       : "
    f"{MAPA_REGIMES.get(pred_logit)}"
)


print(
    "Rede Neural MLP           : "
    f"{MAPA_REGIMES.get(pred_neural)}"
)


print("\nPROBABILIDADES")
print("-" * 100)


print(
    probabilidades
    .round(2)
    .to_string(index=False)
)


# =============================================================================
# 18. VERIFICAR CONSENSO
# =============================================================================

previsoes = [
    pred_persistencia,
    pred_logit,
    pred_neural
]


if len(set(previsoes)) == 1:

    consenso = True

    regime_consenso = (
        MAPA_REGIMES.get(
            previsoes[0]
        )
    )

else:

    consenso = False

    regime_consenso = None


print("\n" + "-" * 100)


if consenso:

    print(
        "CONSENSO DOS TRÊS MÉTODOS: "
        f"{regime_consenso}"
    )

else:

    print(
        "NÃO HOUVE CONSENSO "
        "ENTRE OS TRÊS MÉTODOS."
    )


# =============================================================================
# 19. SALVAR RESULTADOS
# =============================================================================

probabilidades.to_csv(

    os.path.join(
        PASTA_SAIDA,
        "modelo4D_probabilidades.csv"
    ),

    index=False
)


resumo = pd.DataFrame(
    [
        {
            "periodo_atual":
                str(periodo_atual),

            "periodo_previsto":
                str(periodo_previsto),

            "regime_atual":
                regime_atual,

            "regime_atual_economico":
                MAPA_REGIMES.get(
                    regime_atual
                ),

            "persistencia":
                pred_persistencia,

            "persistencia_economico":
                MAPA_REGIMES.get(
                    pred_persistencia
                ),

            "logistica":
                pred_logit,

            "logistica_economico":
                MAPA_REGIMES.get(
                    pred_logit
                ),

            "neural":
                pred_neural,

            "neural_economico":
                MAPA_REGIMES.get(
                    pred_neural
                ),

            "consenso":
                consenso
        }
    ]
)


resumo.to_csv(

    os.path.join(
        PASTA_SAIDA,
        "modelo4D_previsao.csv"
    ),

    index=False
)


# =============================================================================
# 20. SALVAR MAPEAMENTO ECONÔMICO DOS REGIMES
# =============================================================================

mapeamento_saida = pd.DataFrame(
    [
        {
            "regime":
                int(regime),

            "regime_economico":
                MAPA_REGIMES[
                    int(regime)
                ],

            "media_juros_cartao_pf_aa_pct":
                float(media)
        }

        for regime, media
        in medias_regimes.items()
    ]
)


mapeamento_saida.to_csv(

    os.path.join(
        PASTA_SAIDA,
        "modelo4D_mapeamento_regimes.csv"
    ),

    index=False
)


print("\nArquivos gerados:")

print(
    f"{PASTA_SAIDA}/"
    "modelo4D_probabilidades.csv"
)

print(
    f"{PASTA_SAIDA}/"
    "modelo4D_previsao.csv"
)

print(
    f"{PASTA_SAIDA}/"
    "modelo4D_mapeamento_regimes.csv"
)


print("\n" + "=" * 100)
print("FIM DO MODELO 4D")
print("=" * 100)