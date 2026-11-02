"""
config.py
=========================================================
Configurações centrais do framework de acompanhamento
do mercado brasileiro de cartão de crédito PF.

Este arquivo concentra:
- caminhos do projeto;
- nomes dos arquivos;
- definição dos períodos;
- métricas do dashboard;
- metadados das métricas;
- fontes;
- configurações dos modelos;
- parâmetros de atualização.

IMPORTANTE:
Não executar modelos neste arquivo.
Ele deve conter apenas configurações compartilhadas.
=========================================================
"""

from pathlib import Path


# =========================================================
# 1. CAMINHOS DO PROJETO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Raiz geral do projeto bcb_project.
# O dataset_model_ready.csv é produzido nesta pasta.
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
TABELAS_DIR = OUTPUTS_DIR / "tabelas"
GRAFICOS_DIR = OUTPUTS_DIR / "graficos"
MODELOS_DIR = OUTPUTS_DIR / "modelos"

for pasta in [
    DATA_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    TABELAS_DIR,
    GRAFICOS_DIR,
    MODELOS_DIR,
]:
    pasta.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. ARQUIVOS PRINCIPAIS
# =========================================================

ARQUIVO_BASE_ORIGINAL = (
    PROJECT_DIR / "dataset_model_ready.csv"
)

ARQUIVO_MASTER_MENSAL = (
    DATA_DIR / "metricas_master_mensal.csv"
)

ARQUIVO_MASTER_TRIMESTRAL = (
    DATA_DIR / "metricas_master_trimestral.csv"
)

ARQUIVO_METRICAS_DASHBOARD = (
    DATA_DIR / "metricas_dashboard.csv"
)

ARQUIVO_HISTORICO_PREVISOES = (
    DATA_DIR / "historico_previsoes.csv"
)

ARQUIVO_STATUS_ATUALIZACAO = (
    DATA_DIR / "status_atualizacao.csv"
)

ARQUIVO_DICIONARIO_METRICAS = (
    DATA_DIR / "dicionario_metricas.csv"
)


# =========================================================
# 3. COLUNA DE DATA
# =========================================================

COLUNA_DATA = "date"


# =========================================================
# 4. PERÍODOS DA PANDEMIA
# =========================================================

PRE_PANDEMIA_FIM = "2019-12-31"

PANDEMIA_INICIO = "2020-01-01"
PANDEMIA_FIM = "2021-12-31"

POS_PANDEMIA_INICIO = "2022-01-01"


# =========================================================
# 5. FONTES DE DADOS
# =========================================================

FONTES_DADOS = {
    "BCB_SGS":
        "Banco Central do Brasil - SGS/BCData",

    "IFDATA_CREDITO":
        "Banco Central do Brasil - IFData/Olinda, "
        "Relatório 1 - Carteira de Crédito",

    "IFDATA_CARTAO":
        "Banco Central do Brasil - IFData/Olinda, "
        "Relatório 11 - grupo Cartão de Crédito",

    "IFDATA_COMPETICAO":
        "Banco Central do Brasil - IFData/Olinda "
        "e cadastro de instituições",

    "IBGE_SIDRA":
        "Instituto Brasileiro de Geografia e Estatística "
        "- SIDRA",

    "CVM_ITR":
        "Comissão de Valores Mobiliários - ITR",

    "CALCULO_PROPRIO":
        "Cálculo próprio a partir das bases públicas "
        "utilizadas no estudo",
}


# =========================================================
# 6. CÓDIGOS SGS
# =========================================================

SGS_SERIES = {
    "selic_am_pct": 4390,
    "juros_cartao_pf_aa_pct": 22024,
    "inadimplencia_90d_pct": 21082,
    "concessoes_total_mr": 20631,
    "saldo_cartao_pf_total_mr": 20590,
    "cartoes_emitidos_milhoes": 25147,
    "juros_cartao_rotativo_pf_mensal_pct": 25477,
    "juros_cartao_parcelado_pf_mensal_pct": 25478,
    "saldo_cartao_rotativo_pf_mr": 20587,
    "saldo_cartao_parcelado_pf_mr": 20588,
    "concessoes_cartao_rotativo_pf_mr": 20679,
    "concessoes_cartao_parcelado_pf_mr": 20680,
    "endividamento_familias_pct_renda": 29038,
    "comprometimento_renda_pct": 29265,
}


# =========================================================
# 7. INFORMAÇÕES IFData
# =========================================================

IFDATA_RELATORIO_HHI = "1"

IFDATA_NOMECOLUNA_CREDITO_ATE_2024 = (
    "Carteira de Crédito Classificada"
)

IFDATA_NOMECOLUNA_CREDITO_DESDE_2025 = (
    "Carteira de Crédito"
)

IFDATA_RELATORIO_CARTAO = "11"

IFDATA_GRUPO_CARTAO = "Cartão de Crédito"


# =========================================================
# 8. FUNÇÃO PARA DEFINIR METADADOS
# =========================================================

def _meta(
    nome,
    grupo,
    descricao,
    fonte,
    unidade,
    frequencia,
    formato="numero",
    classificacao="Secundário",
    construcao="Observação direta da fonte.",
    uso_modelos="Análise descritiva",
    referencia=None,
):
    return {
        "nome": nome,
        "grupo": grupo,
        "descricao": descricao,
        "fonte": fonte,
        "unidade": unidade,
        "frequencia": frequencia,
        "formato": formato,
        "classificacao": classificacao,
        "construcao": construcao,
        "uso_modelos": uso_modelos,
        "referencia": referencia,
    }


# =========================================================
# 9. DICIONÁRIO CENTRAL DAS 62 MÉTRICAS
# =========================================================

METADADOS_METRICAS = {

    # -----------------------------------------------------
    # BCB / SGS
    # -----------------------------------------------------

    "selic_am_pct": _meta(
        "Selic",
        "Política monetária",
        "Taxa Selic mensal utilizada como referência "
        "do custo básico de recursos na economia.",
        "BCB_SGS",
        "% ao mês",
        "mensal",
        "percentual",
        "KPI principal",
        referencia="SGS 4390",
        uso_modelos="Modelos 3, 4D e 4E",
    ),

    "juros_cartao_pf_aa_pct": _meta(
        "Juros do cartão PF",
        "Preço do crédito",
        "Taxa média de juros das operações de cartão "
        "de crédito para pessoas físicas.",
        "BCB_SGS",
        "% ao ano",
        "mensal",
        "percentual",
        "KPI principal",
        referencia="SGS 22024",
        uso_modelos="Modelos 2, 3, 4D e 4E",
    ),

    "inadimplencia_90d_pct": _meta(
        "Inadimplência > 90 dias",
        "Risco de crédito",
        "Percentual das operações de crédito com "
        "atraso superior a 90 dias.",
        "BCB_SGS",
        "%",
        "mensal",
        "percentual",
        "KPI principal",
        referencia="SGS 21082",
        uso_modelos="Modelos 2, 3, 4D e 4E",
    ),

    "concessoes_total_mr": _meta(
        "Concessões totais de crédito",
        "Mercado de crédito",
        "Fluxo de concessões de crédito utilizado para "
        "acompanhar a atividade do mercado.",
        "BCB_SGS",
        "R$ milhões",
        "mensal",
        "numero",
        "Secundário",
        referencia="SGS 20631",
    ),

    "saldo_cartao_pf_total_mr": _meta(
        "Saldo total cartão PF",
        "Mercado de cartão",
        "Estoque total das operações de cartão de "
        "crédito de pessoas físicas.",
        "BCB_SGS",
        "R$ milhões",
        "mensal",
        "numero",
        "KPI principal",
        referencia="SGS 20590",
        uso_modelos="Modelos 3 e análises de mercado",
    ),

    "cartoes_emitidos_milhoes": _meta(
        "Cartões emitidos",
        "Mercado de cartão",
        "Quantidade de cartões emitidos.",
        "BCB_SGS",
        "milhões de cartões",
        "mensal",
        "numero",
        "Secundário",
        referencia="SGS 25147",
    ),

    "juros_cartao_rotativo_pf_mensal_pct": _meta(
        "Juros do rotativo",
        "Preço do crédito",
        "Taxa mensal de juros do crédito rotativo "
        "do cartão PF.",
        "BCB_SGS",
        "% ao mês",
        "mensal",
        "percentual",
        "Secundário",
        referencia="SGS 25477",
    ),

    "juros_cartao_parcelado_pf_mensal_pct": _meta(
        "Juros do parcelado",
        "Preço do crédito",
        "Taxa mensal de juros do cartão parcelado PF.",
        "BCB_SGS",
        "% ao mês",
        "mensal",
        "percentual",
        "Secundário",
        referencia="SGS 25478",
    ),

    "saldo_cartao_rotativo_pf_mr": _meta(
        "Saldo do rotativo",
        "Mercado de cartão",
        "Saldo das operações de crédito rotativo "
        "do cartão PF.",
        "BCB_SGS",
        "R$ milhões",
        "mensal",
        "numero",
        "Secundário",
        referencia="SGS 20587",
    ),

    "saldo_cartao_parcelado_pf_mr": _meta(
        "Saldo do parcelado",
        "Mercado de cartão",
        "Saldo das operações parceladas do cartão PF.",
        "BCB_SGS",
        "R$ milhões",
        "mensal",
        "numero",
        "Secundário",
        referencia="SGS 20588",
    ),

    "concessoes_cartao_rotativo_pf_mr": _meta(
        "Concessões do rotativo",
        "Mercado de cartão",
        "Fluxo de concessões do crédito rotativo "
        "do cartão PF.",
        "BCB_SGS",
        "R$ milhões",
        "mensal",
        "numero",
        "Secundário",
        referencia="SGS 20679",
    ),

    "concessoes_cartao_parcelado_pf_mr": _meta(
        "Concessões do parcelado",
        "Mercado de cartão",
        "Fluxo de concessões das operações parceladas "
        "do cartão PF.",
        "BCB_SGS",
        "R$ milhões",
        "mensal",
        "numero",
        "Secundário",
        referencia="SGS 20680",
    ),

    "endividamento_familias_pct_renda": _meta(
        "Endividamento das famílias",
        "Famílias",
        "Endividamento das famílias em relação "
        "à renda.",
        "BCB_SGS",
        "% da renda",
        "mensal",
        "percentual",
        "Secundário",
        referencia="SGS 29038",
    ),

    "comprometimento_renda_pct": _meta(
        "Comprometimento de renda",
        "Famílias",
        "Percentual da renda das famílias comprometido "
        "com o serviço das dívidas.",
        "BCB_SGS",
        "%",
        "mensal",
        "percentual",
        "KPI principal",
        referencia="SGS 29265",
        uso_modelos="Modelos 3, 4D e 4E",
    ),

    # -----------------------------------------------------
    # IFData — concentração do crédito
    # -----------------------------------------------------

    "hhi_credito_0_1": _meta(
        "HHI do crédito",
        "Competição",
        "Índice Herfindahl-Hirschman calculado a partir "
        "das participações das instituições na carteira "
        "de crédito.",
        "IFDATA_CREDITO",
        "índice de 0 a 1",
        "trimestral",
        "decimal",
        "KPI principal",
        construcao=(
            "Soma dos quadrados das participações das "
            "instituições na carteira de crédito. Até "
            "2024Q4 utiliza 'Carteira de Crédito "
            "Classificada'; desde 2025Q1 utiliza "
            "'Carteira de Crédito'."
        ),
        referencia="IFData relatório 1",
        uso_modelos="Modelos 1, 2C, 3, 4D e 4E",
    ),

    "hhi_credito_0_10000": _meta(
        "HHI do crédito - escala 10.000",
        "Competição",
        "HHI apresentado na escala convencional "
        "de zero a dez mil pontos.",
        "IFDATA_CREDITO",
        "pontos HHI",
        "trimestral",
        "numero",
        "Apoio/diagnóstico",
        construcao="hhi_credito_0_1 × 10.000.",
        referencia="IFData relatório 1",
    ),

    "cr5_credito": _meta(
        "CR5 do crédito",
        "Competição",
        "Participação conjunta das cinco maiores "
        "instituições na carteira de crédito.",
        "IFDATA_CREDITO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Secundário",
        construcao=(
            "Soma das participações das cinco maiores "
            "instituições na carteira de crédito."
        ),
        referencia="IFData relatório 1",
        uso_modelos="Modelos 3, 4D e 4E",
    ),

    "n_instituicoes_credito": _meta(
        "Número de instituições de crédito",
        "Competição",
        "Quantidade de instituições consideradas "
        "no cálculo da estrutura de mercado.",
        "IFDATA_CREDITO",
        "instituições",
        "trimestral",
        "numero",
        "Secundário",
        referencia="IFData relatório 1",
    ),

    # -----------------------------------------------------
    # IFData — cartão
    # -----------------------------------------------------

    "total_cartao": _meta(
        "Total de cartão - IFData",
        "Mercado de cartão",
        "Valor agregado do grupo Cartão de Crédito "
        "entre as instituições presentes no IFData.",
        "IFDATA_CARTAO",
        "valor monetário da fonte",
        "trimestral",
        "numero",
        "Secundário",
        construcao=(
            "Soma do campo total por instituição para "
            "o grupo Cartão de Crédito."
        ),
        referencia="IFData relatório 11",
    ),

    "total_cartao_ativo": _meta(
        "Total ativo de cartão",
        "Mercado de cartão",
        "Parcela não vencida do total de cartão "
        "obtido no IFData.",
        "IFDATA_CARTAO",
        "valor monetário da fonte",
        "trimestral",
        "numero",
        "Secundário",
        construcao=(
            "total_cartao - total_cartao_vencido."
        ),
        referencia="IFData relatório 11",
    ),

    "total_cartao_vencido": _meta(
        "Total vencido de cartão",
        "Risco de crédito",
        "Valor vencido agregado das operações do "
        "grupo Cartão de Crédito no IFData.",
        "IFDATA_CARTAO",
        "valor monetário da fonte",
        "trimestral",
        "numero",
        "Secundário",
        construcao=(
            "Soma do valor vencido entre instituições."
        ),
        referencia="IFData relatório 11",
    ),

    "tx_vencido_cartao": _meta(
        "Taxa de vencidos do cartão",
        "Risco de crédito",
        "Participação do valor vencido no total "
        "de cartão do IFData.",
        "IFDATA_CARTAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Secundário",
        construcao=(
            "total_cartao_vencido / total_cartao."
        ),
        referencia="IFData relatório 11",
    ),

    "ihhn_cartao_0_1": _meta(
        "IHHn do cartão",
        "Competição",
        "Índice de concentração calculado especificamente "
        "para o grupo Cartão de Crédito.",
        "IFDATA_CARTAO",
        "índice de 0 a 1",
        "trimestral",
        "decimal",
        "Secundário",
        construcao=(
            "Soma dos quadrados das participações de "
            "cada instituição no total de cartão."
        ),
        referencia="IFData relatório 11",
    ),

    "en_cartao": _meta(
        "Número equivalente de instituições - cartão",
        "Competição",
        "Número efetivo de instituições correspondente "
        "ao grau de concentração observado no cartão.",
        "IFDATA_CARTAO",
        "instituições equivalentes",
        "trimestral",
        "numero",
        "Secundário",
        construcao="1 / ihhn_cartao_0_1.",
        referencia="IFData relatório 11",
    ),

    # -----------------------------------------------------
    # IFData — estrutura competitiva
    # -----------------------------------------------------

    "share_bancos_tradicionais": _meta(
        "Participação dos bancos tradicionais",
        "Competição",
        "Participação dos bancos classificados como "
        "tradicionais na carteira analisada.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Secundário",
        uso_modelos="Modelos 1 e 3",
    ),

    "share_bancos_medios": _meta(
        "Participação dos bancos médios",
        "Competição",
        "Participação das instituições classificadas "
        "como bancos médios.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Secundário",
    ),

    "share_fintechs": _meta(
        "Participação das fintechs",
        "Competição",
        "Participação das instituições classificadas "
        "como fintechs no mercado analisado.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "KPI principal",
        uso_modelos="Modelos 1, 2B, 2D, 3 e 4D",
    ),

    "share_cooperativas": _meta(
        "Participação das cooperativas",
        "Competição",
        "Participação das cooperativas de crédito.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Secundário",
    ),

    "share_instituicoes_pagamento": _meta(
        "Participação das instituições de pagamento",
        "Competição",
        "Participação das instituições classificadas "
        "como instituições de pagamento.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Secundário",
    ),

    "share_financeiras": _meta(
        "Participação das financeiras",
        "Competição",
        "Participação das instituições classificadas "
        "como financeiras.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Secundário",
    ),

    "share_agencias_fomento": _meta(
        "Participação das agências de fomento",
        "Competição",
        "Participação das agências de fomento "
        "na carteira analisada.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Apoio/diagnóstico",
    ),

    "share_mercado_capitais": _meta(
        "Participação do mercado de capitais",
        "Competição",
        "Participação do grupo classificado como "
        "mercado de capitais.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Apoio/diagnóstico",
    ),

    "share_outros": _meta(
        "Participação de outras instituições",
        "Competição",
        "Participação residual das demais categorias "
        "institucionais.",
        "IFDATA_COMPETICAO",
        "proporção de 0 a 1",
        "trimestral",
        "proporcao",
        "Apoio/diagnóstico",
    ),

    "entrada_inst": _meta(
        "Entrada de instituições",
        "Competição",
        "Número de instituições presentes no período "
        "atual e ausentes no período anterior.",
        "IFDATA_COMPETICAO",
        "instituições",
        "trimestral",
        "numero",
        "Secundário",
        construcao=(
            "Contagem de instituições classificadas "
            "como entrantes em relação ao trimestre anterior."
        ),
        uso_modelos="Modelo 1",
    ),

    "saida_inst": _meta(
        "Saída de instituições",
        "Competição",
        "Número de instituições presentes no período "
        "anterior e ausentes no período atual.",
        "IFDATA_COMPETICAO",
        "instituições",
        "trimestral",
        "numero",
        "Secundário",
        construcao=(
            "Contagem de instituições classificadas "
            "como saídas em relação ao trimestre anterior."
        ),
        uso_modelos="Modelo 1",
    ),

    "check_soma_shares": _meta(
        "Validação da soma das participações",
        "Qualidade dos dados",
        "Indicador utilizado para verificar se as "
        "participações institucionais agregadas são "
        "consistentes.",
        "CALCULO_PROPRIO",
        "proporção",
        "trimestral",
        "decimal",
        "Apoio/diagnóstico",
        construcao=(
            "Soma das participações das categorias "
            "institucionais."
        ),
        uso_modelos="Controle de qualidade",
    ),

    # -----------------------------------------------------
    # IBGE — inflação
    # -----------------------------------------------------

    "ipca_mensal_pct": _meta(
        "IPCA",
        "Macroeconomia",
        "Variação mensal do Índice Nacional de Preços "
        "ao Consumidor Amplo.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Secundário",
        referencia="SIDRA tabela 7060",
    ),

    "ipca_alimentacao_pct": _meta(
        "IPCA - Alimentação e bebidas",
        "Macroeconomia",
        "Variação do grupo Alimentação e bebidas "
        "no IPCA.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7060",
    ),

    "ipca_transportes_pct": _meta(
        "IPCA - Transportes",
        "Macroeconomia",
        "Variação do grupo Transportes no IPCA.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7060",
    ),

    "ipca_saude_pct": _meta(
        "IPCA - Saúde e cuidados pessoais",
        "Macroeconomia",
        "Variação do grupo Saúde e cuidados pessoais "
        "no IPCA.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7060",
    ),

    "ipca_despesas_pessoais_pct": _meta(
        "IPCA - Despesas pessoais",
        "Macroeconomia",
        "Variação do grupo Despesas pessoais no IPCA.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7060",
    ),

    "ipca_educacao_pct": _meta(
        "IPCA - Educação",
        "Macroeconomia",
        "Variação do grupo Educação no IPCA.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7060",
    ),

    "ipca15_mensal_pct": _meta(
        "IPCA-15",
        "Macroeconomia",
        "Variação do IPCA-15.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Secundário",
        referencia="SIDRA tabela 7062",
    ),

    "ipca15_alimentacao_pct": _meta(
        "IPCA-15 - Alimentação e bebidas",
        "Macroeconomia",
        "Variação do grupo Alimentação e bebidas "
        "no IPCA-15.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7062",
    ),

    "ipca15_transportes_pct": _meta(
        "IPCA-15 - Transportes",
        "Macroeconomia",
        "Variação do grupo Transportes no IPCA-15.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7062",
    ),

    "ipca15_saude_pct": _meta(
        "IPCA-15 - Saúde e cuidados pessoais",
        "Macroeconomia",
        "Variação do grupo Saúde e cuidados pessoais "
        "no IPCA-15.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7062",
    ),

    "ipca15_despesas_pessoais_pct": _meta(
        "IPCA-15 - Despesas pessoais",
        "Macroeconomia",
        "Variação do grupo Despesas pessoais no IPCA-15.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7062",
    ),

    "ipca15_educacao_pct": _meta(
        "IPCA-15 - Educação",
        "Macroeconomia",
        "Variação do grupo Educação no IPCA-15.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Apoio/diagnóstico",
        referencia="SIDRA tabela 7062",
    ),

    # -----------------------------------------------------
    # IBGE — trabalho e renda
    # -----------------------------------------------------

    "taxa_desocupacao_movel_pct": _meta(
        "Taxa de desocupação móvel",
        "Macroeconomia",
        "Taxa de desocupação da PNAD Contínua "
        "em janela móvel.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Secundário",
        referencia="SIDRA tabela 6381",
    ),

    "taxa_subutilizacao_movel_pct": _meta(
        "Taxa de subutilização móvel",
        "Macroeconomia",
        "Taxa composta de subutilização da força de "
        "trabalho em janela móvel.",
        "IBGE_SIDRA",
        "%",
        "mensal",
        "percentual",
        "Secundário",
        referencia="SIDRA tabela 6441",
    ),

    "taxa_desocupacao_trimestral_pct": _meta(
        "Taxa de desocupação",
        "Macroeconomia",
        "Taxa trimestral de desocupação da população.",
        "IBGE_SIDRA",
        "%",
        "trimestral",
        "percentual",
        "KPI principal",
        referencia="SIDRA tabela 6468",
        uso_modelos="Modelos 3, 4D e 4E",
    ),

    "taxa_subutilizacao_trimestral_pct": _meta(
        "Taxa de subutilização",
        "Macroeconomia",
        "Taxa trimestral composta de subutilização "
        "da força de trabalho.",
        "IBGE_SIDRA",
        "%",
        "trimestral",
        "percentual",
        "Secundário",
        referencia="SIDRA tabela 6485",
    ),

    "rendimento_medio_real_brl": _meta(
        "Rendimento médio real",
        "Macroeconomia",
        "Rendimento médio mensal real do trabalho.",
        "IBGE_SIDRA",
        "R$",
        "trimestral",
        "moeda",
        "KPI principal",
        referencia="SIDRA tabela 6470",
        uso_modelos="Modelos 3, 4D e 4E",
    ),

    "massa_rendimento_real_mr": _meta(
        "Massa de rendimento real",
        "Macroeconomia",
        "Massa de rendimento mensal real do trabalho.",
        "IBGE_SIDRA",
        "R$ milhões",
        "trimestral",
        "numero",
        "Secundário",
        referencia="SIDRA tabela 6474",
    ),

    # -----------------------------------------------------
    # MÉTRICAS DERIVADAS
    # -----------------------------------------------------

    "dlog_total_cartao": _meta(
        "Crescimento do total de cartão",
        "Métricas derivadas",
        "Variação logarítmica do total de cartão.",
        "CALCULO_PROPRIO",
        "variação logarítmica",
        "trimestral",
        "decimal",
        "Apoio/diagnóstico",
        construcao="ln(total_cartao_t) - ln(total_cartao_t-1).",
    ),

    "delta_hhi_cartao": _meta(
        "Variação do IHHn do cartão",
        "Métricas derivadas",
        "Mudança do índice de concentração específico "
        "do cartão entre períodos.",
        "CALCULO_PROPRIO",
        "variação do índice",
        "trimestral",
        "decimal",
        "Secundário",
        construcao=(
            "ihhn_cartao_0_1_t - ihhn_cartao_0_1_t-1."
        ),
    ),

    "spread_cartao_pf": _meta(
        "Spread do cartão PF",
        "Preço do crédito",
        "Diferença aproximada entre a taxa anual "
        "do cartão PF e a Selic anualizada.",
        "CALCULO_PROPRIO",
        "pontos percentuais",
        "mensal",
        "percentual",
        "KPI principal",
        construcao=(
            "juros_cartao_pf_aa_pct - "
            "(selic_am_pct × 12)."
        ),
        uso_modelos="Modelo 2",
    ),

    "diferencial_rotativo_parcelado": _meta(
        "Diferencial rotativo x parcelado",
        "Preço do crédito",
        "Diferença entre as taxas mensais do crédito "
        "rotativo e do cartão parcelado.",
        "CALCULO_PROPRIO",
        "pontos percentuais",
        "mensal",
        "percentual",
        "Secundário",
        construcao=(
            "juros_cartao_rotativo_pf_mensal_pct - "
            "juros_cartao_parcelado_pf_mensal_pct."
        ),
    ),

    "prop_rotativo": _meta(
        "Proporção do rotativo",
        "Mercado de cartão",
        "Participação do saldo rotativo no saldo total "
        "de cartão PF.",
        "CALCULO_PROPRIO",
        "proporção de 0 a 1",
        "mensal",
        "proporcao",
        "KPI principal",
        construcao=(
            "saldo_cartao_rotativo_pf_mr / "
            "saldo_cartao_pf_total_mr."
        ),
        uso_modelos="Modelo 4E e análises de mercado",
    ),

    "dlog_saldo_cartao_pf_total_mr": _meta(
        "Crescimento do saldo cartão PF",
        "Métricas derivadas",
        "Variação logarítmica do saldo total "
        "de cartão PF.",
        "CALCULO_PROPRIO",
        "variação logarítmica",
        "mensal",
        "decimal",
        "Apoio/diagnóstico",
        construcao=(
            "ln(saldo_cartao_pf_total_mr_t) - "
            "ln(saldo_cartao_pf_total_mr_t-1)."
        ),
    ),

    "delta_hhi_credito": _meta(
        "Variação do HHI do crédito",
        "Competição",
        "Mudança do HHI da carteira de crédito entre "
        "dois períodos consecutivos.",
        "CALCULO_PROPRIO",
        "variação do índice",
        "trimestral",
        "decimal",
        "Secundário",
        construcao=(
            "hhi_credito_0_1_t - "
            "hhi_credito_0_1_t-1."
        ),
        uso_modelos="Modelo 1C",
    ),

    "stress_familias": _meta(
        "Estresse financeiro das famílias",
        "Famílias",
        "Indicador composto de endividamento e "
        "comprometimento de renda.",
        "CALCULO_PROPRIO",
        "índice",
        "mensal",
        "numero",
        "Secundário",
        construcao=(
            "(endividamento_familias_pct_renda × "
            "comprometimento_renda_pct) / 100."
        ),
        uso_modelos="Modelo 2A e análises econômicas",
    ),
}


# =========================================================
# 10. VALIDAÇÃO DA QUANTIDADE DE MÉTRICAS
# =========================================================

METRICAS_ECONOMICAS = list(
    METADADOS_METRICAS.keys()
)

NUMERO_METRICAS_ECONOMICAS = len(
    METRICAS_ECONOMICAS
)


# =========================================================
# 11. DICIONÁRIOS DERIVADOS DOS METADADOS
# =========================================================

NOMES_METRICAS = {
    metrica: info["nome"]
    for metrica, info
    in METADADOS_METRICAS.items()
}

DESCRICOES_METRICAS = {
    metrica: info["descricao"]
    for metrica, info
    in METADADOS_METRICAS.items()
}

FORMATOS_METRICAS = {
    metrica: info["formato"]
    for metrica, info
    in METADADOS_METRICAS.items()
}

FREQUENCIA_METRICAS = {
    metrica: info["frequencia"]
    for metrica, info
    in METADADOS_METRICAS.items()
}

GRUPOS_METRICAS = {
    metrica: info["grupo"]
    for metrica, info
    in METADADOS_METRICAS.items()
}

CLASSIFICACAO_METRICAS = {
    metrica: info["classificacao"]
    for metrica, info
    in METADADOS_METRICAS.items()
}


# =========================================================
# 12. ALIASES UTILIZADOS POR ALGUNS MODELOS
# =========================================================

# Alguns scripts antigos adicionam o prefixo "ibge_"
# após o merge das bases. Mantemos os aliases para
# compatibilidade, mas a variável oficial no master
# continua sem o prefixo.

NOMES_METRICAS.update({
    "ibge_taxa_desocupacao_trimestral_pct":
        "Taxa de desocupação",

    "ibge_rendimento_medio_real_brl":
        "Rendimento médio real",
})

FORMATOS_METRICAS.update({
    "ibge_taxa_desocupacao_trimestral_pct":
        "percentual",

    "ibge_rendimento_medio_real_brl":
        "moeda",
})

FREQUENCIA_METRICAS.update({
    "ibge_taxa_desocupacao_trimestral_pct":
        "trimestral",

    "ibge_rendimento_medio_real_brl":
        "trimestral",
})


# =========================================================
# 13. KPIs DA CAPA
# =========================================================

KPIS_PRINCIPAIS = [
    "juros_cartao_pf_aa_pct",
    "spread_cartao_pf",
    "hhi_credito_0_1",
    "share_fintechs",
    "inadimplencia_90d_pct",
    "saldo_cartao_pf_total_mr",
    "comprometimento_renda_pct",
    "selic_am_pct",
    "taxa_desocupacao_trimestral_pct",
    "rendimento_medio_real_brl",
]


# Mantido por compatibilidade com o pipeline atual.
# Inclui também indicadores secundários utilizados
# em outras partes do dashboard.

METRICAS_DASHBOARD = [
    "juros_cartao_pf_aa_pct",
    "spread_cartao_pf",
    "inadimplencia_90d_pct",
    "saldo_cartao_pf_total_mr",
    "prop_rotativo",
    "selic_am_pct",
    "hhi_credito_0_1",
    "cr5_credito",
    "share_fintechs",
    "share_bancos_tradicionais",
    "comprometimento_renda_pct",
    "taxa_desocupacao_trimestral_pct",
    "rendimento_medio_real_brl",
]


# =========================================================
# 14. MÉTRICAS PARA CORRELAÇÃO
# =========================================================

METRICAS_CORRELACAO = [
    "juros_cartao_pf_aa_pct",
    "spread_cartao_pf",
    "selic_am_pct",
    "inadimplencia_90d_pct",
    "hhi_credito_0_1",
    "cr5_credito",
    "share_fintechs",
    "share_bancos_tradicionais",
    "comprometimento_renda_pct",
    "saldo_cartao_pf_total_mr",
    "prop_rotativo",
    "taxa_desocupacao_trimestral_pct",
    "rendimento_medio_real_brl",
]


# =========================================================
# 15. MODELO 3 — MARKOV
# =========================================================

VARIAVEL_REGIME = "juros_cartao_pf_aa_pct"

NUMERO_REGIMES = 3

FREQUENCIA_MODELOS = "trimestral"

# Estes códigos são apenas rótulos econômicos do dashboard.
# O Modelo 3 deve classificar os estados pelo nível médio
# estimado dos juros, não pelo número bruto retornado pelo
# MarkovRegression.

REGIMES = {
    0: "Baixo",
    1: "Intermediário",
    2: "Alto",
}

ORDEM_REGIMES = [
    "Baixo",
    "Intermediário",
    "Alto",
]


# =========================================================
# 16. MODELO 4D
# =========================================================

FEATURES_MODELO_4D = [
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


# =========================================================
# 17. MODELO 4E
# =========================================================

METRICAS_PREVISAO = [
    "juros_cartao_pf_aa_pct",
    "inadimplencia_90d_pct",
    "hhi_credito_0_1",
    "cr5_credito",
]

FEATURES_MODELO_4E = [
    "juros_cartao_pf_aa_pct",
    "selic_am_pct",
    "inadimplencia_90d_pct",
    "hhi_credito_0_1",
    "cr5_credito",
    "comprometimento_renda_pct",
    "taxa_desocupacao_trimestral_pct",
    "rendimento_medio_real_brl",
    "prop_rotativo",
]


# =========================================================
# 18. CONFIGURAÇÕES DE PREVISÃO
# =========================================================

HORIZONTE_PREVISAO_TRIMESTRES = 1

MIN_TREINO_MODELO_4D = 16
MIN_TREINO_MODELO_4E = 16

RANDOM_STATE = 42

MODELOS_PREVISAO = [
    "Persistência",
    "Ridge",
    "MLP",
]


# =========================================================
# 19. HISTÓRICO DAS PREVISÕES
# =========================================================

COLUNAS_HISTORICO_PREVISOES = [
    "data_execucao",
    "origem_previsao",
    "periodo_previsto",
    "tipo_previsao",
    "metrica",
    "valor_previsto",
    "modelo",
    "ultimo_valor_disponivel",
    "periodo_fonte",
    "valor_realizado",
    "erro",
]


# =========================================================
# 20. DASHBOARD
# =========================================================

DASHBOARD_TITULO = (
    "Mercado Brasileiro de Cartão de Crédito PF"
)

DASHBOARD_SUBTITULO = (
    "Framework Híbrido Markov-Neural"
)

DASHBOARD_LAYOUT = "wide"

ABAS_DASHBOARD = [
    "Visão Geral",
    "Tabelão",
    "Histórico",
    "Competição",
    "Famílias",
    "IBGE",
    "Correlações",
    "Pré / Pandemia / Pós",
    "Regimes",
    "Previsões",
    "Resultados da Pesquisa",
    "Dicionário",
]


# =========================================================
# 21. ATUALIZAÇÃO AUTOMÁTICA
# =========================================================

FREQUENCIA_ATUALIZACAO = "mensal"

RODAR_MODELOS_APENAS_NOVO_TRIMESTRE = True


# =========================================================
# 22. FUNÇÕES AUXILIARES
# =========================================================

def nome_metrica(coluna):
    return NOMES_METRICAS.get(
        coluna,
        coluna,
    )


def descricao_metrica(coluna):
    return DESCRICOES_METRICAS.get(
        coluna,
        "Descrição ainda não cadastrada "
        "para esta métrica.",
    )


def formato_metrica(coluna):
    return FORMATOS_METRICAS.get(
        coluna,
        "numero",
    )


def frequencia_metrica(coluna):
    return FREQUENCIA_METRICAS.get(
        coluna,
        "desconhecida",
    )


def grupo_metrica(coluna):
    info = METADADOS_METRICAS.get(
        coluna,
        {},
    )

    return info.get(
        "grupo",
        "Não classificado",
    )


def fonte_metrica(coluna):
    info = METADADOS_METRICAS.get(
        coluna,
        {},
    )

    chave = info.get(
        "fonte"
    )

    if chave is None:
        return "Fonte não cadastrada"

    return FONTES_DADOS.get(
        chave,
        chave,
    )


def unidade_metrica(coluna):
    info = METADADOS_METRICAS.get(
        coluna,
        {},
    )

    return info.get(
        "unidade",
        "",
    )


def classificacao_metrica(coluna):
    info = METADADOS_METRICAS.get(
        coluna,
        {},
    )

    return info.get(
        "classificacao",
        "",
    )


def construcao_metrica(coluna):
    info = METADADOS_METRICAS.get(
        coluna,
        {},
    )

    return info.get(
        "construcao",
        "",
    )


def uso_modelos_metrica(coluna):
    info = METADADOS_METRICAS.get(
        coluna,
        {},
    )

    return info.get(
        "uso_modelos",
        "",
    )


def referencia_metrica(coluna):
    info = METADADOS_METRICAS.get(
        coluna,
        {},
    )

    referencia = info.get(
        "referencia"
    )

    if referencia:
        return referencia

    if coluna in SGS_SERIES:
        return f"SGS {SGS_SERIES[coluna]}"

    return ""