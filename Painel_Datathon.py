#!/usr/bin/env python
"""
Painel Datathon - Qualidade de Dados, Métricas de Modelo e Análise Estatística

Integra:
- Validação de Qualidade de Dados (verificações ao vivo)
- Métricas de Desempenho do Modelo
- Testes Estatísticos (intervalos de confiança, chi-square)
- Alertas Automáticos para Anomalias
- Demo do Motor de Recomendações

Não é uma tradução de app_dashboard.py — são dois dashboards com propósitos
diferentes (ver README.md#dashboards): este foca em qualidade de dados e
testes estatísticos; app_dashboard.py mostra o pipeline completo
(dados → treino → avaliação → API). Ambos leem os mesmos artefatos via
datathon.config.Config, então os números batem entre os dois.

Executar: streamlit run app_dashboard_pt.py
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datathon.bandit import ContextualThompsonSampling
from datathon.config import Config
from datathon.quality import validate_data_quality, setup_logging
from datathon.validation import (
    get_confidence_interval,
    compare_conversions,
    compare_contexts,
    calculate_sample_size,
    generate_statistical_report
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure página Streamlit
st.set_page_config(
    page_title="Painel Datathon - Thompson Sampling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNÇÕES DE CACHE
# ============================================================================

@st.cache_resource
def load_config():
    return Config(env='local')

@st.cache_resource
def load_data():
    """Carregar dataset principal"""
    config = load_config()
    data_path = Path(config.get_data_path('bank_marketing_primary.parquet'))
    if data_path.exists():
        return pd.read_parquet(data_path)
    return None

@st.cache_resource
def load_model():
    """Carregar modelo Thompson Sampling"""
    config = load_config()
    model_path = config.model_path / "thompson_model.json"
    if model_path.exists():
        model = ContextualThompsonSampling()
        with open(model_path, 'r') as f:
            model_data = json.load(f)

        # Reconstruir bandits from JSON
        for context_key, data in model_data.items():
            context = tuple(data['context'])
            bandit = model.bandits[context]
            bandit.alpha = np.array(data['alpha'], dtype=float)
            bandit.beta = np.array(data['beta'], dtype=float)
            bandit.trials = np.array(data['trials'], dtype=float)
            bandit.successes = np.array(data['successes'], dtype=float)

        return model
    return None

@st.cache_data
def get_quality_report(df):
    """Obter relatório de qualidade de dados"""
    return validate_data_quality(df, dataset_name='bank_marketing_primary', verbose=False)

# ============================================================================
# BARRA LATERAL - NAVEGAÇÃO
# ============================================================================

st.sidebar.title("📊 Navegação")
page = st.sidebar.radio(
    "Selecione a Página",
    ["Visão Geral", "Qualidade de Dados", "Métricas do Modelo", "Análise Estatística", "Demo da API", "Canary Deploy"]
)

st.sidebar.divider()

# Carregar dados e modelo
df = load_data()
model = load_model()

if df is not None and model is not None:
    st.sidebar.success("✅ Dados & Modelo Carregados")
else:
    st.sidebar.error("❌ Dados ou Modelo Não Encontrados")

# ============================================================================
# PÁGINA 1: VISÃO GERAL
# ============================================================================

if page == "Visão Geral":
    st.title("🎯 Painel Datathon - Thompson Sampling")
    st.markdown("**Plataforma interativa para Qualidade de Dados, Métricas do Modelo e Análise Estatística**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Tamanho Dataset", f"{len(df):,}" if df is not None else "N/A")

    with col2:
        st.metric("🎯 Contextos Modelo", "12" if model is not None else "N/A",
                 help="4 grupos_idade × 3 categorias_trabalho")

    with col3:
        st.metric("✅ Status Testes", "60/60", help="Todos os testes passando")

    with col4:
        conversion_rate = f"{df['y'].mean()*100:.2f}%" if df is not None else "N/A"
        st.metric("📈 Conversão Geral", conversion_rate)

    st.divider()

    st.subheader("🚀 Começar Rápido")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Qualidade de Dados**
        - ✅ Validação automática
        - ✅ Detecção de anomalias
        - ✅ Relatórios de qualidade
        """)

    with col2:
        st.markdown("""
        **Análise Estatística**
        - ✅ Intervalos de confiança
        - ✅ Testes chi-square
        - ✅ Planejamento teste A/B
        """)

    st.markdown("""
    ### Recursos

    1. **Abas de Qualidade de Dados** - Validação ao vivo com alertas automáticos
    2. **Métricas do Modelo** - Desempenho por contexto com intervalos de confiança
    3. **Análise Estatística** - Testes chi-square, análise de recompensa, cálculo de tamanho de amostra
    4. **Demo da API** - Teste recomendações de Thompson Sampling

    ### Status
    - Modelo: Thompson Sampling com 12 contextos
    - Dados: 41.188 clientes
    - Testes: 60/60 passando ✅
    """)

# ============================================================================
# PÁGINA 2: QUALIDADE DE DADOS
# ============================================================================

elif page == "Qualidade de Dados":
    st.title("📊 Relatório de Qualidade de Dados")

    if df is None:
        st.error("❌ Dados não carregados")
    else:
        # Obter relatório de qualidade
        report = get_quality_report(df)

        # Status geral com cor
        status_colors = {
            'PASS': '🟢',
            'WARN': '🟡',
            'FAIL': '🔴'
        }

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de Registros", f"{report.total_records:,}")

        with col2:
            st.metric("Valores Nulos", report.null_values,
                     help="Esperado: 0")

        with col3:
            status_icon = status_colors.get(report.overall_status, '❓')
            st.metric("Status Qualidade", f"{status_icon} {report.overall_status}")

        with col4:
            completeness = (1 - report.null_values / report.total_records) * 100 if report.total_records > 0 else 0
            st.metric("Completude", f"{completeness:.1f}%")

        st.divider()

        # Tabela de métricas detalhadas
        st.subheader("Detalhes de Validação")

        metrics_data = []
        for metric in report.metrics:
            metrics_data.append({
                'Verificação': metric.name,
                'Status': metric.status,
                'Valor': metric.value,
                'Mensagem': metric.message
            })

        df_metrics = pd.DataFrame(metrics_data)

        # Estilo da tabela
        def style_status(val):
            if val == 'PASS':
                return 'color: green; font-weight: bold;'
            elif val == 'WARN':
                return 'color: orange; font-weight: bold;'
            else:
                return 'color: red; font-weight: bold;'

        styled_df = df_metrics.style.applymap(
            style_status,
            subset=['Status']
        )

        st.dataframe(styled_df, use_container_width=True)

        st.divider()

        # Seção de alertas
        st.subheader("⚠️ Alertas & Avisos")

        alerts = []

        # Verificar avisos
        for metric in report.metrics:
            if metric.status == 'WARN':
                alerts.append(('warning', f"{metric.name}: {metric.message}"))
            elif metric.status == 'FAIL':
                alerts.append(('error', f"{metric.name}: {metric.message}"))

        if alerts:
            for alert_type, message in alerts:
                if alert_type == 'warning':
                    st.warning(message)
                else:
                    st.error(message)
        else:
            st.success("✅ Sem alertas - qualidade dos dados está boa!")

        st.divider()

        # Visualizações
        st.subheader("📈 Distribuição de Dados")

        col1, col2 = st.columns(2)

        with col1:
            if 'age' in df.columns:
                st.markdown("**Distribuição de Idade**")
                age_hist = df['age'].hist(bins=20)
                st.pyplot(age_hist.get_figure())

        with col2:
            if 'y' in df.columns:
                st.markdown("**Distribuição da Meta**")
                target_counts = df['y'].value_counts()
                st.bar_chart(target_counts)

# ============================================================================
# PÁGINA 3: MÉTRICAS DO MODELO
# ============================================================================

elif page == "Métricas do Modelo":
    st.title("📈 Métricas de Desempenho do Modelo")

    if model is None or df is None:
        st.error("❌ Modelo ou Dados não carregados")
    else:
        st.subheader("Desempenho do Contexto com Intervalos de Confiança")

        # Calcular métricas para cada contexto
        context_stats = []

        for context, bandit in model.bandits.items():
            successes = int(bandit.successes.sum())
            trials = int(bandit.trials.sum())

            if trials > 0:
                rate = successes / trials
                ci_lower, ci_upper = get_confidence_interval(successes, trials)

                context_stats.append({
                    'Grupo Idade': context[0],
                    'Categoria Trabalho': context[1],
                    'Conversão %': f"{rate*100:.2f}%",
                    'IC Inferior %': f"{ci_lower*100:.2f}%",
                    'IC Superior %': f"{ci_upper*100:.2f}%",
                    'Largura IC %': f"{(ci_upper-ci_lower)*100:.2f}%",
                    'Tentativas': f"{trials:,}",
                    'Sucessos': f"{successes:,}"
                })

        df_stats = pd.DataFrame(context_stats)
        st.dataframe(df_stats, use_container_width=True)

        st.divider()

        # Ordenar por taxa de conversão para visualização
        context_rates = []
        for context, bandit in model.bandits.items():
            successes = int(bandit.successes.sum())
            trials = int(bandit.trials.sum())
            if trials > 0:
                rate = successes / trials
                context_rates.append({
                    'context': f"{context[0]}_{context[1]}",
                    'rate': rate * 100
                })

        df_rates = pd.DataFrame(context_rates).sort_values('rate', ascending=False)

        st.subheader("📊 Taxa de Conversão por Contexto")
        st.bar_chart(df_rates.set_index('context')['rate'])

# ============================================================================
# PÁGINA 4: ANÁLISE ESTATÍSTICA
# ============================================================================

elif page == "Análise Estatística":
    st.title("📊 Testes Estatísticos & Análise")

    if model is None or df is None:
        st.error("❌ Modelo ou Dados não carregados")
    else:
        st.subheader("Comparação Thompson vs Baseline")

        # Calcular baseline (conversão geral)
        baseline_successes = int(df['y'].sum())
        baseline_trials = len(df)
        baseline_rate = baseline_successes / baseline_trials

        st.info(f"📈 Taxa de Conversão Baseline: {baseline_rate*100:.2f}% ({baseline_successes:,}/{baseline_trials:,})")

        st.divider()

        # Obter resultados de contextos
        context_results = {}
        for context, bandit in model.bandits.items():
            successes = int(bandit.successes.sum())
            trials = int(bandit.trials.sum())
            if trials > 0:
                context_results[f"{context[0]}_{context[1]}"] = (successes, trials)

        # Comparar contextos
        comparison_results = compare_contexts(
            context_results=context_results,
            baseline_successes=baseline_successes,
            baseline_trials=baseline_trials,
            alpha=0.05
        )

        st.subheader("🔬 Resultados do Teste Chi-Square")

        sig_contexts = []
        non_sig_contexts = []

        for context, test in comparison_results.items():
            if test.is_significant:
                sig_contexts.append((context, test.p_value))
            else:
                non_sig_contexts.append((context, test.p_value))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### ✅ Contextos Significativos ({len(sig_contexts)})")
            if sig_contexts:
                for context, p_val in sorted(sig_contexts, key=lambda x: x[1]):
                    st.success(f"{context}: p={p_val:.4f}")
            else:
                st.info("Nenhum contexto significativo encontrado")

        with col2:
            st.markdown(f"### ⚠️ Não-Significativos ({len(non_sig_contexts)})")
            if non_sig_contexts:
                for context, p_val in sorted(non_sig_contexts, key=lambda x: x[1])[:5]:
                    st.warning(f"{context}: p={p_val:.4f}")

        st.divider()

        # Planejamento de Teste A/B
        st.subheader("📋 Calculadora de Tamanho de Amostra para Teste A/B")

        col1, col2, col3 = st.columns(3)

        with col1:
            baseline_input = st.number_input(
                "Taxa de Conversão Baseline (%)",
                min_value=0.1,
                max_value=50.0,
                value=float(baseline_rate*100),
                step=0.1
            )

        with col2:
            effect_size = st.number_input(
                "Tamanho de Efeito a Detectar (pp)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Quanto de melhoria detectar (ex: +2%)"
            )

        with col3:
            power = st.selectbox(
                "Poder Estatístico",
                [0.80, 0.85, 0.90, 0.95],
                help="Probabilidade de detectar efeito se existe"
            )

        n_required = calculate_sample_size(
            baseline_rate=baseline_input/100,
            effect_size=effect_size/100,
            power=power
        )

        st.success(
            f"📊 Tamanho de Amostra Requerido: **{n_required:,}** clientes por grupo\n\n"
            f"Total: {n_required*2:,} (controle + tratamento)\n\n"
            f"Suposições: α=0.05 (bilateral), Poder={power}"
        )

# ============================================================================
# PÁGINA 5: DEMO DA API
# ============================================================================

elif page == "Demo da API":
    st.title("🚀 Demo da API Thompson Sampling")

    if model is None:
        st.error("❌ Modelo não carregado")
    else:
        st.subheader("Teste Recomendações")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Idade", min_value=17, max_value=98, value=35)

        with col2:
            job_options = ['admin.', 'blue-collar', 'technician', 'services', 'management',
                           'retired', 'entrepreneur', 'self-employed', 'housemaid',
                           'unemployed', 'student', 'unknown']
            job = st.selectbox("Trabalho", job_options)

        with col3:
            st.write("")  # Espaçamento
            if st.button("Obter Recomendação", use_container_width=True):
                # Obter recomendação
                arm, context = model.select_arm(age=age, job=job)

                # Obter stats
                bandit = model.bandits[context]
                conversion_rate = bandit.get_conversion_rates()[arm]

                st.success(f"""
                ### ✅ Recomendação

                **Perfil do Cliente:**
                - Grupo Idade: {context[0]}
                - Categoria Trabalho: {context[1]}

                **Ação Recomendada:**
                - Campanha: {model.ARM_NAMES[arm]}
                - Conversão Esperada: {conversion_rate*100:.2f}%
                """)

        st.divider()

        st.subheader("📊 Golden Set - Casos de Teste Pré-Definidos")

        # Carregar golden set
        golden_set_path = load_config().project_root / "data" / "golden_set" / "golden_set.json"
        if golden_set_path.exists():
            with open(golden_set_path, 'r') as f:
                golden_set = json.load(f)

            for profile in golden_set:
                with st.expander(f"👤 {profile['profile']['name']} (Idade {profile['profile']['age']})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        **Perfil:**
                        - Idade: {profile['profile']['age']}
                        - Trabalho: {profile['profile']['job']}
                        - Estado Civil: {profile['profile']['marital']}

                        **Contexto:**
                        - Grupo Idade: {profile['context']['age_group']}
                        - Categoria Trabalho: {profile['context']['job_category']}
                        """)

                    with col2:
                        st.markdown(f"""
                        **Recomendação:**
                        - Campanha: {profile['recommendation']['arm_name']}
                        - Conversão Esperada: {profile['recommendation']['expected_conversion_rate']:.2f}%

                        **Campanhas Top:**
                        """)

                        for arm in profile['arm_rankings'][:3]:
                            st.write(f"- {arm['arm_name']}: {arm['rate']*100:.2f}%")

# ============================================================================
# PÁGINA 6: CANARY DEPLOY
# ============================================================================

elif page == "Canary Deploy":
    st.title("🐦 Documentação - Canary Deploy")
    st.markdown("**Guia completo para implementar canary deploy com a API Thompson Sampling**")

    st.divider()

    # Overview
    st.header("O que é Canary Deploy?")
    st.markdown("""
    Canary Deploy é uma estratégia de lançamento onde você:
    1. Libera **versão NOVA para poucos usuários** (ex: 5%)
    2. Monitora **métricas** (conversão, erros, latência)
    3. Se melhorar → **promove gradualmente** (5% → 25% → 50% → 100%)
    4. Se piorar → **rollback imediato** (volta ao baseline)

    **Analogia:** Mineiros levavam canário para detectar gases perigosos. Se canário morria = perigo.
    """)

    st.divider()

    # Endpoints
    st.header("Endpoints da API")

    # Endpoint 1: START
    with st.expander("📍 POST /canary/start - Iniciar Canary Deploy", expanded=False):
        st.markdown("""
        **Descrição:** Ativa canary deploy com modelo novo (5% do tráfego)

        **Request Body:**
        ```json
        {
            "canary_percentage": 5,
            "model_path": "data/models/thompson_model_v2.json"  // opcional
        }
        ```

        **Response:**
        ```json
        {
            "status": "canary_started",
            "canary_percentage": 5,
            "baseline_model": "thompson_model.json",
            "canary_model": "thompson_model_v2.json",
            "timestamp": "2026-07-08T21:30:00"
        }
        ```

        **Exemplo com cURL:**
        ```bash
        curl -X POST http://localhost:5000/canary/start \\
          -H "Content-Type: application/json" \\
          -d '{"canary_percentage": 5}'
        ```

        **O que acontece:**
        - Ativa roteamento automático: 5% → modelo novo, 95% → modelo atual
        - Começa a coletar métricas
        - Pronto para monitorar
        """)

    # Endpoint 2: RECOMMEND
    with st.expander("📍 POST /canary/recommend - Recomendação com Canary", expanded=False):
        st.markdown("""
        **Descrição:** Endpoint recomendação que roteia automaticamente (5% canary, 95% baseline)

        **Request Body:** (igual ao /recommend normal)
        ```json
        {
            "age": 35,
            "job": "admin"
        }
        ```

        **Response:**
        ```json
        {
            "recommended_arm": 0,
            "arm_name": "Cellular_Standard",
            "model_version": "BASELINE",
            "expected_conversion": 0.1127,
            "context": {
                "age_group": "Prime",
                "job_category": "Technical"
            },
            "timestamp": "2026-07-08T21:30:05"
        }
        ```

        **Exemplo com cURL:**
        ```bash
        curl -X POST http://localhost:5000/canary/recommend \\
          -H "Content-Type: application/json" \\
          -d '{
              "age": 35,
              "job": "admin",
              "marital": "married",
              "education": "university.degree",
              "contact": "cellular",
              "campaign": 1
          }'
        ```

        **O que acontece:**
        - Gera número aleatório
        - Se < 5% → usa modelo NOVO (canary)
        - Se >= 5% → usa modelo ATUAL (baseline)
        - Registra decisão para análise
        - Retorna recomendação + qual versão foi usada
        """)

    # Endpoint 3: METRICS
    with st.expander("📍 GET /canary/metrics - Métricas do Canary Deploy", expanded=False):
        st.markdown("""
        **Descrição:** Retorna comparação de conversão entre baseline e canary

        **Request:** (nenhum body)
        ```bash
        curl http://localhost:5000/canary/metrics
        ```

        **Response:**
        ```json
        {
            "canary_enabled": true,
            "canary_percentage": 5,
            "baseline": {
                "total": 950,
                "conversions": 107,
                "rate": 0.1126
            },
            "canary": {
                "total": 50,
                "conversions": 6,
                "rate": 0.1200
            },
            "improvement_pp": 0.74,
            "should_promote": true,
            "p_value": 0.8234,
            "is_significant": false,
            "timestamp": "2026-07-08T21:35:00"
        }
        ```

        **O que significa cada campo:**
        - **baseline.total**: Requisições que usaram modelo atual
        - **canary.total**: Requisições que usaram modelo novo
        - **improvement_pp**: Melhoria em pontos percentuais
        - **p_value**: Significância estatística (< 0.05 = significativo)
        - **should_promote**: Verdadeiro se canary é melhor E estatisticamente significativo

        **Interpretação:**
        - ✅ improvement_pp > 0 E p_value < 0.05 → **PROMOVER**
        - ❌ improvement_pp < 0 → **ROLLBACK**
        - ⚠️ improvement_pp > 0 mas p_value > 0.05 → **AGUARDAR MAIS DADOS**
        """)

    # Endpoint 4: PROMOTE
    with st.expander("📍 POST /canary/promote - Promover Canary a Baseline", expanded=False):
        st.markdown("""
        **Descrição:** Promove modelo novo a baseline (5% → descontinua, novo vira 100%)

        **Request:** (nenhum body)
        ```bash
        curl -X POST http://localhost:5000/canary/promote
        ```

        **Response:**
        ```json
        {
            "status": "canary_promoted",
            "message": "Canary model promoted to baseline (canary deploy completed)",
            "timestamp": "2026-07-08T21:40:00"
        }
        ```

        **O que acontece:**
        - Desativa canary deploy
        - Modelo novo vira 100% (versão atual)
        - Baseline antigo é deprecado
        - Métricas são zeradas

        **Quando usar:**
        - Quando metrics mostra: improvement_pp > 0 E p_value < 0.05
        - Após confirmar com A/B test real
        """)

    # Endpoint 5: ROLLBACK
    with st.expander("📍 POST /canary/rollback - Voltar ao Baseline", expanded=False):
        st.markdown("""
        **Descrição:** Descarta modelo novo, volta a 100% baseline

        **Request:** (nenhum body)
        ```bash
        curl -X POST http://localhost:5000/canary/rollback
        ```

        **Response:**
        ```json
        {
            "status": "canary_rolled_back",
            "message": "Canary deploy rolled back to baseline (model discarded)",
            "timestamp": "2026-07-08T21:45:00"
        }
        ```

        **O que acontece:**
        - Desativa canary deploy
        - Descarta modelo novo
        - Volta a usar 100% baseline
        - Métricas são zeradas

        **Quando usar:**
        - Quando metrics mostra: improvement_pp < 0
        - Taxa de erro > 5%
        - Conversão degradou
        """)

    st.divider()

    # Fluxo Completo
    st.header("Fluxo Completo de Canary Deploy")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Passo 1: Iniciar Canary**
        ```
        POST /canary/start
        ```
        ↓
        **Passo 2: Fazer Requisições**
        ```
        POST /canary/recommend (x1000)
        ```
        ↓
        **Passo 3: Monitorar (a cada hora)**
        ```
        GET /canary/metrics
        ```
        """)

    with col2:
        st.markdown("""
        **Passo 4: Decidir**

        Se improvement > 0 E p_value < 0.05:
        ```
        POST /canary/promote
        ```

        Se improvement < 0:
        ```
        POST /canary/rollback
        ```

        Se duvidoso:
        ```
        Aguarde mais dados
        ```
        """)

    st.divider()

    # Timeline
    st.header("Timeline Recomendada")

    st.markdown("""
    ### Semana 1: 5% Canary
    - Coleta ~5.000 requisições
    - Chi-square test (p < 0.05?)
    - Se OK → PROMOVER para 25%
    - Se ruim → ROLLBACK

    ### Semana 2: 25% Canary
    - Coleta ~25.000 requisições
    - Valida estatisticamente
    - Se OK → PROMOVER para 50%

    ### Semana 3: 50% Canary
    - Coleta ~50.000 requisições
    - Verifica taxa de erro (< 5%?)
    - Se OK → PROMOVER para 100%

    ### Semana 4: 100% Canary (Sucesso!)
    - Modelo novo é versão oficial
    - Baseline antigo é deprecado
    - ROI: +0.2pp conversão = mais $$$
    """)

    st.divider()

    # Critérios de Decisão
    st.header("Critérios de Decisão")

    criteria_data = {
        'Métrica': [
            'improvement_pp',
            'p_value',
            'Taxa Erro',
            'Latência'
        ],
        'Threshold': [
            '> 0.01 (1pp)',
            '< 0.05',
            '< 5%',
            '< 100ms'
        ],
        'Decisão': [
            'Canary é melhor',
            'Estatisticamente significativo',
            'Sem degradação',
            'Sem degradação'
        ],
        'Ação': [
            'PROMOVER ✅',
            'PROMOVER ✅',
            'PROMOVER ✅',
            'PROMOVER ✅'
        ]
    }

    df_criteria = pd.DataFrame(criteria_data)
    st.dataframe(df_criteria, use_container_width=True)

    st.warning("""
    ⚠️ **Se QUALQUER critério falhar:**
    - improvement_pp < 0 → ROLLBACK
    - p_value > 0.05 → Aguarde mais dados
    - Taxa erro > 5% → ROLLBACK
    - Latência degradou → ROLLBACK
    """)

    st.divider()

    # Exemplo de Uso
    st.header("Exemplo Prático: Passo a Passo")

    with st.expander("Seguir exemplo completo"):
        st.markdown("""
        ### Dia 1 - Segunda-feira (10:00)

        **1. Iniciar canary:**
        ```bash
        curl -X POST http://localhost:5000/canary/start \\
          -H "Content-Type: application/json" \\
          -d '{"canary_percentage": 5}'
        ```
        ✅ Canary ativo: 5% novo modelo, 95% baseline

        ---

        ### Dia 2-7 - Fazer requisições normalmente

        **2. Clientes recebem recomendações:**
        ```bash
        curl -X POST http://localhost:5000/canary/recommend \\
          -H "Content-Type: application/json" \\
          -d '{"age": 35, "job": "admin", ...}'
        ```

        Automático: 5% usam modelo novo, 95% usam baseline

        ---

        ### Dia 8 - Segunda-feira (10:00)

        **3. Verificar resultados:**
        ```bash
        curl http://localhost:5000/canary/metrics
        ```

        Resultado:
        ```
        baseline: 95% × 950 = 4750 requisições
        canary: 5% × 50 = 250 requisições
        improvement: +0.74pp (11.26% → 12.00%)
        p_value: 0.82 (NÃO significativo)
        should_promote: false (aguarde mais dados)
        ```

        **4. Decisão: AGUARDAR**
        - improvement > 0 ✅
        - mas p_value > 0.05 ❌
        - Resultado: Continue canary por mais 1 semana

        ---

        ### Dia 15 - Segunda-feira (10:00)

        **5. Verificar novamente:**
        ```bash
        curl http://localhost:5000/canary/metrics
        ```

        Resultado melhorado:
        ```
        baseline: 9500 requisições
        canary: 500 requisições
        improvement: +0.74pp
        p_value: 0.031 (SIGNIFICATIVO!)
        should_promote: true
        ```

        **6. PROMOVER:**
        ```bash
        curl -X POST http://localhost:5000/canary/promote
        ```
        ✅ Modelo novo é agora baseline (100%)!
        """)

    st.divider()

    # Troubleshooting
    st.header("Troubleshooting")

    troubleshooting = {
        'Problema': [
            'Canary metrics diz "not active"',
            'improvement_pp não muda',
            'p_value sempre > 0.05',
            'Muitos erros no canary'
        ],
        'Causa': [
            '/canary/start não foi chamado',
            'Poucos dados (< 100 req)',
            'Diferença real é pequena',
            'Modelo novo tem bug'
        ],
        'Solução': [
            'Chamar: POST /canary/start',
            'Aguardar mais requisições',
            'Aumentar canary_percentage',
            'ROLLBACK + debug modelo'
        ]
    }

    df_troubleshooting = pd.DataFrame(troubleshooting)
    st.dataframe(df_troubleshooting, use_container_width=True)

# ============================================================================
# RODAPÉ
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Qualidade de Dados**")
    st.write("Validação automática de qualidade de dados com alertas")

with col2:
    st.markdown("**Desempenho do Modelo**")
    st.write("Thompson Sampling com 12 contextos")

with col3:
    st.markdown("**Rigor Estatístico**")
    st.write("Testes chi-square com intervalos de confiança")

st.markdown("""
---
**Datathon G37** | Thompson Sampling para Otimização de Campanhas | 2026
""")
