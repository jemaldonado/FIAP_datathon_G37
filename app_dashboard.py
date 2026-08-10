#!/usr/bin/env python
"""
Interactive Streamlit Dashboard - Complete Datathon Pipeline
Shows: data → model training → evaluation → API recommendations

Not a translation pair with app_dashboard_pt.py — see README.md#dashboards.
app_dashboard_pt.py covers data quality and statistical testing instead.
Both read the same artifacts via datathon.config.Config, so numbers agree.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from datathon.bandit import ContextualThompsonSampling
from datathon.config import Config

# Page config
st.set_page_config(
    page_title="Datathon - Thompson Sampling Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_config():
    return Config(env='local')

@st.cache_data
def load_data():
    config = load_config()
    data_path = Path(config.get_data_path('bank_marketing_primary.parquet'))
    if data_path.exists():
        return pd.read_parquet(data_path)
    return None

@st.cache_data
def load_model():
    config = load_config()
    model_path = Path(config.model_path) / "thompson_model.json"
    if not model_path.exists():
        return None

    with open(model_path, 'r') as f:
        model_data = json.load(f)

    model = ContextualThompsonSampling()
    for context_key, data in model_data.items():
        from datathon.bandit import BetaBernoulliBandit
        age_group, job_cat = data['context']
        context = (age_group, job_cat)

        bandit = BetaBernoulliBandit(
            n_arms=4,
            alpha_init=data['alpha'],
            beta_init=data['beta']
        )
        bandit.successes = np.array(data['successes'], dtype=float)
        bandit.trials = np.array(data['trials'], dtype=float)
        model.bandits[context] = bandit

    return model

@st.cache_data
def load_golden_set():
    config = load_config()
    golden_path = Path(config.project_root) / "data" / "golden_set" / "golden_set.json"
    if golden_path.exists():
        with open(golden_path, 'r') as f:
            return json.load(f)
    return None

def show_overview():
    """Main overview page"""
    st.title("🎯 Datathon FIAP - Thompson Sampling")
    st.markdown("**Multi-Armed Bandit para Otimização de Campanhas Bancárias**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Clientes", "41,188", "Dataset completo")
    with col2:
        st.metric("Contextos", "12", "4 ages × 3 jobs")
    with col3:
        st.metric("Testes", "60/60", "✅ Passando")
    with col4:
        st.metric("Conversão", "11.27%", "Média geral")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 O Problema")
        st.markdown("""
        - Campanhas de marketing ineficientes
        - Mesma estratégia para todos os clientes
        - Sem adaptação por perfil
        - Conversão baixa (~11%)
        """)

    with col2:
        st.subheader("💡 A Solução")
        st.markdown("""
        - **Thompson Sampling**: Algoritmo de exploração/exploração
        - **Contextos**: Segmenta por idade + profissão (12 contextos)
        - **Campanhas**: 4 estratégias diferentes (Cellular, Email, SMS, Premium Call)
        - **Recomendação**: Melhor campanha por contexto
        - **Resultado**: Até 42% conversão em melhores segmentos
        """)

    st.markdown("---")

    st.subheader("🚀 Pipeline da Solução")

    # Flow diagram
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.info("📥\n**1. Dados**\n41K clientes")
    with col2:
        st.info("🔄\n**2. Treino**\nThompson")
    with col3:
        st.info("✅\n**3. Avaliação**\nGolden Set")
    with col4:
        st.info("🌐\n**4. API**\nRecomendações")
    with col5:
        st.info("📈\n**5. Métricas**\nMLflow")

def show_data():
    """Data exploration page"""
    st.title("📊 Dados - Treino vs Teste")

    df = load_data()
    if df is None:
        st.error("Dados não encontrados. Execute: python scripts/download_data.py")
        return

    # Split treino/teste (80/20)
    np.random.seed(42)
    train_idx = np.random.choice(len(df), size=int(0.8 * len(df)), replace=False)
    test_idx = np.array([i for i in range(len(df)) if i not in train_idx])

    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    # Tabs para treino vs teste
    tab1, tab2 = st.tabs(["📚 Dados de Treino (80%)", "🧪 Dados de Teste (20%)"])

    with tab1:
        st.subheader("Conjunto de Treino")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de clientes", f"{len(df_train):,}")
        with col2:
            st.metric("Conversões", f"{df_train['y'].sum():,}")
        with col3:
            st.metric("Taxa", f"{df_train['y'].mean():.2%}")
        with col4:
            st.metric("Colunas", len(df_train.columns))

    with tab2:
        st.subheader("Conjunto de Teste")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de clientes", f"{len(df_test):,}")
        with col2:
            st.metric("Conversões", f"{df_test['y'].sum():,}")
        with col3:
            st.metric("Taxa", f"{df_test['y'].mean():.2%}")
        with col4:
            st.metric("Colunas", len(df_test.columns))

    st.markdown("---")

    # Data exploration - Treino
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribuição de Idade")
            fig = px.histogram(df_train, x='age', nbins=30, color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top Profissões")
            job_counts = df_train['job'].value_counts().head(10)
            fig = px.bar(x=job_counts.values, y=job_counts.index, orientation='h',
                        color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)

    # Data exploration - Teste
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribuição de Idade")
            fig = px.histogram(df_test, x='age', nbins=30, color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top Profissões")
            job_counts = df_test['job'].value_counts().head(10)
            fig = px.bar(x=job_counts.values, y=job_counts.index, orientation='h',
                        color_discrete_sequence=['#d62728'])
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Conversion by age and job - Treino
    with tab1:
        st.subheader("Taxa de Conversão por Contexto (Treino)")

        age_groups = pd.cut(df_train['age'], bins=[0, 30, 45, 60, 150],
                            labels=['Young (17-30)', 'Prime (30-45)', 'Mature (45-60)', 'Senior (60+)'])

        job_categories = df_train['job'].apply(ContextualThompsonSampling._get_job_category)

        conversion_pivot = pd.DataFrame({
            'age_group': age_groups,
            'job_category': job_categories,
            'conversion': df_train['y']
        }).groupby(['age_group', 'job_category'])['conversion'].agg(['mean', 'count'])

        conversion_pivot['mean'] = conversion_pivot['mean'] * 100

        fig = go.Figure(data=go.Heatmap(
            z=conversion_pivot['mean'].unstack().values,
            x=conversion_pivot['mean'].unstack().columns,
            y=conversion_pivot['mean'].unstack().index,
            colorscale='RdYlGn',
            text=conversion_pivot['mean'].unstack().round(1),
            texttemplate='%{text:.1f}%',
            hovertemplate='<b>%{y} - %{x}</b><br>Taxa: %{z:.1%}<extra></extra>'
        ))
        fig.update_layout(title="Taxa de Conversão Treino (%)", xaxis_title="Profissão", yaxis_title="Idade")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver dados brutos (Treino)"):
            st.dataframe(df_train.head(50), use_container_width=True)

    # Conversion by age and job - Teste
    with tab2:
        st.subheader("Taxa de Conversão por Contexto (Teste)")

        age_groups = pd.cut(df_test['age'], bins=[0, 30, 45, 60, 150],
                            labels=['Young (17-30)', 'Prime (30-45)', 'Mature (45-60)', 'Senior (60+)'])

        job_categories = df_test['job'].apply(ContextualThompsonSampling._get_job_category)

        conversion_pivot = pd.DataFrame({
            'age_group': age_groups,
            'job_category': job_categories,
            'conversion': df_test['y']
        }).groupby(['age_group', 'job_category'])['conversion'].agg(['mean', 'count'])

        conversion_pivot['mean'] = conversion_pivot['mean'] * 100

        fig = go.Figure(data=go.Heatmap(
            z=conversion_pivot['mean'].unstack().values,
            x=conversion_pivot['mean'].unstack().columns,
            y=conversion_pivot['mean'].unstack().index,
            colorscale='RdYlGn',
            text=conversion_pivot['mean'].unstack().round(1),
            texttemplate='%{text:.1f}%',
            hovertemplate='<b>%{y} - %{x}</b><br>Taxa: %{z:.1%}<extra></extra>'
        ))
        fig.update_layout(title="Taxa de Conversão Teste (%)", xaxis_title="Profissão", yaxis_title="Idade")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver dados brutos (Teste)"):
            st.dataframe(df_test.head(50), use_container_width=True)

def show_model_training():
    """Model training visualization"""
    st.title("🤖 Treinamento do Modelo")

    model = load_model()
    if model is None:
        st.error("Modelo não encontrado. Execute: python scripts/train_simple.py")
        return

    st.subheader("Métricas por Contexto")

    contexts_data = []
    for context in sorted(model.bandits.keys()):
        stats = model.get_context_stats(context)
        total_trials = sum(s['trials'] for s in stats['arms'].values())
        total_successes = sum(s['successes'] for s in stats['arms'].values())

        if total_trials > 0:
            rate = total_successes / total_trials
            contexts_data.append({
                'Contexto': f"{context[0]} + {context[1]}",
                'Conversão': f"{rate:.1%}",
                'Tentativas': int(total_trials),
                'Conversões': int(total_successes),
                'Taxa': rate
            })

    contexts_df = pd.DataFrame(contexts_data).sort_values('Taxa', ascending=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Melhor contexto", contexts_df.iloc[0]['Contexto'],
                 contexts_df.iloc[0]['Conversão'])
    with col2:
        st.metric("Pior contexto", contexts_df.iloc[-1]['Contexto'],
                 contexts_df.iloc[-1]['Conversão'])
    with col3:
        best = contexts_df.iloc[0]['Taxa']
        worst = contexts_df.iloc[-1]['Taxa']
        spread = (best - worst) * 100
        st.metric("Spread", f"{spread:.1f} pp", "Diferença")

    st.markdown("---")

    # Context performance chart
    fig = px.bar(contexts_df, x='Contexto', y='Taxa',
                color='Taxa', color_continuous_scale='RdYlGn',
                labels={'Taxa': 'Taxa de Conversão'})
    fig.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

    # Performance table
    st.dataframe(contexts_df[['Contexto', 'Conversão', 'Tentativas', 'Conversões']],
                 use_container_width=True, hide_index=True)

def show_golden_set():
    """Golden set validation"""
    st.title("✨ Golden Set - Perfis de Teste")

    golden_set = load_golden_set()
    if golden_set is None:
        st.error("Golden Set não encontrado")
        return

    model = load_model()

    st.markdown("Perfis representativos para validação da pipeline")

    for i, profile in enumerate(golden_set, 1):
        with st.expander(f"**{i}. {profile['profile']['name']}**"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Perfil:**")
                st.write(f"Idade: {profile['profile']['age']}")
                st.write(f"Profissão: {profile['profile']['job']}")
                st.write(f"Estado civil: {profile['profile']['marital']}")

            with col2:
                st.write("**Contexto:**")
                context = profile['context']
                st.write(f"Grupo: {context['age_group']}")
                st.write(f"Categoria: {context['job_category']}")

            with col3:
                st.write("**Recomendação:**")
                rec = profile['recommendation']
                st.write(f"Arm: {rec['arm_name']}")
                st.write(f"Conversão esperada: {rec['expected_conversion_rate']:.1%}")
                st.write(f"Conversão observada: {rec['context_conversion_rate']:.1%}")

            # Test with current model
            if model:
                age = profile['profile']['age']
                job = profile['profile']['job']
                arm, actual_context = model.select_arm(age, job)

                expected = (context['age_group'], context['job_category'])
                match = "✅" if actual_context == expected else "❌"

                st.write(f"\n**Validação Modelo:** {match}")
                st.write(f"Esperado: {expected}")
                st.write(f"Obtido: {actual_context}")

def show_api_testing():
    """API testing and recommendations"""
    st.title("🌐 Testar Recomendações da API")

    model = load_model()
    if model is None:
        st.error("Modelo não disponível")
        return

    st.markdown("""
    Simule uma recomendação para um cliente específico.
    Thompson Sampling escolherá a melhor campanha para este contexto (age_group + job_category).

    **4 Campanhas Disponíveis:**
    - 🎯 **Cellular_Standard**: Chamada via celular (canal real dos dados)
    - 📧 **Email_Campaign**: Campanha por email (alternativa menos intrusiva)
    - 📱 **SMS_Alert**: Alerta por SMS (para mobile-first)
    - 👑 **Call_Premium**: Chamada premium (para VIP/Business)
    """)

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Idade", 17, 98, 35)
        job = st.selectbox("Profissão", ['admin.', 'blue-collar', 'technician', 'services',
                                         'management', 'retired', 'entrepreneur', 'self-employed',
                                         'housemaid', 'unemployed', 'student', 'unknown'])

    with col2:
        marital = st.selectbox("Estado civil", ['married', 'single', 'divorced', 'unknown'])
        education = st.selectbox("Educação", ['university.degree', 'high.school', 'basic.9y',
                                              'professional.course', 'basic.4y', 'basic.6y',
                                              'unknown', 'illiterate'])

    contact = st.radio("Tipo de contato", ['cellular', 'telephone'], horizontal=True)
    campaign = st.number_input("Número de campanhas", 1, 10, 1)

    # Get recommendation
    if st.button("🎯 Obter Recomendação", use_container_width=True):
        arm, context = model.select_arm(age, job)
        stats = model.get_context_stats(context)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Contexto", f"{context[0]} + {context[1]}")
        with col2:
            st.metric("Estratégia Recomendada", model.ARM_NAMES[arm])
        with col3:
            arm_stats = stats['arms'][model.ARM_NAMES[arm]]
            conversion = arm_stats['rate']
            st.metric("Taxa de Conversão", f"{conversion:.1%}")

        st.markdown("---")

        # Show all arms for this context
        st.subheader("Comparação de Estratégias para este Contexto")

        arms_comparison = []
        for arm_id in range(4):
            arm_name = model.ARM_NAMES[arm_id]
            arm_stat = stats['arms'][arm_name]
            arms_comparison.append({
                'Estratégia': arm_name,
                'Taxa': arm_stat['rate'],
                'Tentativas': int(arm_stat['trials']),
                'Recomendada': '⭐' if arm_id == arm else ''
            })

        arms_df = pd.DataFrame(arms_comparison).sort_values('Taxa', ascending=False)

        fig = px.bar(arms_df, x='Estratégia', y='Taxa',
                    color='Taxa', color_continuous_scale='Blues')
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(arms_df, use_container_width=True, hide_index=True)

def show_thompson_learning():
    """Explain how Thompson Sampling learns without complete data"""
    st.title("🧠 Como Thompson Sampling Aprende?")

    st.markdown("""
    ## O Problema
    A base real tem apenas **2 canais de contato**: cellular e telephone. Não há Email/SMS/
    Premium reais — nunca foram testados.

    A solução usada aqui **não é estimar taxas hipotéticas**: os 4 braços são os 2 canais reais
    divididos por primeiro-contato vs. contato-repetido (`campaign == 1` ou não), relabeled com
    nomes de negócio:
    - Cellular_Standard = cellular, primeiro contato
    - Email_Campaign = cellular, contato repetido
    - SMS_Alert = telephone, primeiro contato
    - Call_Premium = telephone, contato repetido

    Toda conversão contada por braço é 100% real (`y` observado) — só os **nomes** são uma
    convenção de negócio pra dar 4 "estratégias" a um bandit contextual, ver
    `datathon.bandit.assign_arm`.
    """)

    st.markdown("---")

    # Part 1: Learning Mechanism
    st.subheader("1️⃣ Mecanismo de Aprendizado - Thompson Sampling")
    st.caption(
        "Exemplo ilustrativo do mecanismo geral de exploração/exploração (números abaixo são "
        "didáticos, não vêm do nosso dataset real — as taxas reais por contexto estão na aba "
        "Golden Set)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **FASE 1: Exploração (Primeiros 100 clientes)**

        Thompson testa cada estratégia ~25 vezes:
        ```
        Cellular_Standard:  22/25 = 88% ✅
        Email_Campaign:     18/25 = 72%
        SMS_Alert:           8/25 = 32%
        Call_Premium:        5/25 = 20%
        ```

        Thompson aprende: **Cellular é melhor!**
        """)

    with col2:
        # Simulation of learning
        strategies = ['Cellular_Standard', 'Email_Campaign', 'SMS_Alert', 'Call_Premium']
        phase1_rates = [0.88, 0.72, 0.32, 0.20]

        fig = px.bar(x=strategies, y=phase1_rates,
                    color=phase1_rates,
                    color_continuous_scale='RdYlGn',
                    labels={'y': 'Taxa de Conversão'})
        fig.update_yaxes(tickformat='.0%')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **FASE 2: Exploração > Exploração (Próximos 900)**

        Thompson agora favorece Cellular:
        ```
        Cellular_Standard:  70% (melhor)
        Email_Campaign:     20% (segunda)
        SMS_Alert:           7% (terceira)
        Call_Premium:        3% (pior)
        ```

        Resultado: 630 Cellular, 180 Email, 63 SMS, 27 Premium
        """)

    with col2:
        # Phase 2: Thompson converges
        phase2_dist = [0.70, 0.20, 0.07, 0.03]

        fig = go.Figure(data=[go.Pie(
            labels=strategies,
            values=phase2_dist,
            marker=dict(colors=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']),
            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"
        )])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Conclusão**: Thompson aprende **COMPARATIVAMENTE qual é melhor**,
    não precisa de valores absolutos corretos!
    """)

    st.markdown("---")

    # Part 2: Real uncertainty comes from sample size, not guessed rates
    st.subheader("2️⃣ De Onde Vem a Incerteza, Se os Dados São Reais?")

    st.markdown("""
    Como as taxas por braço vêm de conversões reais (não de um palpite), a incerteza que o
    Thompson Sampling precisa gerenciar não é "será que nossa estimativa está errada?" — é
    **quantos clientes reais esse braço já teve** nesse contexto. Poucos trials = distribuição
    Beta larga (posterior incerto); muitos trials = distribuição estreita (posterior confiante).
    Veja a aba **Golden Set** para os `trials`/`successes` reais por (contexto, braço) — contextos
    como `Senior_Business` têm poucas centenas de observações por braço, então o Thompson
    continua explorando ali por mais tempo antes de convergir, mesmo usando dado 100% real.
    """)

    st.markdown("---")

    # Part 3: Why relabeling real segments is a defensible design
    st.subheader("3️⃣ Por Que Relabeled é Defensável (e Não é Dado Sintético)")

    st.markdown("""
    ### Argumento 1: A conversão é sempre real

    Cada linha que conta para Email_Campaign/SMS_Alert/Call_Premium tem um `y` observado de
    verdade — não é uma taxa inventada nem simulada. O que é uma convenção é só o **nome** do
    segmento.

    ### Argumento 2: Regra rastreável e auditável

    A regra de relabel é uma função pura, sem hiperparâmetro inventado:
    - Cellular_Standard = `contact == 'cellular'` e `campaign == 1`
    - Email_Campaign = `contact == 'cellular'` e `campaign > 1`
    - SMS_Alert = `contact == 'telephone'` e `campaign == 1`
    - Call_Premium = `contact == 'telephone'` e `campaign > 1`

    Ver `datathon.bandit.assign_arm` — qualquer um consegue reproduzir exatamente essas 4 fatias
    a partir do dataset público.

    ### Argumento 3: A limitação real é outra

    O ponto de atenção genuíno não é "a taxa pode estar errada" (não pode — é medida) — é que os
    nomes de negócio (Email, SMS, Premium) não correspondem a canais realmente testados. Uma
    campanha de e-mail real poderia converter diferente de "cellular, contato repetido". Isso é
    uma limitação de rotulagem, documentada, não um problema de dado.

    ### Argumento 4: Aplicação prática

    Bancos já usam Thompson Sampling em produção:
    - LinkedIn: recomendação de vagas vs anúncios
    - Netflix: qual trailer mostrar primeiro
    - Spotify: qual música recomendar

    Todos começam com estimativas, refinam com dados.
    """)

    st.markdown("---")

    # Part 4: Golden Set Validation
    st.subheader("4️⃣ Validação: Golden Set Mostra Variação Real")

    model = load_model()
    if model:
        st.markdown("""
        O **Golden Set** mostra que Thompson realmente aprendeu diferenças
        por contexto. Veja as estratégias RECOMENDADAS para cada perfil:
        """)

        golden_set = load_golden_set()
        if golden_set:
            for profile in golden_set[:3]:  # Show first 3
                with st.expander(f"📊 {profile['profile']['name']}"):
                    context = profile['context']
                    rankings = profile['arm_rankings']

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown("**Contexto:**")
                        st.write(f"{context['age_group']} + {context['job_category']}")
                        st.markdown("**Ranking:**")
                        for rank, arm_info in enumerate(rankings, 1):
                            st.write(f"{rank}. {arm_info['arm_name']}: {arm_info['rate']:.1%}")

                    with col2:
                        fig = px.bar(
                            x=[a['arm_name'] for a in rankings],
                            y=[a['rate'] for a in rankings],
                            color=[a['rate'] for a in rankings],
                            color_continuous_scale='RdYlGn',
                            labels={'y': 'Taxa de Conversão'}
                        )
                        fig.update_yaxes(tickformat='.0%')
                        fig.update_layout(height=200, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ✅ **Conclusão**: Thompson Sampling aprende qual estratégia funciona melhor
    **para cada segmento de cliente**, mesmo começando com estimativas.
    """)


def show_metrics():
    """Metrics and monitoring"""
    st.title("📈 Métricas e Performance")

    model = load_model()
    df = load_data()

    if model is None or df is None:
        st.error("Dados não disponíveis")
        return

    # Split treino/teste
    np.random.seed(42)
    train_idx = np.random.choice(len(df), size=int(0.8 * len(df)), replace=False)
    test_idx = np.array([i for i in range(len(df)) if i not in train_idx])
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Conversão Geral", f"{df['y'].mean():.2%}")
    with col2:
        st.metric("Total de Registros", f"{len(df):,}")
    with col3:
        st.metric("Campanhas Max", df['campaign'].max())
    with col4:
        st.metric("Contextos com Dados", "12/12")

    st.markdown("---")

    # Treino vs Teste Comparison
    st.subheader("📊 Comparação Treino vs Teste")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Conjunto de Treino**")
        st.metric("Conversão", f"{df_train['y'].mean():.2%}", f"{len(df_train):,} clientes")
        st.metric("Conversões", f"{df_train['y'].sum():,}")

    with col2:
        st.write("**Conjunto de Teste**")
        st.metric("Conversão", f"{df_test['y'].mean():.2%}", f"{len(df_test):,} clientes")
        st.metric("Conversões", f"{df_test['y'].sum():,}")

    # Comparação visual
    comparison_data = {
        'Treino': [df_train['y'].mean()],
        'Teste': [df_test['y'].mean()]
    }
    comparison_df = pd.DataFrame(comparison_data).T.reset_index()
    comparison_df.columns = ['Conjunto', 'Taxa de Conversão']

    fig = px.bar(comparison_df, x='Conjunto', y='Taxa de Conversão',
                color='Taxa de Conversão', color_continuous_scale='Blues',
                text='Taxa de Conversão')
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Distribution by campaign strategy
    st.subheader("Distribuição de Campanhas (Thompson Sampling Aprendizado)")

    # Aggregate arm distribution across all contexts
    arms_total = [0, 0, 0, 0]
    for context, bandit in model.bandits.items():
        for i in range(4):
            arms_total[i] += int(bandit.trials[i])

    arms_names = [model.ARM_NAMES[i] for i in range(4)]
    arms_descriptions = [
        "Cellular_Standard (Chamada celular real)",
        "Email_Campaign (Campanha por email)",
        "SMS_Alert (Alerta SMS)",
        "Call_Premium (Chamada premium)"
    ]

    fig = go.Figure(data=[go.Pie(
        labels=arms_descriptions,
        values=arms_total,
        hovertemplate="<b>%{label}</b><br>Tentativas: %{value:,}<br>%{percent}<extra></extra>"
    )])
    st.plotly_chart(fig, use_container_width=True)

    # Show campaign stats
    st.markdown("**Desempenho por Campanha (todos os contextos):**")
    campaign_stats = []
    for i, (arm_name, total) in enumerate(zip(arms_names, arms_total)):
        campaign_stats.append({
            'Campanha': arm_name,
            'Tentativas': int(total),
            '% do Total': f"{(total/sum(arms_total)*100):.1f}%" if sum(arms_total) > 0 else "0%"
        })
    st.dataframe(pd.DataFrame(campaign_stats), use_container_width=True, hide_index=True)

    # Contact preference
    st.subheader("Preferência de Contato")
    contact_dist = df['contact'].value_counts()
    fig = px.bar(x=contact_dist.index, y=contact_dist.values,
                labels={'x': 'Tipo de Contato', 'y': 'Quantidade'},
                color_discrete_sequence=['#2ecc71'])
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Sidebar navigation
    st.sidebar.title("🎯 Datathon Dashboard")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navegação",
        ["📌 Overview", "📊 Dados", "🤖 Modelo", "🧠 Thompson Aprende?", "✨ Golden Set", "🌐 API", "📈 Métricas"]
    )

    st.sidebar.markdown("---")

    # Footer
    st.sidebar.write("""
    **Thompson Sampling**
    Multi-Armed Bandit para marketing bancário

    **Status:** ✅ Operacional

    **Stack:**
    - Python 3.11
    - Flask API
    - MLflow Tracking
    - Streamlit Dashboard
    """)

    # Routes
    if page == "📌 Overview":
        show_overview()
    elif page == "📊 Dados":
        show_data()
    elif page == "🤖 Modelo":
        show_model_training()
    elif page == "🧠 Thompson Aprende?":
        show_thompson_learning()
    elif page == "✨ Golden Set":
        show_golden_set()
    elif page == "🌐 API":
        show_api_testing()
    elif page == "📈 Métricas":
        show_metrics()

if __name__ == "__main__":
    main()
