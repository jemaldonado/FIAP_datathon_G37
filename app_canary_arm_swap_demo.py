#!/usr/bin/env python
"""
Canary Deployment Demo - Real Arm Swap Case (Young_Technical)

Focused, visual walkthrough of a single real finding: comparing a snapshot
trained on the first 70% of contacts in real chronological order (the
"baseline") with the model retrained on ALL available data (the
"candidate"), the Thompson Sampling model changes its favorite offer for
the Young_Technical customer segment (age 17-29, job in admin./blue-collar/
technician/services).

    baseline (70% cronológico): Email_Campaign    (~10.72% expected conversion)
    candidato (100%):           Cellular_Standard  (~19.66% expected conversion)

The bank-marketing dataset is ordered by real contact date, so this is a
genuine "what the model knew earlier vs. after more data" comparison — not
a random resample of the same rows (see audit finding 2026-08-05: an
earlier version of this demo used a random 70% subsample, which is
resampling noise, not new data).

Run:
    python src/datathon/api/app.py      # terminal 1
    streamlit run app_canary_arm_swap_demo.py   # terminal 2
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from datathon.bandit import BetaBernoulliBandit, ContextualThompsonSampling

# ==============================================================================
# CONFIG & INIT
# ==============================================================================

st.set_page_config(
    page_title="Arm Swap Demo - Young_Technical",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:5000"
REPO_ROOT = Path(__file__).parent
CONTEXT = ("Young", "Technical")
BASELINE_MODEL_PATH = "data/models/thompson_model_baseline_temporal.json"

# Real customer profile that maps to the Young_Technical context:
# age_group='Young' -> 17 <= age < 30, job_category='Technical' -> job in
# ['admin', 'technician', 'blue-collar', 'services'].
CUSTOMER = {
    "age": 28,
    "job": "admin",
    "marital": "single",
    "education": "university.degree",
    "contact": "cellular",
    "campaign": 1,
}

if 'canary_active' not in st.session_state:
    st.session_state.canary_active = False
if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None

# ==============================================================================
# LOCAL MODEL LOADING (no API needed for the static comparison)
# ==============================================================================


def load_model(path: Path) -> ContextualThompsonSampling:
    with open(path, "r") as f:
        model_data = json.load(f)

    model = ContextualThompsonSampling()
    for data in model_data.values():
        context = tuple(data["context"])
        bandit = BetaBernoulliBandit(
            n_arms=4, alpha_init=data["alpha"], beta_init=data["beta"]
        )
        bandit.trials = np.array(data["trials"], dtype=float)
        bandit.successes = np.array(data["successes"], dtype=float)
        model.bandits[context] = bandit
    return model


def best_arm(model: ContextualThompsonSampling, context):
    bandit = model.bandits[context]
    rates = bandit.get_conversion_rates()
    arm_id = int(np.argmax(rates))
    return arm_id, model.ARM_NAMES[arm_id], float(rates[arm_id])


@st.cache_data
def static_comparison():
    baseline = load_model(REPO_ROOT / BASELINE_MODEL_PATH)
    canary = load_model(REPO_ROOT / "data" / "models" / "thompson_model_v2.json")

    b_id, b_name, b_rate = best_arm(baseline, CONTEXT)
    c_id, c_name, c_rate = best_arm(canary, CONTEXT)

    return {
        "baseline_arm": b_name, "baseline_rate": b_rate,
        "canary_arm": c_name, "canary_rate": c_rate,
        "swapped": b_id != c_id,
    }


# ==============================================================================
# HEADER
# ==============================================================================

st.markdown("# 🔀 Canary Deployment — A Real Arm Swap")
st.markdown(
    "**Segmento `Young_Technical`** (idade 17-29, cargo admin./blue-collar/technician/"
    "services): o retreino muda a oferta vencedora. Baseline = primeiros 70% dos "
    "contatos em ordem cronológica real; candidato = todos os dados disponíveis."
)

comparison = static_comparison()

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    st.metric("🎯 Baseline recomenda", comparison["baseline_arm"],
               f"{comparison['baseline_rate']*100:.2f}% conversão esperada")
with col2:
    st.markdown("<h1 style='text-align:center;margin-top:20px;'>→</h1>", unsafe_allow_html=True)
with col3:
    st.metric("🐦 Canary (v2) recomenda", comparison["canary_arm"],
               f"{comparison['canary_rate']*100:.2f}% conversão esperada")

if comparison["swapped"]:
    st.success(
        f"✅ **Troca confirmada**: o braço vencedor muda de "
        f"**{comparison['baseline_arm']}** para **{comparison['canary_arm']}** "
        f"depois do retreino — com dado real e ordem cronológica real, não reamostragem "
        f"embaralhada."
    )
else:
    st.warning("Os modelos carregados não divergem para este contexto.")

st.markdown("---")

# ==============================================================================
# SIDEBAR - CONTROLS
# ==============================================================================

with st.sidebar:
    st.markdown("## 🎮 Controles")
    st.markdown(f"**Cliente fixo desta demo:**")
    st.json(CUSTOMER)

    canary_pct = st.slider("% do tráfego para o canary (nesta demo):", 10, 90, 50, 10)

    if st.button("▶️ Iniciar Canary", use_container_width=True):
        try:
            response = requests.post(
                f"{API_URL}/canary/start",
                json={
                    "canary_percentage": canary_pct,
                    "model_path": "data/models/thompson_model_v2.json",
                    "baseline_model_path": BASELINE_MODEL_PATH,
                },
                timeout=10,
            )
            if response.status_code == 200:
                st.session_state.canary_active = True
                st.success("Canary iniciado!")
            else:
                st.error(f"Erro: {response.status_code}")
        except Exception as e:
            st.error(f"Não foi possível conectar à API: {e}")

    n_requests = st.slider("Número de requisições:", 50, 1000, 300, 50)

    if st.button("🔄 Rodar Simulação", use_container_width=True):
        if not st.session_state.canary_active:
            st.warning("Inicie o canary primeiro!")
        else:
            with st.spinner(f"Enviando {n_requests} requisições do mesmo cliente..."):
                results = {"BASELINE": Counter(), "CANARY": Counter()}
                errors = 0
                for _ in range(n_requests):
                    try:
                        r = requests.post(f"{API_URL}/canary/recommend", json=CUSTOMER, timeout=10)
                        r.raise_for_status()
                        data = r.json()
                        results[data["model_version"]][data["arm_name"]] += 1
                    except Exception:
                        errors += 1
                st.session_state.sim_results = results
                st.success(f"Simulação completa! {errors} erro(s).")
                st.rerun()

    if st.button("⏹️ Reset", use_container_width=True):
        st.session_state.canary_active = False
        st.session_state.sim_results = None
        st.rerun()

# ==============================================================================
# MAIN CONTENT - LIVE SIMULATION RESULTS
# ==============================================================================

st.markdown("### 📊 Prova ao vivo: mesmo cliente, oferta diferente por versão do modelo")

if not st.session_state.canary_active:
    st.info("👈 Clique **▶️ Iniciar Canary** na barra lateral, depois **🔄 Rodar Simulação**.")
elif st.session_state.sim_results is None:
    st.info("👈 Canary ativo. Clique **🔄 Rodar Simulação** para enviar requisições reais.")
else:
    results = st.session_state.sim_results
    all_arms = sorted({arm for counts in results.values() for arm in counts})

    col1, col2 = st.columns(2)
    with col1:
        baseline_total = sum(results["BASELINE"].values())
        st.metric("Requisições BASELINE", baseline_total)
    with col2:
        canary_total = sum(results["CANARY"].values())
        st.metric("Requisições CANARY", canary_total)

    fig = go.Figure(data=[
        go.Bar(
            name='Baseline',
            x=all_arms,
            y=[results["BASELINE"].get(arm, 0) for arm in all_arms],
            marker_color='#1f77b4'
        ),
        go.Bar(
            name='Canary',
            x=all_arms,
            y=[results["CANARY"].get(arm, 0) for arm in all_arms],
            marker_color='#ff7f0e'
        )
    ])
    fig.update_layout(
        title="Oferta recomendada, por versão do modelo (mesmo cliente em todas as chamadas)",
        barmode='group',
        height=450,
        yaxis_title="Nº de requisições",
        xaxis_title="Oferta recomendada",
    )
    st.plotly_chart(fig, use_container_width=True)

    baseline_top = results["BASELINE"].most_common(1)
    canary_top = results["CANARY"].most_common(1)
    if baseline_top and canary_top:
        b_arm, b_count = baseline_top[0]
        c_arm, c_count = canary_top[0]
        b_pct = b_count / sum(results["BASELINE"].values()) * 100
        c_pct = c_count / sum(results["CANARY"].values()) * 100
        if b_arm != c_arm:
            st.success(
                f"✅ **Troca confirmada ao vivo**: BASELINE recomenda majoritariamente "
                f"**{b_arm}** ({b_pct:.0f}% das vezes), CANARY recomenda majoritariamente "
                f"**{c_arm}** ({c_pct:.0f}% das vezes) — para o **mesmo cliente**."
            )
        else:
            st.warning(
                "Não houve troca de maioria nesta rodada — Thompson Sampling é "
                "estocástico quando os braços estão próximos. Tente aumentar o "
                "número de requisições."
            )

st.markdown("---")

# ==============================================================================
# EXPLANATION
# ==============================================================================

st.markdown("""
### 💡 Por que isso acontece

O dataset bank-marketing é ordenado pela data real de contato. O "baseline" desta demo é
literalmente o que o modelo teria aprendido num ponto anterior da campanha (primeiros 70% dos
contatos); o "candidato" incorpora o resto da campanha, incluindo um período de conversão
muito mais alta (a taxa geral do contexto mais que dobra: ~7% → ~14%). Com mais — e mais recente —
evidência real, a oferta que parecia melhor deixa de ser: Cellular_Standard passa de 7.3% pra
19.6%, ultrapassando Email_Campaign, que só vai de 10.6% pra 16.9%.

Isso é o ponto central da demo: **o retreino com dados adicionais desloca a maioria** — o
braço que antes vencia na maior parte das vezes passa a perder, porque a nova evidência real
mudou a taxa de conversão medida de cada oferta para esse segmento de cliente.

**Por que o canary deploy importa aqui:** antes de aplicar essa mudança para 100% dos
clientes `Young_Technical`, o canary expõe a nova versão a uma fração do tráfego e
mede se a mudança realmente melhora a conversão — em vez de confiar cegamente no
retreino.
""")

st.markdown("---")
st.markdown("**Datathon G37** | Thompson Sampling — Canary Arm Swap Case | 2026")
