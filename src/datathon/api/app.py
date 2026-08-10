"""
Datathon API - Contextual Thompson Sampling Recommendation Service

Flask API serving contextual Thompson Sampling bandit recommendations.
- Model: 12 Thompson Sampling bandits (4 age_groups × 3 job_categories)
- Follows fiap_projeto pattern: flat structure, model loaded at startup, Swagger documented
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import sys
import copy

# Add parent directory to path so we can import datathon
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import model classes
from datathon.bandit import ContextualThompsonSampling, assign_arm
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Model cache (loaded once at startup)
MODEL = None
MODEL_METADATA = None

# Segundo modelo carregado sob demanda quando o canary é iniciado
CANARY_MODEL = None

# Baseline alternativo, carregado sob demanda só quando /canary/start
# recebe baseline_model_path — usado por demos que precisam comparar
# contra um snapshot diferente do MODEL de produção sem substituí-lo
# globalmente (ex.: demo_arm_swap_case.py comparando um corte temporal
# mais antigo contra o modelo atual). None = usa MODEL normalmente.
CANARY_BASELINE_MODEL = None

# Taxa de conversão REAL histórica por (age_group, job_category, arm),
# usada como ground truth da simulação do canary (ver load_real_arm_rates)
REAL_ARM_RATES = {}

# Campaign strategy names mapping
ARM_NAMES = {
    0: "Cellular_Standard",
    1: "Email_Campaign",
    2: "SMS_Alert",
    3: "Call_Premium"
}

# Canary Deploy Configuration
CANARY_CONFIG = {
    'enabled': False,
    'canary_percentage': 5,  # 5% do tráfego
    'baseline_model': None,
    'canary_model': None,
    'metrics': {
        'baseline': {'conversions': 0, 'total': 0, 'decisions': []},
        'canary': {'conversions': 0, 'total': 0, 'decisions': []}
    }
}


def load_model():
    """Load Contextual Thompson Sampling model at startup"""
    global MODEL, MODEL_METADATA

    model_path = Path(__file__).parent.parent.parent.parent / "data" / "models" / "thompson_model.json"

    try:
        if model_path.exists():
            # Load model data from JSON
            MODEL = ContextualThompsonSampling()

            with open(model_path, 'r') as f:
                model_data = json.load(f)

            # Reconstruct bandits from saved data
            for context_key, data in model_data.items():
                context = tuple(data['context'])
                bandit = MODEL.bandits[context]
                bandit.alpha = np.array(data['alpha'], dtype=float)
                bandit.beta = np.array(data['beta'], dtype=float)
                bandit.trials = np.array(data['trials'], dtype=float)
                bandit.successes = np.array(data['successes'], dtype=float)

            MODEL_METADATA = {
                "status": "loaded",
                "path": str(model_path),
                "type": "ContextualThompsonSampling",
                "n_arms": 4,
                "n_contexts": 12,
                "dimensions": ["age_group", "job_category"],
                "algorithm": "BetaBernoulliBandit"
            }
            logger.info(f"Contextual Thompson model loaded from {model_path}")
            logger.info(f"Model has {len(MODEL.bandits)} contexts")
        else:
            logger.warning(f"Model file not found at {model_path}. API will return mock recommendations.")
            MODEL = None
            MODEL_METADATA = {
                "status": "not_found",
                "path": str(model_path),
                "note": "Using deterministic fallback predictions"
            }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        MODEL = None
        MODEL_METADATA = {
            "status": "error",
            "error": str(e),
            "note": "Using deterministic fallback predictions"
        }

def load_real_arm_rates():
    """
    Carrega a taxa de conversão REAL histórica por (age_group, job_category, arm)
    a partir dos dados de treino — ground truth fixo e independente de
    qualquer modelo, usado pela simulação do /canary/recommend.

    Sem isso, o endpoint tinha um bug circular (Achado Crítico 3 da auditoria
    2026-08-05): gerava "converteu?" a partir da própria crença do modelo
    (bandit.get_conversion_rates()) e realimentava esse resultado no mesmo
    modelo — ruído auto-referente, não validação contra dado real. Aqui a
    taxa vem de fora do modelo, igual ao padrão já usado em
    compute_baseline_vs_thompson (Etapa 3).
    """
    global REAL_ARM_RATES

    data_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "bank_marketing_primary.parquet"
    if not data_path.exists():
        logger.warning(
            f"Training data not found at {data_path}; canary simulation will "
            "fall back to the model's own posterior (loses the real-data guarantee)."
        )
        return

    df = pd.read_parquet(data_path)
    template = ContextualThompsonSampling()
    df['age_group'] = df['age'].apply(template._get_age_group)
    df['job_category'] = df['job'].apply(template._get_job_category)
    df['arm'] = assign_arm(df)

    rates = df.groupby(['age_group', 'job_category', 'arm'])['y'].mean()
    REAL_ARM_RATES = {
        (age_group, job_category, int(arm)): float(rate)
        for (age_group, job_category, arm), rate in rates.items()
    }
    logger.info(f"Loaded {len(REAL_ARM_RATES)} real (context, arm) conversion rates for canary simulation")


def load_model_from_path(model_path: Path):
    """Load a ContextualThompsonSampling model from an arbitrary JSON path.

    Usado pelo canary deploy para carregar um segundo modelo (versão
    candidata) sem afetar o MODEL de produção.
    """
    model = ContextualThompsonSampling()

    with open(model_path, 'r') as f:
        model_data = json.load(f)

    for context_key, data in model_data.items():
        context = tuple(data['context'])
        bandit = model.bandits[context]
        bandit.alpha = np.array(data['alpha'], dtype=float)
        bandit.beta = np.array(data['beta'], dtype=float)
        bandit.trials = np.array(data['trials'], dtype=float)
        bandit.successes = np.array(data['successes'], dtype=float)

    return model

def validate_recommend_request(data):
    """
    Validate request data for /recommend endpoint.

    Returns: (is_valid, error_message)
    """
    required_fields = ['age', 'job', 'marital', 'education', 'contact', 'campaign']

    if not data:
        return False, "Request body is empty"

    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    # Validate types
    if not isinstance(data['age'], int) or data['age'] < 0 or data['age'] > 150:
        return False, "age must be a positive integer (0-150)"

    if not isinstance(data['campaign'], int) or data['campaign'] < 1:
        return False, "campaign must be a positive integer (>= 1)"

    if data['contact'] not in ['cellular', 'telephone']:
        return False, "contact must be 'cellular' or 'telephone'"

    return True, None

def select_arm(age, job, contact, campaign):
    """
    Select arm using Contextual Thompson Sampling.

    Args:
        age: Customer age
        job: Customer job title
        contact: Contact type ('cellular' or 'telephone')
        campaign: Campaign number

    Returns:
        Tuple: (arm_id, context_tuple, conversion_rate)
    """
    if MODEL is not None:
        try:
            # Use Contextual Thompson Sampling. contact/campaign are NOT
            # used here on purpose: the trained model only needs (age, job)
            # to pick a context, then Thompson Sampling itself decides the
            # arm (that decision is what contact/campaign encoded at
            # TRAINING time, via assign_arm — at inference time for a new
            # customer, which arm to use is the very thing being decided,
            # not an input). They're required/validated above only for the
            # deterministic fallback below and schema consistency.
            arm, context = MODEL.select_arm(age, job)

            # Get expected conversion rate for this context+arm. Uses the
            # raw empirical rate (successes/trials via get_context_stats),
            # not the Bayesian posterior mean (get_conversion_rates(), used
            # for arm selection/comparison elsewhere e.g. compare_models.py)
            # — the two can diverge for low-trial contexts. Reported here
            # as "what we've observed so far", not "what Thompson believes".
            stats = MODEL.get_context_stats(context)
            arm_name = ARM_NAMES[arm]
            expected_conv = stats['arms'][arm_name]['rate']

            return arm, context, expected_conv
        except Exception as e:
            logger.error(f"Error selecting arm with model: {e}")

    # Fallback: deterministic strategy (traditional multi-armed bandit)
    if contact == 'cellular':
        arm = 0 if campaign == 1 else 1
    else:
        arm = 2 if campaign == 1 else 3

    # Job/age category for fallback — reuse the model's own mapping (single
    # source of truth) instead of a hand-copied list.
    job_category = ContextualThompsonSampling._get_job_category(job)
    age_group = ContextualThompsonSampling._get_age_group(age)
    context = (age_group, job_category)
    expected_conv = 0.12

    return arm, context, expected_conv

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = "loaded" if MODEL is not None else "not_loaded"
    return jsonify({
        "status": "ok",
        "model": model_status
    }), 200

@app.route('/apidocs', methods=['GET'])
def swagger_ui():
    """Serve Swagger UI documentation"""
    html = """
<!DOCTYPE html>
<html>
  <head>
    <title>Datathon API - Documentacao Interativa</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css">
    <style>
      html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
      *, *:before, *:after { box-sizing: inherit; }
      body { margin:0; padding:0; }
    </style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-bundle.js"></script>
    <script>
      const ui = SwaggerUIBundle({
        url: "/swagger.json",
        dom_id: '#swagger-ui',
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      })
      window.ui = ui
    </script>
  </body>
</html>
    """
    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/swagger.json', methods=['GET'])
def swagger_spec():
    """Serve Swagger specification as JSON"""
    swagger_path = Path(__file__).parent.parent.parent.parent / "swagger.yaml"

    try:
        with open(swagger_path, 'r') as f:
            spec = yaml.safe_load(f)
        return jsonify(spec), 200
    except Exception as e:
        logger.error(f"Error loading swagger.yaml: {e}")
        return jsonify({"error": "Could not load API specification"}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Get contextual offer recommendation for a customer.

    Uses 12 Thompson Sampling bandits (4 age_groups × 3 job_categories).

    Request body (JSON):
    {
        "age": 35,
        "job": "admin",
        "marital": "married",
        "education": "university.degree",
        "contact": "cellular",
        "campaign": 1
    }

    Response:
    {
        "recommended_arm": 0,
        "arm_name": "Cellular_1x",
        "expected_conversion": 0.145,
        "context": {
            "age_group": "Prime",
            "job_category": "Technical"
        },
        "rationale": "Thompson Sampling for Prime + Technical context recommends..."
    }
    """
    # Get request data
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.warning(f"Invalid JSON in request: {e}")
        return jsonify({"error": "Invalid JSON in request body"}), 400

    # Validate request
    is_valid, error_msg = validate_recommend_request(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        # Extract fields
        age = int(data['age'])
        job = data.get('job', 'unknown').lower()
        contact = data['contact'].lower()
        campaign = int(data['campaign'])

        # Select arm using Contextual Thompson Sampling
        arm_id, context, expected_conv = select_arm(age, job, contact, campaign)
        arm_name = ARM_NAMES[arm_id]
        age_group, job_category = context

        # Build rationale
        rationale = f"Contextual Thompson Sampling for {age_group} + {job_category} recommends {arm_name} ({expected_conv:.1%} historical conversion rate)."

        return jsonify({
            "recommended_arm": int(arm_id),
            "arm_name": arm_name,
            "expected_conversion": round(float(expected_conv), 4),
            "context": {
                "age_group": age_group,
                "job_category": job_category
            },
            "rationale": rationale
        }), 200

    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# CANARY DEPLOY ENDPOINTS
# ============================================================================

@app.route('/canary/start', methods=['POST'])
def start_canary_deploy():
    """
    Inicia canary deploy com modelo novo

    Body:
    {
        "canary_percentage": 5,  # % do tráfego para modelo novo
        "model_path": "data/models/thompson_model_v2.json",  # opcional
        "baseline_model_path": "data/models/thompson_model_baseline_temporal.json"  # opcional
    }

    Response:
    {
        "status": "canary_started",
        "canary_percentage": 5,
        "baseline_model": "thompson_model.json",
        "canary_model": "thompson_model_v2.json"
    }
    """
    global CANARY_CONFIG, CANARY_MODEL, CANARY_BASELINE_MODEL

    try:
        data = request.get_json() or {}
        canary_pct = data.get('canary_percentage', 5)

        # Validar percentage
        if not (0 < canary_pct < 100):
            return jsonify({"error": "canary_percentage must be between 1 and 99"}), 400

        # Carregar modelo canary
        repo_root = Path(__file__).parent.parent.parent.parent
        canary_model_path = Path(data.get('model_path', 'data/models/thompson_model_v2.json'))
        if not canary_model_path.is_absolute():
            canary_model_path = repo_root / canary_model_path

        # Se modelo não existe, usar o baseline como canary (para teste)
        if not canary_model_path.exists():
            logger.warning(f"Canary model not found at {canary_model_path}, using baseline as canary")
            canary_model_path = repo_root / "data" / "models" / "thompson_model.json"

        # Carrega o modelo canary de fato em memória (antes disso, o
        # endpoint só guardava o caminho e usava o modelo de produção
        # para os dois lados — os dois modelos nunca eram realmente
        # diferentes)
        CANARY_MODEL = load_model_from_path(canary_model_path)
        logger.info(f"Canary model loaded from {canary_model_path}")

        # Baseline: SEMPRE um objeto separado do MODEL global de produção,
        # nunca o MODEL em si — canary_recommend() chama model.update() no
        # lado BASELINE pra simular aprendizado, e se isso apontasse pro
        # MODEL usado por /recommend, cada requisição de canary corromperia
        # o modelo de produção com conversões simuladas (achado da
        # auditoria de re-verificação 2026-08-05: aliasing, não cópia).
        baseline_model_name = 'thompson_model.json'
        baseline_model_path = data.get('baseline_model_path')
        if baseline_model_path:
            baseline_path = Path(baseline_model_path)
            if not baseline_path.is_absolute():
                baseline_path = repo_root / baseline_path
            CANARY_BASELINE_MODEL = load_model_from_path(baseline_path)
            baseline_model_name = baseline_path.name
            logger.info(f"Canary baseline override loaded from {baseline_path}")
        else:
            CANARY_BASELINE_MODEL = copy.deepcopy(MODEL)

        # Ativar canary
        CANARY_CONFIG['enabled'] = True
        CANARY_CONFIG['canary_percentage'] = canary_pct
        CANARY_CONFIG['baseline_model'] = baseline_model_name
        CANARY_CONFIG['canary_model'] = str(canary_model_path.name)
        CANARY_CONFIG['metrics'] = {
            'baseline': {'conversions': 0, 'total': 0, 'decisions': []},
            'canary': {'conversions': 0, 'total': 0, 'decisions': []}
        }

        logger.info(f"Canary deploy started: {canary_pct}% do tráfego")

        return jsonify({
            "status": "canary_started",
            "canary_percentage": canary_pct,
            "baseline_model": CANARY_CONFIG['baseline_model'],
            "canary_model": CANARY_CONFIG['canary_model'],
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error starting canary deploy: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/canary/recommend', methods=['POST'])
def canary_recommend():
    """
    Endpoint recomendação COM canary deploy ativo
    Roteia: 5% → modelo novo, 95% → modelo atual

    Body: (mesmo que /recommend)
    {
        "age": 35,
        "job": "admin",
        ...
    }

    Response:
    {
        "recommended_arm": 0,
        "arm_name": "Cellular_Standard",
        "model_version": "BASELINE",  # ou "CANARY"
        "expected_conversion": 0.1127,
        ...
    }
    """
    if not CANARY_CONFIG['enabled']:
        return jsonify({"error": "Canary deploy not active. Use /canary/start first"}), 400

    try:
        data = request.get_json()

        # Validar entrada
        if not data or 'age' not in data or 'job' not in data:
            return jsonify({"error": "Missing required fields: age, job"}), 400

        # Decidir: baseline ou canary?
        if random.random() < (CANARY_CONFIG['canary_percentage'] / 100):
            model_version = "CANARY"
            model = CANARY_MODEL if CANARY_MODEL is not None else MODEL
        else:
            model_version = "BASELINE"
            model = CANARY_BASELINE_MODEL if CANARY_BASELINE_MODEL is not None else MODEL

        # Gerar recomendação
        arm_id, context = model.select_arm(age=data['age'], job=data['job'])
        age_group, job_category = context

        # Stats — crença atual do modelo (reportada ao cliente, não usada
        # como fonte do outcome simulado abaixo)
        bandit = model.bandits[context]
        conversions = bandit.get_conversion_rates()
        expected_conv = conversions[arm_id]

        # Simula o resultado (converteu ou não) usando a taxa REAL
        # histórica desse (contexto, arm) — não a crença do próprio modelo.
        # Usar a crença do modelo aqui seria circular: cada chamada
        # realimentaria ruído auto-referente no mesmo modelo sendo avaliado
        # (Achado Crítico 3). REAL_ARM_RATES vem de fora, calculado uma vez
        # a partir do dado de treino (load_real_arm_rates).
        true_rate = REAL_ARM_RATES.get((age_group, job_category, arm_id), expected_conv)
        converted = 1 if random.random() < true_rate else 0
        model.update(context, arm_id, converted)

        # Registrar decisão
        decision = {
            'timestamp': datetime.now().isoformat(),
            'age': data['age'],
            'job': data['job'],
            'context': (age_group, job_category),
            'arm': int(arm_id),
            'arm_name': ARM_NAMES[arm_id],
            'conversion_rate': float(expected_conv),
            'converted': converted
        }

        if model_version == 'BASELINE':
            CANARY_CONFIG['metrics']['baseline']['total'] += 1
            CANARY_CONFIG['metrics']['baseline']['conversions'] += converted
            CANARY_CONFIG['metrics']['baseline']['decisions'].append(decision)
        else:
            CANARY_CONFIG['metrics']['canary']['total'] += 1
            CANARY_CONFIG['metrics']['canary']['conversions'] += converted
            CANARY_CONFIG['metrics']['canary']['decisions'].append(decision)

        arm_name = ARM_NAMES[arm_id]
        rationale = f"Thompson Sampling ({model_version}): {age_group} + {job_category} → {arm_name}"

        return jsonify({
            "recommended_arm": int(arm_id),
            "arm_name": arm_name,
            "model_version": model_version,  # ← Indica qual versão foi usada
            "expected_conversion": round(float(expected_conv), 4),
            "converted": bool(converted),
            "context": {
                "age_group": age_group,
                "job_category": job_category
            },
            "rationale": rationale,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in canary recommendation: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/canary/metrics', methods=['GET'])
def canary_metrics():
    """
    Retorna métricas do canary deploy

    Response:
    {
        "canary_enabled": true,
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
        "improvement": 0.0074,
        "should_promote": true,
        "p_value": 0.8234
    }
    """
    if not CANARY_CONFIG['enabled']:
        return jsonify({"error": "Canary deploy not active"}), 400

    try:
        baseline_stats = CANARY_CONFIG['metrics']['baseline']
        canary_stats = CANARY_CONFIG['metrics']['canary']

        baseline_rate = baseline_stats['conversions'] / baseline_stats['total'] if baseline_stats['total'] > 0 else 0
        canary_rate = canary_stats['conversions'] / canary_stats['total'] if canary_stats['total'] > 0 else 0

        # Chi-square test simples
        from datathon.validation import compare_conversions

        try:
            if baseline_stats['total'] > 0 and canary_stats['total'] > 0:
                test_result = compare_conversions(
                    canary_stats['conversions'], canary_stats['total'],
                    baseline_stats['conversions'], baseline_stats['total']
                )
                is_significant = bool(test_result.is_significant)
                p_value = float(test_result.p_value)
            else:
                is_significant = False
                p_value = 1.0
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            is_significant = False
            p_value = 1.0

        improvement = canary_rate - baseline_rate

        return jsonify({
            "canary_enabled": True,
            "canary_percentage": CANARY_CONFIG['canary_percentage'],
            "baseline": {
                "total": baseline_stats['total'],
                "conversions": baseline_stats['conversions'],
                "rate": round(baseline_rate, 4)
            },
            "canary": {
                "total": canary_stats['total'],
                "conversions": canary_stats['conversions'],
                "rate": round(canary_rate, 4)
            },
            "improvement_pp": round(improvement * 100, 2),
            "should_promote": bool(is_significant and improvement > 0.001),
            "p_value": round(p_value, 4),
            "is_significant": is_significant,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        import traceback
        logger.error(f"Error getting canary metrics: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/canary/promote', methods=['POST'])
def promote_canary():
    """
    Promove canary para baseline (faz commit do modelo novo)

    Response:
    {
        "status": "canary_promoted",
        "message": "Canary model promoted to baseline"
    }
    """
    global CANARY_BASELINE_MODEL

    if not CANARY_CONFIG['enabled']:
        return jsonify({"error": "Canary deploy not active"}), 400

    try:
        logger.warning(f"PROMOTING canary model to baseline!")
        logger.warning(f"Baseline stats: {CANARY_CONFIG['metrics']['baseline']}")
        logger.warning(f"Canary stats: {CANARY_CONFIG['metrics']['canary']}")

        # Desativar canary
        CANARY_CONFIG['enabled'] = False
        CANARY_BASELINE_MODEL = None
        CANARY_CONFIG['metrics'] = {
            'baseline': {'conversions': 0, 'total': 0, 'decisions': []},
            'canary': {'conversions': 0, 'total': 0, 'decisions': []}
        }

        return jsonify({
            "status": "canary_promoted",
            "message": "Canary model promoted to baseline (canary deploy completed)",
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error promoting canary: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/canary/rollback', methods=['POST'])
def rollback_canary():
    """
    Rollback: volta para 100% baseline (descarta modelo novo)

    Response:
    {
        "status": "canary_rolled_back",
        "message": "Canary deploy rolled back to baseline"
    }
    """
    global CANARY_BASELINE_MODEL

    if not CANARY_CONFIG['enabled']:
        return jsonify({"error": "Canary deploy not active"}), 400

    try:
        logger.error(f"ROLLING BACK canary deploy!")
        logger.error(f"Baseline stats: {CANARY_CONFIG['metrics']['baseline']}")
        logger.error(f"Canary stats: {CANARY_CONFIG['metrics']['canary']}")

        # Desativar canary
        CANARY_CONFIG['enabled'] = False
        CANARY_BASELINE_MODEL = None
        CANARY_CONFIG['metrics'] = {
            'baseline': {'conversions': 0, 'total': 0, 'decisions': []},
            'canary': {'conversions': 0, 'total': 0, 'decisions': []}
        }

        return jsonify({
            "status": "canary_rolled_back",
            "message": "Canary deploy rolled back to baseline (model discarded)",
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error rolling back canary: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Load model at app startup (not just when run as __main__)
load_model()
load_real_arm_rates()

if __name__ == '__main__':
    # Run server
    print("=" * 60)
    print("Datathon API - Thompson Sampling")
    print("=" * 60)
    print(f"Model status: {MODEL_METADATA['status']}")
    print("API running on http://localhost:5000")
    print("Swagger UI: http://localhost:5000/apidocs")
    print("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=5000)
