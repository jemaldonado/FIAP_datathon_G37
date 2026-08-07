"""
Tests for Datathon API canary deploy endpoints.

This file tests:
1. POST /canary/start   - Activate canary deploy
2. POST /canary/recommend - Recommendation routed baseline/canary
3. GET  /canary/metrics - Comparative metrics + chi-square gate
4. POST /canary/promote - Promote canary to baseline
5. POST /canary/rollback - Discard canary, keep baseline

These endpoints previously shipped two bugs found by code audit
(2026-08-05): a circular reward simulation (canary outcome depended on
the model's own belief instead of the real historical rate) and a
model-aliasing bug (the "baseline" side of the canary pointed at the
same object as the production MODEL used by /recommend, so canary
traffic silently corrupted production). Both are fixed in
src/datathon/api/app.py — the regression test at the end of this file
locks in the aliasing fix specifically, since it's the kind of bug that
only shows up under real traffic and has no other test coverage.

Library: pytest (https://docs.pytest.org/)

Run tests:
    pytest tests/test_canary_endpoints.py -v

Or run all tests:
    pytest tests/ -v
"""

import pytest
import json
import sys
import copy
from pathlib import Path

# Add src to path so we can import the app
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datathon.api.app as app_module
from datathon.api.app import app


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Flask test client, same pattern as test_api_endpoints.py"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_canary_state():
    """
    Canary state lives in module-level globals (CANARY_CONFIG,
    CANARY_MODEL, CANARY_BASELINE_MODEL), shared across every test in
    the process. Reset it before each test so tests don't leak into
    each other regardless of run order.
    """
    app_module.CANARY_CONFIG = {
        'enabled': False,
        'canary_percentage': 5,
        'baseline_model': None,
        'canary_model': None,
        'metrics': {
            'baseline': {'conversions': 0, 'total': 0, 'decisions': []},
            'canary': {'conversions': 0, 'total': 0, 'decisions': []}
        }
    }
    app_module.CANARY_MODEL = None
    app_module.CANARY_BASELINE_MODEL = None
    yield


def _customer(**overrides):
    customer = {
        "age": 28,
        "job": "admin",
        "marital": "single",
        "education": "high.school",
        "contact": "cellular",
        "campaign": 1
    }
    customer.update(overrides)
    return customer


# ============================================================================
# TEST GROUP 1: /canary/start
# ============================================================================

class TestCanaryStart:
    """Tests for POST /canary/start"""

    def test_start_default_percentage(self, client):
        """TEST: Starting with no body should default to 5% canary traffic"""
        response = client.post(
            '/canary/start',
            data=json.dumps({}),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'canary_started'
        assert data['canary_percentage'] == 5

    def test_start_custom_percentage(self, client):
        """TEST: canary_percentage in body should be honored"""
        response = client.post(
            '/canary/start',
            data=json.dumps({"canary_percentage": 20}),
            content_type='application/json'
        )

        assert response.status_code == 200
        assert response.get_json()['canary_percentage'] == 20

    @pytest.mark.parametrize("pct", [0, 100, -5, 150])
    def test_start_rejects_invalid_percentage(self, client, pct):
        """TEST: canary_percentage must be strictly between 0 and 100"""
        response = client.post(
            '/canary/start',
            data=json.dumps({"canary_percentage": pct}),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_start_enables_canary(self, client):
        """TEST: after /canary/start, CANARY_CONFIG['enabled'] is True"""
        client.post('/canary/start', data=json.dumps({}), content_type='application/json')

        assert app_module.CANARY_CONFIG['enabled'] is True


# ============================================================================
# TEST GROUP 2: /canary/recommend
# ============================================================================

class TestCanaryRecommend:
    """Tests for POST /canary/recommend"""

    def test_recommend_requires_active_canary(self, client):
        """TEST: /canary/recommend before /canary/start should 400"""
        response = client.post(
            '/canary/recommend',
            data=json.dumps(_customer()),
            content_type='application/json'
        )

        assert response.status_code == 400
        assert 'error' in response.get_json()

    def test_recommend_after_start_returns_200(self, client):
        """TEST: /canary/recommend after /canary/start should succeed"""
        client.post('/canary/start', data=json.dumps({}), content_type='application/json')

        response = client.post(
            '/canary/recommend',
            data=json.dumps(_customer()),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['model_version'] in ('BASELINE', 'CANARY')
        assert 0 <= data['recommended_arm'] < 4
        assert data['context']['age_group'] == 'Young'
        assert data['context']['job_category'] == 'Technical'

    def test_recommend_missing_required_field(self, client):
        """TEST: missing age/job should 400, same contract as /recommend"""
        client.post('/canary/start', data=json.dumps({}), content_type='application/json')

        response = client.post(
            '/canary/recommend',
            data=json.dumps({"job": "admin"}),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_recommend_updates_metrics_counters(self, client):
        """TEST: each call increments total for whichever side it routed to"""
        client.post('/canary/start', data=json.dumps({"canary_percentage": 50}),
                     content_type='application/json')

        for _ in range(10):
            client.post(
                '/canary/recommend',
                data=json.dumps(_customer()),
                content_type='application/json'
            )

        metrics = app_module.CANARY_CONFIG['metrics']
        total_decisions = metrics['baseline']['total'] + metrics['canary']['total']
        assert total_decisions == 10


# ============================================================================
# TEST GROUP 3: /canary/metrics
# ============================================================================

class TestCanaryMetrics:
    """Tests for GET /canary/metrics"""

    def test_metrics_requires_active_canary(self, client):
        """TEST: /canary/metrics before /canary/start should 400"""
        response = client.get('/canary/metrics')

        assert response.status_code == 400

    def test_metrics_after_recommendations(self, client):
        """TEST: metrics reflect the decisions routed so far"""
        client.post('/canary/start', data=json.dumps({"canary_percentage": 50}),
                     content_type='application/json')

        for _ in range(10):
            client.post(
                '/canary/recommend',
                data=json.dumps(_customer()),
                content_type='application/json'
            )

        response = client.get('/canary/metrics')

        assert response.status_code == 200
        data = response.get_json()
        assert data['canary_enabled'] is True
        assert data['baseline']['total'] + data['canary']['total'] == 10
        assert 'p_value' in data
        assert 'should_promote' in data

    def test_metrics_with_no_traffic_yet(self, client):
        """TEST: metrics right after /canary/start (0 decisions) shouldn't crash"""
        client.post('/canary/start', data=json.dumps({}), content_type='application/json')

        response = client.get('/canary/metrics')

        assert response.status_code == 200
        data = response.get_json()
        assert data['baseline']['total'] == 0
        assert data['canary']['total'] == 0
        assert data['baseline']['rate'] == 0
        assert data['canary']['rate'] == 0


# ============================================================================
# TEST GROUP 4: /canary/promote and /canary/rollback
# ============================================================================

class TestCanaryPromoteRollback:
    """Tests for POST /canary/promote and POST /canary/rollback"""

    def test_promote_requires_active_canary(self, client):
        response = client.post('/canary/promote')
        assert response.status_code == 400

    def test_rollback_requires_active_canary(self, client):
        response = client.post('/canary/rollback')
        assert response.status_code == 400

    def test_promote_deactivates_canary(self, client):
        """TEST: after promote, canary is disabled and metrics reset"""
        client.post('/canary/start', data=json.dumps({}), content_type='application/json')

        response = client.post('/canary/promote')

        assert response.status_code == 200
        assert response.get_json()['status'] == 'canary_promoted'
        assert app_module.CANARY_CONFIG['enabled'] is False

        # Metrics endpoint should reject again — canary session is over
        metrics_response = client.get('/canary/metrics')
        assert metrics_response.status_code == 400

    def test_rollback_deactivates_canary(self, client):
        """TEST: after rollback, canary is disabled and metrics reset"""
        client.post('/canary/start', data=json.dumps({}), content_type='application/json')

        response = client.post('/canary/rollback')

        assert response.status_code == 200
        assert response.get_json()['status'] == 'canary_rolled_back'
        assert app_module.CANARY_CONFIG['enabled'] is False

        metrics_response = client.get('/canary/metrics')
        assert metrics_response.status_code == 400


# ============================================================================
# TEST GROUP 5: Regression — model aliasing bug (auditoria 2026-08-05)
# ============================================================================

class TestCanaryDoesNotMutateProductionModel:
    """
    Regression test for the aliasing bug: the baseline side of a canary
    deploy must be an independent copy of the production model, never
    the production MODEL object itself. Before the fix, model.update()
    inside /canary/recommend mutated MODEL directly, so canary traffic
    silently changed what /recommend returned in production.
    """

    def test_baseline_is_not_the_production_model_object(self, client):
        """TEST: CANARY_BASELINE_MODEL must not be the same object as MODEL"""
        client.post('/canary/start', data=json.dumps({}), content_type='application/json')

        assert app_module.CANARY_BASELINE_MODEL is not app_module.MODEL

    def test_canary_traffic_does_not_change_production_recommendations(self, client):
        """
        TEST: hammering /canary/recommend must not change what
        GET /recommend (production) would answer for the same profile.
        """
        context = ('Young', 'Technical')
        production_model = app_module.MODEL
        bandit_before = production_model.bandits[context]
        trials_before = bandit_before.trials.copy()
        successes_before = bandit_before.successes.copy()

        client.post('/canary/start', data=json.dumps({"canary_percentage": 50}),
                     content_type='application/json')
        for _ in range(20):
            client.post(
                '/canary/recommend',
                data=json.dumps(_customer()),
                content_type='application/json'
            )

        bandit_after = app_module.MODEL.bandits[context]
        assert (bandit_after.trials == trials_before).all()
        assert (bandit_after.successes == successes_before).all()


if __name__ == '__main__':
    # This allows running tests with: python -m pytest tests/test_canary_endpoints.py
    pytest.main([__file__, '-v'])
