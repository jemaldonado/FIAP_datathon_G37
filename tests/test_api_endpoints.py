"""
Tests for Datathon API endpoints.

This file tests:
1. GET / - Returns API metadata
2. GET /health - Health check
3. POST /recommend - Recommendations with context
4. GET /swagger.json - OpenAPI spec

Library: pytest (https://docs.pytest.org/)

Run tests:
    pytest tests/test_api_endpoints.py -v

Or run all tests:
    pytest tests/ -v
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path so we can import the app
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Flask app
from datathon.api.app import app


# ============================================================================
# FIXTURES - Setup code that runs before each test
# ============================================================================

@pytest.fixture
def client():
    """
    Create a test client for the Flask app.

    This fixture:
    1. Creates a test version of the API (without running server)
    2. Provides a client to make HTTP requests
    3. Runs before each test, cleans up after

    Return: Flask test client
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ============================================================================
# TEST GROUP 2: Health Check Endpoint
# ============================================================================

class TestHealthEndpoint:
    """Tests for GET /health endpoint"""

    def test_health_returns_200(self, client):
        """
        TEST: GET /health should return HTTP 200

        Health check is used by load balancers / monitoring to know if API is alive.
        """
        # ACT
        response = client.get('/health')

        # ASSERT
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        """
        TEST: Health response should have status: 'ok'
        """
        # ACT
        response = client.get('/health')
        data = response.get_json()

        # ASSERT
        assert data['status'] == 'ok'

    def test_health_model_status(self, client):
        """
        TEST: Health should report model status

        Either 'loaded' or 'not_loaded'
        """
        # ACT
        response = client.get('/health')
        data = response.get_json()

        # ASSERT
        assert data['model'] in ['loaded', 'not_loaded']


# ============================================================================
# TEST GROUP 3: Recommend Endpoint (Main Logic)
# ============================================================================

class TestRecommendEndpoint:
    """Tests for POST /recommend endpoint"""

    def test_recommend_requires_post(self, client):
        """
        TEST: GET /recommend should NOT work (must be POST)

        The endpoint only accepts POST requests.
        """
        # ACT
        response = client.get('/recommend')

        # ASSERT
        assert response.status_code == 405  # 405 = Method Not Allowed

    def test_recommend_young_technical(self, client):
        """
        TEST: Recommend for young technical person

        Profile: age=28, job=admin
        Expected context: Young + Technical
        Expected arm: Some contact strategy (0-3)
        """
        # ARRANGE
        customer = {
            "age": 28,
            "job": "admin",
            "marital": "single",
            "education": "high.school",
            "contact": "cellular",
            "campaign": 1
        }

        # ACT
        response = client.post(
            '/recommend',
            data=json.dumps(customer),
            content_type='application/json'
        )

        # ASSERT
        assert response.status_code == 200
        data = response.get_json()
        assert data['context']['age_group'] == 'Young'
        assert data['context']['job_category'] == 'Technical'
        assert 0 <= data['recommended_arm'] < 4

    def test_recommend_prime_business(self, client):
        """
        TEST: Recommend for prime business executive

        Profile: age=42, job=management
        Expected context: Prime + Business
        """
        # ARRANGE
        customer = {
            "age": 42,
            "job": "management",
            "marital": "married",
            "education": "university.degree",
            "contact": "cellular",
            "campaign": 2
        }

        # ACT
        response = client.post(
            '/recommend',
            data=json.dumps(customer),
            content_type='application/json'
        )

        # ASSERT
        assert response.status_code == 200
        data = response.get_json()
        assert data['context']['age_group'] == 'Prime'
        assert data['context']['job_category'] == 'Business'

    def test_recommend_senior_retired(self, client):
        """
        TEST: Recommend for senior retiree

        Profile: age=68, job=retired
        Expected context: Senior + Other
        """
        # ARRANGE
        customer = {
            "age": 68,
            "job": "retired",
            "marital": "married",
            "education": "basic.4y",
            "contact": "telephone",
            "campaign": 1
        }

        # ACT
        response = client.post(
            '/recommend',
            data=json.dumps(customer),
            content_type='application/json'
        )

        # ASSERT
        assert response.status_code == 200
        data = response.get_json()
        assert data['context']['age_group'] == 'Senior'
        assert data['context']['job_category'] == 'Other'

    def test_recommend_response_has_required_fields(self, client):
        """
        TEST: Recommend response must have all required fields

        Response structure:
        {
            "recommended_arm": 0,
            "arm_name": "Cellular_Standard",
            "expected_conversion": 0.15,
            "context": {"age_group": "Prime", "job_category": "Technical"},
            "rationale": "..."
        }
        """
        # ARRANGE
        customer = {
            "age": 35,
            "job": "admin",
            "marital": "married",
            "education": "university.degree",
            "contact": "cellular",
            "campaign": 1
        }

        # ACT
        response = client.post(
            '/recommend',
            data=json.dumps(customer),
            content_type='application/json'
        )
        data = response.get_json()

        # ASSERT
        required_fields = [
            'recommended_arm',
            'arm_name',
            'expected_conversion',
            'context',
            'rationale'
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_recommend_arm_name_is_valid(self, client):
        """
        TEST: Arm name should be one of 4 valid campaign strategies

        Valid strategies (all 4 are real segments relabeled, not estimates —
        see datathon.bandit.assign_arm):
        - Cellular_Standard (cellular, first contact)
        - Email_Campaign (cellular, repeat contact)
        - SMS_Alert (telephone, first contact)
        - Call_Premium (telephone, repeat contact)
        """
        # ARRANGE
        customer = {
            "age": 45,
            "job": "technician",
            "marital": "married",
            "education": "secondary",
            "contact": "cellular",
            "campaign": 1
        }

        # ACT
        response = client.post(
            '/recommend',
            data=json.dumps(customer),
            content_type='application/json'
        )
        data = response.get_json()

        # ASSERT
        valid_arms = ['Cellular_Standard', 'Email_Campaign', 'SMS_Alert', 'Call_Premium']
        assert data['arm_name'] in valid_arms

    def test_recommend_conversion_rate_is_percentage(self, client):
        """
        TEST: Expected conversion should be between 0 and 1 (or 0-100 if %)
        """
        # ARRANGE
        customer = {
            "age": 35,
            "job": "admin",
            "marital": "married",
            "education": "university.degree",
            "contact": "cellular",
            "campaign": 1
        }

        # ACT
        response = client.post(
            '/recommend',
            data=json.dumps(customer),
            content_type='application/json'
        )
        data = response.get_json()
        conv = data['expected_conversion']

        # ASSERT - Should be between 0 and 1
        assert 0 <= conv <= 1, f"Conversion {conv} not in [0, 1]"

    def test_recommend_missing_required_field(self, client):
        """
        TEST: Missing required field should return error (400)

        If age is missing, API should return 400 Bad Request
        """
        # ARRANGE - Missing 'age'
        customer = {
            "job": "admin",
            "marital": "married",
            "education": "university.degree",
            "contact": "cellular",
            "campaign": 1
        }

        # ACT
        response = client.post(
            '/recommend',
            data=json.dumps(customer),
            content_type='application/json'
        )

        # ASSERT
        assert response.status_code == 400


# ============================================================================
# TEST GROUP 4: Swagger Specification
# ============================================================================

class TestSwaggerEndpoint:
    """Tests for GET /swagger.json endpoint"""

    def test_swagger_json_returns_200(self, client):
        """
        TEST: GET /swagger.json should return 200
        """
        # ACT
        response = client.get('/swagger.json')

        # ASSERT
        assert response.status_code == 200

    def test_swagger_json_is_valid_json(self, client):
        """
        TEST: Response should be valid JSON
        """
        # ACT
        response = client.get('/swagger.json')

        # ASSERT - If not valid JSON, this will raise exception
        data = response.get_json()
        assert isinstance(data, dict)

    def test_swagger_has_required_fields(self, client):
        """
        TEST: Swagger spec must have swagger version, info, paths
        """
        # ACT
        response = client.get('/swagger.json')
        data = response.get_json()

        # ASSERT
        assert 'swagger' in data
        assert 'info' in data
        assert 'paths' in data

    def test_swagger_has_recommend_endpoint(self, client):
        """
        TEST: Swagger spec should document /recommend endpoint
        """
        # ACT
        response = client.get('/swagger.json')
        data = response.get_json()

        # ASSERT
        assert '/recommend' in data['paths']


# ============================================================================
# TEST GROUP 5: Integration Tests (Multiple endpoints together)
# ============================================================================

class TestIntegration:
    """Integration tests - test multiple components together"""

    def test_full_workflow(self, client):
        """
        TEST: Full workflow - health check, then recommendation

        1. Check API is healthy
        2. Get recommendation
        3. Verify response
        """
        # ARRANGE
        customer = {
            "age": 35,
            "job": "admin",
            "marital": "married",
            "education": "university.degree",
            "contact": "cellular",
            "campaign": 1
        }

        # ACT - Step 1: Check health
        health = client.get('/health')
        assert health.status_code == 200

        # ACT - Step 2: Get recommendation
        recommend = client.post(
            '/recommend',
            data=json.dumps(customer),
            content_type='application/json'
        )

        # ASSERT
        assert recommend.status_code == 200
        data = recommend.get_json()
        assert data['recommended_arm'] is not None

    def test_different_profiles_get_different_contexts(self, client):
        """
        TEST: Different age/job combinations should map to different contexts

        This proves the contextual part is working!
        """
        # ARRANGE
        profiles = [
            {"age": 25, "job": "admin"},           # Young + Technical
            {"age": 40, "job": "management"},      # Prime + Business
            {"age": 70, "job": "retired"},         # Senior + Other
        ]

        # ACT & ASSERT
        contexts = []
        for profile in profiles:
            customer = {
                **profile,
                "marital": "married",
                "education": "university.degree",
                "contact": "cellular",
                "campaign": 1
            }
            response = client.post(
                '/recommend',
                data=json.dumps(customer),
                content_type='application/json'
            )
            data = response.get_json()
            context = (data['context']['age_group'], data['context']['job_category'])
            contexts.append(context)

        # All 3 should be different
        assert contexts[0] != contexts[1]
        assert contexts[1] != contexts[2]
        assert contexts[0] != contexts[2]


if __name__ == '__main__':
    # This allows running tests with: python -m pytest tests/test_api_endpoints.py
    pytest.main([__file__, '-v'])
