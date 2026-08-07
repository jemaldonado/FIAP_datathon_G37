"""
Unit tests for Contextual Thompson Sampling model.

Tests validate:
- Model architecture (12 contexts)
- Context mapping (age_group × job_category)
- Arm selection
- Desk test scenarios
- Model serialization
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datathon.bandit import ContextualThompsonSampling, BetaBernoulliBandit


class TestBetaBernoulliBandit:
    """Test the base Thompson Sampling bandit"""

    def test_init_default_priors(self):
        """Test initialization with default priors"""
        bandit = BetaBernoulliBandit(n_arms=4)
        assert bandit.n_arms == 4
        np.testing.assert_array_equal(bandit.alpha, np.ones(4))
        np.testing.assert_array_equal(bandit.beta, np.ones(4))

    def test_init_custom_priors(self):
        """Test initialization with custom priors"""
        alpha_init = [10, 15, 20, 25]
        beta_init = [5, 10, 15, 20]
        bandit = BetaBernoulliBandit(n_arms=4, alpha_init=alpha_init, beta_init=beta_init)

        np.testing.assert_array_equal(bandit.alpha, np.array(alpha_init, dtype=float))
        np.testing.assert_array_equal(bandit.beta, np.array(beta_init, dtype=float))

    def test_select_arm_returns_valid_arm(self):
        """Test that select_arm returns valid arm index"""
        bandit = BetaBernoulliBandit(n_arms=4)
        for _ in range(100):
            arm = bandit.select_arm()
            assert 0 <= arm < 4

    def test_update_on_success(self):
        """Test that update increments alpha on success"""
        bandit = BetaBernoulliBandit(n_arms=4, alpha_init=[1, 1, 1, 1], beta_init=[1, 1, 1, 1])
        initial_alpha_0 = bandit.alpha[0]

        bandit.update(arm=0, reward=1)

        assert bandit.alpha[0] == initial_alpha_0 + 1
        assert bandit.successes[0] == 1
        assert bandit.trials[0] == 1

    def test_update_on_failure(self):
        """Test that update increments beta on failure"""
        bandit = BetaBernoulliBandit(n_arms=4, alpha_init=[1, 1, 1, 1], beta_init=[1, 1, 1, 1])
        initial_beta_0 = bandit.beta[0]

        bandit.update(arm=0, reward=0)

        assert bandit.beta[0] == initial_beta_0 + 1
        assert bandit.successes[0] == 0
        assert bandit.trials[0] == 1

    def test_get_conversion_rates(self):
        """Test conversion rate calculation"""
        bandit = BetaBernoulliBandit(n_arms=4, alpha_init=[10, 20, 30, 40], beta_init=[90, 80, 70, 60])
        rates = bandit.get_conversion_rates()

        expected = np.array([10/100, 20/100, 30/100, 40/100])
        np.testing.assert_array_almost_equal(rates, expected)


class TestContextualThompsonSampling:
    """Test the Contextual Thompson Sampling model"""

    def test_init_creates_12_contexts(self):
        """Test that initialization creates exactly 12 contexts"""
        model = ContextualThompsonSampling()
        assert len(model.bandits) == 12

    def test_age_group_mapping(self):
        """Test age group mapping logic"""
        model = ContextualThompsonSampling()

        assert model._get_age_group(25) == 'Young'
        assert model._get_age_group(35) == 'Prime'
        assert model._get_age_group(55) == 'Mature'
        assert model._get_age_group(72) == 'Senior'

    def test_age_group_boundaries(self):
        """Test age group boundaries"""
        model = ContextualThompsonSampling()

        # Test boundaries
        assert model._get_age_group(18) == 'Young'
        assert model._get_age_group(29) == 'Young'
        assert model._get_age_group(30) == 'Prime'
        assert model._get_age_group(44) == 'Prime'
        assert model._get_age_group(45) == 'Mature'
        assert model._get_age_group(59) == 'Mature'
        assert model._get_age_group(60) == 'Senior'
        assert model._get_age_group(100) == 'Senior'

    def test_job_category_mapping_technical(self):
        """Test job category mapping for technical jobs"""
        model = ContextualThompsonSampling()

        assert model._get_job_category('admin') == 'Technical'
        assert model._get_job_category('technician') == 'Technical'
        assert model._get_job_category('engineer') == 'Technical'

    def test_job_category_mapping_business(self):
        """Test job category mapping for business jobs"""
        model = ContextualThompsonSampling()

        assert model._get_job_category('management') == 'Business'
        assert model._get_job_category('director') == 'Business'
        assert model._get_job_category('entrepreneur') == 'Business'

    def test_job_category_mapping_other(self):
        """Test job category mapping for other jobs"""
        model = ContextualThompsonSampling()

        assert model._get_job_category('retired') == 'Other'
        assert model._get_job_category('unemployed') == 'Other'
        assert model._get_job_category('student') == 'Other'
        assert model._get_job_category('unknown_job_type') == 'Other'

    def test_select_arm_returns_valid_output(self):
        """Test that select_arm returns (arm, context)"""
        model = ContextualThompsonSampling()
        arm, context = model.select_arm(age=35, job='admin')

        assert isinstance(arm, (int, np.integer))
        assert 0 <= arm < 4
        assert isinstance(context, tuple)
        assert len(context) == 2
        assert context[0] in ['Young', 'Prime', 'Mature', 'Senior']
        assert context[1] in ['Technical', 'Business', 'Other']

    def test_update_increases_alpha_beta(self):
        """Test that update modifies the correct context"""
        model = ContextualThompsonSampling()
        context = ('Prime', 'Business')

        initial_alpha = model.bandits[context].alpha[0].copy()
        initial_beta = model.bandits[context].beta[0].copy()

        model.update(context, arm=0, reward=1)

        assert model.bandits[context].alpha[0] == initial_alpha + 1
        assert model.bandits[context].beta[0] == initial_beta

        model.update(context, arm=0, reward=0)

        assert model.bandits[context].beta[0] == initial_beta + 1

    def test_get_context_stats_valid_format(self):
        """Test that get_context_stats returns proper format"""
        model = ContextualThompsonSampling()
        context = ('Prime', 'Business')

        # Add some training data
        model.update(context, arm=0, reward=1)
        model.update(context, arm=1, reward=0)

        stats = model.get_context_stats(context)

        assert 'context' in stats
        assert 'arms' in stats
        assert len(stats['arms']) == 4

        for arm_name in ['Cellular_Standard', 'Email_Campaign', 'SMS_Alert', 'Call_Premium']:
            assert arm_name in stats['arms']
            assert 'trials' in stats['arms'][arm_name]
            assert 'successes' in stats['arms'][arm_name]
            assert 'rate' in stats['arms'][arm_name]


class TestDeskScenarios:
    """Desk test scenarios to validate model behavior"""

    def test_young_technical_scenario(self):
        """Scenario: Young technical professional"""
        model = ContextualThompsonSampling()
        arm, (age_group, job_cat) = model.select_arm(age=28, job='admin')

        assert age_group == 'Young'
        assert job_cat == 'Technical'
        assert 0 <= arm < 4

    def test_prime_business_scenario(self):
        """Scenario: Prime age business manager"""
        model = ContextualThompsonSampling()
        arm, (age_group, job_cat) = model.select_arm(age=38, job='director')

        assert age_group == 'Prime'
        assert job_cat == 'Business'
        assert 0 <= arm < 4

    def test_senior_other_scenario(self):
        """Scenario: Senior citizen, retired"""
        model = ContextualThompsonSampling()
        arm, (age_group, job_cat) = model.select_arm(age=72, job='retired')

        assert age_group == 'Senior'
        assert job_cat == 'Other'
        assert 0 <= arm < 4

    def test_mature_technical_scenario(self):
        """Scenario: Mature technician"""
        model = ContextualThompsonSampling()
        arm, (age_group, job_cat) = model.select_arm(age=55, job='technician')

        assert age_group == 'Mature'
        assert job_cat == 'Technical'
        assert 0 <= arm < 4


class TestModelSerialization:
    """Test model serialization and deserialization"""

    def test_model_can_be_serialized(self, tmp_path):
        """Test that model can be serialized with joblib"""
        model = ContextualThompsonSampling()

        # Add some data
        for context in model.bandits.keys():
            model.update(context, arm=0, reward=1)

        # Serialize
        model_file = tmp_path / "test_model.pkl"
        joblib.dump(model, model_file)

        assert model_file.exists()

    def test_model_can_be_deserialized(self, tmp_path):
        """Test that model can be deserialized and used"""
        model = ContextualThompsonSampling()

        # Add some training data
        context = ('Prime', 'Business')
        for _ in range(10):
            model.update(context, arm=0, reward=1)

        # Serialize
        model_file = tmp_path / "test_model.pkl"
        joblib.dump(model, model_file)

        # Deserialize
        loaded_model = joblib.load(model_file)

        # Verify it works
        arm, loaded_context = loaded_model.select_arm(age=35, job='director')
        assert loaded_context == ('Prime', 'Business')

        # Verify training data was preserved
        stats = loaded_model.get_context_stats(context)
        assert stats['arms']['Cellular_Standard']['trials'] == 10
        assert stats['arms']['Cellular_Standard']['successes'] == 10


class TestDataValidation:
    """Test data validation and drift detection"""

    def test_all_customers_assigned_to_context(self):
        """Test that all customers can be assigned to a context"""
        data_dir = Path(__file__).parent.parent / "data" / "processed"
        parquet_file = data_dir / "bank_marketing_primary.parquet"

        if not parquet_file.exists():
            pytest.skip("Data file not found")

        df = pd.read_parquet(parquet_file)
        model = ContextualThompsonSampling()

        # Try to map all customers
        df['age_group'] = df['age'].apply(model._get_age_group)
        df['job_category'] = df['job'].apply(model._get_job_category)

        # Count customers in contexts
        total_in_contexts = 0
        for context in model.bandits.keys():
            age_group, job_cat = context
            mask = (df['age_group'] == age_group) & (df['job_category'] == job_cat)
            total_in_contexts += mask.sum()

        assert total_in_contexts == len(df), f"Expected {len(df)} but got {total_in_contexts}"

    def test_context_distribution_matches_expected(self):
        """Test that context distribution is reasonable"""
        data_dir = Path(__file__).parent.parent / "data" / "processed"
        parquet_file = data_dir / "bank_marketing_primary.parquet"

        if not parquet_file.exists():
            pytest.skip("Data file not found")

        df = pd.read_parquet(parquet_file)
        model = ContextualThompsonSampling()

        df['age_group'] = df['age'].apply(model._get_age_group)
        df['job_category'] = df['job'].apply(model._get_job_category)

        # Prime age should be majority
        prime_count = (df['age_group'] == 'Prime').sum()
        prime_pct = prime_count / len(df)

        assert prime_pct > 0.4, f"Expected >40% Prime but got {prime_pct:.1%}"

        # Senior should be minority
        senior_count = (df['age_group'] == 'Senior').sum()
        senior_pct = senior_count / len(df)

        assert senior_pct < 0.1, f"Expected <10% Senior but got {senior_pct:.1%}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
