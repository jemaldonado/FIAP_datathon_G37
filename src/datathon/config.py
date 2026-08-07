"""
Configuration management for Datathon pipeline.

Supports running in:
1. LOCAL - Development/testing with MLflow local
2. AWS - Amazon SageMaker + S3
3. AZURE - Azure ML + Blob Storage

Usage:
    from datathon.config import Config, ENVIRONMENT

    # Automatically detects environment
    config = Config()
    print(f"Running in {config.env}")
    print(f"Data path: {config.data_path}")
"""

import os
from pathlib import Path
from enum import Enum
from typing import Optional
from dotenv import load_dotenv


load_dotenv()


class Environment(Enum):
    """Supported environments"""
    LOCAL = "local"
    AWS = "aws"
    AZURE = "azure"


class Config:
    """
    Configuration for different environments.

    Detects environment from:
    1. DATATHON_ENV environment variable
    2. AWS/Azure SDK presence
    3. Default: LOCAL
    """

    def __init__(self, env: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env: Override environment ('local', 'aws', 'azure')
        """
        self.env = self._detect_environment(env)
        self._setup_paths()
        self._setup_storage()
        self._setup_ml_tracking()

    @staticmethod
    def _detect_environment(override: Optional[str] = None) -> Environment:
        """Detect which environment to run in"""
        if override:
            return Environment(override.lower())

        # Check environment variable
        env_var = os.getenv('DATATHON_ENV', '').lower()
        if env_var in ['aws', 'azure']:
            return Environment(env_var)

        # Check if AWS credentials exist
        if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
            return Environment.AWS

        # Check if Azure credentials exist
        if os.getenv('AZURE_SUBSCRIPTION_ID'):
            return Environment.AZURE

        # Default to local
        return Environment.LOCAL

    def _setup_paths(self):
        """Setup paths based on environment"""
        if self.env == Environment.LOCAL:
            # Local paths (relative to project root)
            self.project_root = Path(__file__).parent.parent.parent
            self.data_path = self.project_root / "data" / "processed"
            # data/models/ (not the root models/ dir) is what the API
            # (src/datathon/api/app.py) and the dashboard actually read —
            # keeping this in sync avoids promoting/retraining into a
            # folder nothing in production looks at.
            self.model_path = self.project_root / "data" / "models"
            self.output_path = self.project_root / "outputs"

        elif self.env == Environment.AWS:
            # AWS paths (S3 and local cache)
            self.s3_bucket = os.getenv('AWS_S3_BUCKET', 'datathon-bucket')
            self.s3_prefix = os.getenv('AWS_S3_PREFIX', 'datathon/')
            self.project_root = Path('/opt/ml')
            self.data_path = self.project_root / "input" / "data"
            self.model_path = self.project_root / "model"
            self.output_path = self.project_root / "output"

        elif self.env == Environment.AZURE:
            # Azure paths (Blob Storage and local cache)
            self.blob_container = os.getenv('AZURE_BLOB_CONTAINER', 'datathon')
            self.blob_prefix = os.getenv('AZURE_BLOB_PREFIX', 'datathon/')
            
            # FALLBACK DE MLOPS: Verifica se está rodando fisicamente na nuvem
            azure_mount_path = '/mnt/batch/tasks/shared'
            if os.path.exists(azure_mount_path):
                self.project_root = Path(azure_mount_path)
            else:
                # Se não encontrar o caminho da Azure, usa o caminho local da sua máquina
                self.project_root = Path(__file__).parent.parent.parent

            self.data_path = self.project_root / "data" / "processed"
            self.model_path = self.project_root / "data" / "models"
            self.output_path = self.project_root / "outputs"

        # Create directories if they don't exist
        for path in [self.data_path, self.model_path, self.output_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _setup_storage(self):
        """Setup storage backend based on environment"""
        if self.env == Environment.LOCAL:
            self.storage_type = 'local'
            self.storage_config = {
                'type': 'local',
                'path': str(self.data_path)
            }

        elif self.env == Environment.AWS:
            self.storage_type = 's3'
            self.storage_config = {
                'type': 's3',
                'bucket': self.s3_bucket,
                'prefix': self.s3_prefix,
                'region': os.getenv('AWS_REGION', 'us-east-1')
            }

        elif self.env == Environment.AZURE:
            self.storage_type = 'blob'
            self.storage_config = {
                'type': 'blob',
                'container': self.blob_container,
                'prefix': self.blob_prefix,
                'connection_string': os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            }

    def _setup_ml_tracking(self):
        """Setup ML experiment tracking based on environment"""
        if self.env == Environment.LOCAL:
            # MLflow local (no setup needed, runs on localhost)
            self.tracking_type = 'mlflow_local'
            self.tracking_uri = 'http://localhost:5000'
            self.tracking_config = {
                'type': 'mlflow',
                'uri': self.tracking_uri,
                'local': True
            }

        elif self.env == Environment.AWS:
            # SageMaker Experiments (or MLflow on S3)
            self.tracking_type = 'sagemaker'
            self.tracking_config = {
                'type': 'sagemaker',
                'experiment_name': os.getenv('SAGEMAKER_EXPERIMENT', 'datathon'),
                'role': os.getenv('SAGEMAKER_ROLE_ARN')
            }

        elif self.env == Environment.AZURE:
            # Azure ML Experiments
            self.tracking_type = 'azure_ml'
            self.tracking_config = {
                'type': 'azure_ml',
                'workspace': os.getenv('AZURE_ML_WORKSPACE'),
                'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
                'resource_group': os.getenv('AZURE_RESOURCE_GROUP')
            }

    def get_model_path(self, filename: str = 'thompson_model.json') -> str:
        """Get full path to model file"""
        return str(self.model_path / filename)

    def get_data_path(self, filename: str = 'bank_marketing_primary.parquet') -> str:
        """Get full path to data file"""
        return str(self.data_path / filename)

    def get_output_path(self, filename: str) -> str:
        """Get full path to output file"""
        return str(self.output_path / filename)

    def __repr__(self):
        return f"Config(env={self.env.value}, storage={self.storage_type}, tracking={self.tracking_type})"


# Global config instance (initialized on first import)
_config: Optional[Config] = None


def get_config(env: Optional[str] = None) -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = Config(env=env)
    return _config


def get_environment() -> Environment:
    """Get current environment"""
    return get_config().env


def is_local() -> bool:
    """Check if running locally"""
    return get_environment() == Environment.LOCAL


def is_aws() -> bool:
    """Check if running on AWS"""
    return get_environment() == Environment.AWS


def is_azure() -> bool:
    """Check if running on Azure"""
    return get_environment() == Environment.AZURE