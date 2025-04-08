# register_lerobot_env.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import yaml
import os

# Load Azure ML config
with open("azure/azure_config.yml", "r") as f:
    azure_config = yaml.safe_load(f)

# Connect to workspace
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=azure_config["subscription_id"],
    resource_group_name=azure_config["resource_group"],
    workspace_name=azure_config["workspace_name"],
)

# Create and register the environment
env = Environment(
    name="trossen_lerobot_env",
    description="Environment for LeRobot based on pyproject.toml dependencies",
    conda_file="azure/azure_trossen_lerobot_env.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

# Register the environment
registered_env = ml_client.environments.create_or_update(env)
print(f"Environment 'trossen-lerobot-env' registered with version {registered_env.version}")