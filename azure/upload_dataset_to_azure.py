"""Creates and runs an Azure ML command job."""

import logging
from pathlib import Path
import argparse

from azure.ai.ml import MLClient, Input, Output, PyTorchDistribution, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, Model, ComputeInstance, Data
from azure.identity import DefaultAzureCredential
import os
import yaml


def load_yaml(path: str):
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def main(dataset_dir: str, dataset_name: str = None) -> None:
    logging.basicConfig(level=logging.INFO)
    dirname = os.path.abspath(os.path.dirname(__file__))
    azure_config = load_yaml(os.path.join(dirname, "azure_config.yml"))
    
    subscription_id = azure_config["subscription_id"]
    resource_group = azure_config["resource_group"]
    workspace_name = azure_config["workspace_name"]
    
    # Use provided dataset name or fall back to config
    data_name = dataset_name if dataset_name else azure_config["data_name"]
    
    print(f"Uploading dataset '{data_name}' from: {dataset_dir}")
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    
    # Get workspace info
    ws = ml_client.workspaces.get(workspace_name)
    print(f"Workspace location: {ws.location} | Resource group: {ws.resource_group}")

    # Ensure dataset directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Create the dataset
    dataset = Data(
        name=data_name,
        description=f"Dataset uploaded from {dataset_dir}",
        path=os.path.abspath(dataset_dir),
        type=AssetTypes.URI_FOLDER,
    )
    
    print(f"Creating/updating dataset '{data_name}'...")
    ml_client.data.create_or_update(dataset)
    print(f"Successfully uploaded dataset '{data_name}'")

    # Create a small verification job
    verification_job = command(
        inputs=dict(dataset=Input(path=f"{data_name}@latest")),
        command="ls -la ${{inputs.dataset}} && ls -la ${{inputs.dataset}}/data && ls -la ${{inputs.dataset}}/meta && ls -la ${{inputs.dataset}}/videos",
        # other job parameters...
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a dataset to Azure ML")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                       help="Local directory path containing the dataset")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Name for the dataset in Azure ML (uses config value if not provided)")
    args = parser.parse_args()
    main(dataset_dir=args.dataset_dir, dataset_name=args.dataset_name)
