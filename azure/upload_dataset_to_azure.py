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


def upload_single_file(dataset_name: str, local_file_path: str, destination_path: str) -> None:
    """Upload a single file to an existing dataset.
    
    Args:
        dataset_name: Name of the existing dataset
        local_file_path: Path to the file to upload
        destination_path: Relative path within the dataset (e.g., 'meta/new_file.json')
    """
    # Load configuration and set up ML client
    dirname = os.path.abspath(os.path.dirname(__file__))
    azure_config = load_yaml(os.path.join(dirname, "azure_config.yml"))
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=azure_config["subscription_id"],
        resource_group_name=azure_config["resource_group"],
        workspace_name=azure_config["workspace_name"],
    )
    
    # Get dataset details
    dataset = ml_client.data.get(name=dataset_name, version="1")
    
    # Extract storage info and upload directly (similar to Option 2)
    # ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a dataset to Azure ML")
    
    # Create an argument group for the different upload modes
    group = parser.add_mutually_exclusive_group(required=True)
    
    # Add the dataset_dir as an option in one group
    group.add_argument("--dataset_dir", type=str, 
                       help="Local directory path containing the dataset")
    
    # Add the single_file as an option in another group
    group.add_argument("--single_file", type=str,
                       help="Upload a single file instead of the entire directory")
    
    # Other arguments
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Name for the dataset in Azure ML (uses config value if not provided)")
    parser.add_argument("--destination_path", type=str, default=None,
                       help="Destination path for the single file within the dataset")
    
    args = parser.parse_args()
    
    # Check which mode we're running in
    if args.single_file:
        if not args.destination_path:
            parser.error("--destination_path is required when using --single_file")
        upload_single_file(args.dataset_name, args.single_file, args.destination_path)
    else:
        main(dataset_dir=args.dataset_dir, dataset_name=args.dataset_name)
