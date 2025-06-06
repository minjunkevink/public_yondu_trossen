"""Creates and runs an Azure ML command job with automatic token refresh."""

import logging
import time
import threading
from pathlib import Path
import argparse
import subprocess
import json
from datetime import datetime, timedelta

from azure.ai.ml import MLClient, Input, Output, PyTorchDistribution, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, Model, ComputeInstance, Data
from azure.identity import DefaultAzureCredential, AzureCliCredential
import os
import yaml


def load_yaml(path: str):
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def get_token_expiry():
    """Get the current Azure CLI token expiry time."""
    try:
        result = subprocess.run(
            ["az", "account", "get-access-token", "--query", "expiresOn", "-o", "tsv"],
            capture_output=True,
            text=True,
            check=True
        )
        expiry_str = result.stdout.strip()
        # Parse the expiry time
        expiry_time = datetime.strptime(expiry_str, "%Y-%m-%d %H:%M:%S.%f")
        return expiry_time
    except Exception as e:
        logging.warning(f"Could not get token expiry: {e}")
        return None


def refresh_azure_token():
    """Refresh the Azure CLI token."""
    try:
        logging.info("Refreshing Azure CLI token...")
        result = subprocess.run(
            ["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.returncode == 0:
            logging.info("Azure CLI token refreshed successfully")
            return True
        else:
            logging.error("Failed to refresh Azure CLI token")
            return False
    except Exception as e:
        logging.error(f"Error refreshing token: {e}")
        return False


class TokenRefreshManager:
    """Manages automatic token refresh in a background thread."""
    
    def __init__(self, refresh_interval_minutes=30):
        self.refresh_interval = refresh_interval_minutes * 60  # Convert to seconds
        self.stop_event = threading.Event()
        self.refresh_thread = None
        
    def start(self):
        """Start the token refresh thread."""
        self.refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.refresh_thread.start()
        logging.info(f"Started token refresh manager (refresh every {self.refresh_interval/60} minutes)")
        
    def stop(self):
        """Stop the token refresh thread."""
        if self.refresh_thread:
            self.stop_event.set()
            self.refresh_thread.join()
            logging.info("Stopped token refresh manager")
            
    def _refresh_loop(self):
        """Background loop that refreshes tokens periodically."""
        while not self.stop_event.wait(self.refresh_interval):
            expiry = get_token_expiry()
            if expiry:
                time_until_expiry = expiry - datetime.now()
                logging.info(f"Token expires in: {time_until_expiry}")
                
                # Refresh if less than 45 minutes remaining
                if time_until_expiry < timedelta(minutes=45):
                    refresh_azure_token()
            else:
                # If we can't get expiry, refresh anyway
                refresh_azure_token()


def main(dataset_dir: str, dataset_name: str = None) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Start token refresh manager
    token_manager = TokenRefreshManager(refresh_interval_minutes=30)
    token_manager.start()
    
    try:
        dirname = os.path.abspath(os.path.dirname(__file__))
        azure_config = load_yaml(os.path.join(dirname, "azure_config.yml"))
        
        subscription_id = azure_config["subscription_id"]
        resource_group = azure_config["resource_group"]
        workspace_name = azure_config["workspace_name"]
        
        # Use provided dataset name or fall back to config
        data_name = dataset_name if dataset_name else azure_config["data_name"]
        
        print(f"Uploading dataset '{data_name}' from: {dataset_dir}")
        
        # Check initial token expiry
        expiry = get_token_expiry()
        if expiry:
            time_until_expiry = expiry - datetime.now()
            logging.info(f"Initial token expires in: {time_until_expiry}")
            
            # If less than 10 minutes, refresh immediately
            if time_until_expiry < timedelta(minutes=10):
                logging.warning("Token expires soon, refreshing immediately...")
                refresh_azure_token()
        
        # Use AzureCliCredential specifically to work with refreshed tokens
        credential = AzureCliCredential()
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
        
        # Get dataset size for progress tracking
        total_size = sum(f.stat().st_size for f in Path(dataset_dir).rglob('*') if f.is_file())
        logging.info(f"Total dataset size: {total_size / (1024**3):.2f} GB")
        
        # Create the dataset
        dataset = Data(
            name=data_name,
            description=f"Dataset uploaded from {dataset_dir} (size: {total_size / (1024**3):.2f} GB)",
            path=os.path.abspath(dataset_dir),
            type=AssetTypes.URI_FOLDER,
        )
        
        print(f"Creating/updating dataset '{data_name}'...")
        logging.info("Starting dataset upload...")
        
        # Upload with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ml_client.data.create_or_update(dataset)
                print(f"Successfully uploaded dataset '{data_name}'")
                logging.info("Dataset upload completed successfully")
                break
            except Exception as e:
                logging.error(f"Upload attempt {attempt + 1} failed: {e}")
                if "AuthenticationFailed" in str(e) or "token" in str(e).lower():
                    logging.info("Authentication error detected, refreshing token...")
                    refresh_azure_token()
                    # Recreate the ML client with fresh credentials
                    credential = AzureCliCredential()
                    ml_client = MLClient(
                        credential=credential,
                        subscription_id=subscription_id,
                        resource_group_name=resource_group,
                        workspace_name=workspace_name,
                    )
                
                if attempt == max_retries - 1:
                    raise e
                else:
                    wait_time = (attempt + 1) * 30  # Progressive backoff
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

    finally:
        # Stop the token refresh manager
        token_manager.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a dataset to Azure ML with automatic token refresh")
    
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Local directory path containing the dataset")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Name for the dataset in Azure ML (uses config value if not provided)")
    
    args = parser.parse_args()
    
    main(dataset_dir=args.dataset_dir, dataset_name=args.dataset_name) 