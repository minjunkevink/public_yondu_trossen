"""Creates and runs an Azure ML job for train.py based on YAML configuration."""

import argparse
import logging
import os
from pathlib import Path
import yaml
from datetime import datetime

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


def configure_logging():
    """Configure logging to reduce verbosity of Azure SDK logs."""
    # Set higher logging threshold for noisy libraries
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline").setLevel(logging.ERROR)
    logging.getLogger("azure.identity").setLevel(logging.ERROR)
    logging.getLogger("azure.storage").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    
    # Set up a basic configuration for your own logs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_yaml(path: str):
    """Load a YAML file."""
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary for command-line arguments.
    Example: {'policy': {'type': 'act'}} -> {'policy.type': 'act'}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_to_cmd_args(param_dict):
    """Convert a dictionary to command-line arguments."""
    cmd_args = []
    for k, v in param_dict.items():
        if v is None or v == "null":
            v = "null"
        elif isinstance(v, bool):
            # Handle Python boolean values
            v = "true" if v else "false"
        elif isinstance(v, str) and v.lower() in ("true", "false"):
            # Handle string boolean values, converting to lowercase
            v = v.lower()
        cmd_args.append(f"--{k}={v}")
    return " ".join(cmd_args)


def main():
    parser = argparse.ArgumentParser(description="Run a training job using a YAML config")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--azure", action="store_true", help="Run on Azure (otherwise runs locally)")
    parser.add_argument("--update-status", action="store_true", help="Update experiment status in YAML")
    args = parser.parse_args()
    
    # Configure logging to reduce verbosity
    configure_logging()
    
    # Load experiment config
    exp_config = load_yaml(args.config)
    exp_name = exp_config['parameters']['job_name']
    
    # Extract parameters for command-line arguments
    params = flatten_dict(exp_config['parameters'])
    
    # Update experiment status if requested
    if args.update_status:
        exp_config['metadata']['status'] = "running"
        exp_config['metadata']['date'] = datetime.now().strftime("%Y-%m-%d")
        with open(args.config, 'w') as f:
            yaml.dump(exp_config, f, default_flow_style=False)
    
    # Generate timestamp for logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.azure:
        # Run on Azure ML
        dirname = os.path.abspath(os.path.dirname(__file__))
        azure_config = load_yaml(os.path.join(dirname, "azure_config.yml"))
        
        # Azure configuration
        subscription_id = azure_config["subscription_id"]
        resource_group = azure_config["resource_group"]
        workspace_name = azure_config["workspace_name"]
        compute_name = azure_config["compute_name"]
        environment_name = azure_config["environment_name"]
        
        # Determine dataset name from parameters or metadata
        dataset_name = exp_config['metadata'].get('dataset_name', None)
        if not dataset_name and 'dataset' in params and 'root' in params['dataset']:
            dataset_path = Path(params['dataset.root'])
            if dataset_path.exists():
                dataset_name = dataset_path.name
        
        if not dataset_name:
            dataset_name = "trossen_orange_1_10_400_merged"  # Default if not specified
            
        print(f"Using dataset: {dataset_name}")
        
        # Path to code
        CODE_PATH = Path(Path(__file__).parent.parent)
        
        # Azure ML client setup
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        
        # Get workspace info
        ws = ml_client.workspaces.get(workspace_name)
        print(f"Workspace: {ws.location} | {ws.resource_group}")
        
        # Define output location
        save_location = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/workspaceblobstore/paths/models/{exp_name}"
        
        # Configure outputs
        outputs = {
            "model_output": Output(
                type=AssetTypes.URI_FOLDER,
                path=save_location,
            )
        }
        
        # Log file name 
        log_filename = f"{exp_name}_{timestamp}.log"
        
        # Remove dataset.root from params for Azure runs
        # as we'll use the Azure ML input dataset instead
        azure_params = params.copy()
        if 'dataset.root' in azure_params:
            del azure_params['dataset.root']
        
        # Convert params to command-line args
        cmd_args = dict_to_cmd_args(azure_params)
        
        # Create job with .amlignore file for excluding directories
        job = command(
            description=exp_config['metadata']['description'],
            experiment_name=exp_name,
            compute=compute_name,
            inputs=dict(data_dir=Input(path=f"{dataset_name}@latest")),
            outputs=outputs,
            code=CODE_PATH,
            code_configuration={"ignore_path_in_source_directory": ".amlignore"},
            environment="trossen_lerobot_env@latest",
            resources=dict(instance_count=1),
            command=f"""
                # Add current directory to Python path
                export PYTHONPATH=$PYTHONPATH:$(pwd)
                
                # Verify Python can find the module
                python -c "import sys; print(sys.path); import lerobot; print('lerobot found at:', lerobot.__file__)"

                # Add timestamp to output directory
                export OUTPUT_DIR="${{outputs.model_output}}/{timestamp}"

                # Set environment variables
                export HF_USER="Aravindh25"
                export NVIDIA_TF32_OVERRIDE=1  # Enable TF32 for A100
                
                # Create directories
                mkdir -p ./logs
                mkdir -p /tmp/dataset_cache
                
                # Copy dataset to local SSD
                echo "Copying dataset to local SSD..."
                cp -r ${{inputs.data_dir}}/* /tmp/dataset_cache/
                
                # Run training with fixed parameters
                python lerobot/scripts/train.py {cmd_args} --dataset.root=/tmp/dataset_cache --output_dir=$OUTPUT_DIR --policy.use_amp=true 2>&1 | tee ./logs/{log_filename}
                cp ./logs/{log_filename} ${{outputs.model_output}}/
            """,
        )
        
        # Submit the job
        job = ml_client.jobs.create_or_update(job)
        print(f"Job submitted: {job.name}")
        print(f"Job details URL: {job.studio_url}")
        print(f"Log will be saved as: {log_filename} in the output directory")
        
        # Update YAML with job info if requested
        if args.update_status:
            if 'azure' not in exp_config['metadata']:
                exp_config['metadata']['azure'] = {}
            exp_config['metadata']['azure']['job_id'] = job.name
            exp_config['metadata']['azure']['url'] = job.studio_url
            exp_config['metadata']['azure']['log_file'] = log_filename
            with open(args.config, 'w') as f:
                yaml.dump(exp_config, f, default_flow_style=False)
    else:
        # Run locally with logging (using original params)
        cmd_args = dict_to_cmd_args(params)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"{exp_name}_{timestamp}.log"
        
        cmd = f"python lerobot/scripts/train.py {cmd_args} 2>&1 | tee {log_file}"
        print(f"Running command: {cmd}")
        print(f"Logging output to: {log_file}")
        
        # Update the metadata with the log file location
        if args.update_status:
            if 'local' not in exp_config['metadata']:
                exp_config['metadata']['local'] = {}
            exp_config['metadata']['local']['log_file'] = str(log_file)
            with open(args.config, 'w') as f:
                yaml.dump(exp_config, f, default_flow_style=False)
        
        os.system(cmd)


if __name__ == "__main__":
    main() 