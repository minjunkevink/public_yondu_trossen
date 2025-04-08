"""Creates and runs an Azure ML command job."""

import logging
from pathlib import Path
import argparse

import datetime as dt
from azure.ai.ml import MLClient, Input, Output, PyTorchDistribution, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, Model, ComputeInstance, Data
from azure.identity import DefaultAzureCredential
import os
import pdb
from yondu_gnm.utils.base import load_yaml

CODE_PATH = Path(Path(__file__).parent.parent)

"""
TODO move data upload into separate script. use config file for common constants e.g. data_name, sub_id, resourcegroup, etc.
"""


def current_timestamp_str():
    return dt.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    dirname = os.path.abspath(os.path.dirname(__file__))
    azure_config = load_yaml(os.path.join(dirname, "azure_config.yml"))
    subscription_id = azure_config["subscription_id"]
    resource_group = azure_config["resource_group"]
    workspace_name = azure_config["workspace_name"]
    experiment_name = azure_config["experiment_name"]
    environment_name = "new_lerobot_env"
    # compute_name = "cluster-light-cpu"
    # compute_name = "high-ram-gpu-cluster"
    compute_name = "cluster-distributed-gpu-simple"
    # compute_name = "a100-gpu"
    # conda_path = os.path.join(dirname, azure_config["conda_path"])
    conda_path = "/home/yondu/mnthome/yondu/lerobot/azure/conda.yml"
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    ws = ml_client.workspaces.get(workspace_name)
    print(
        f"Workspace location: {ws.location} | Workspace resource group: {ws.resource_group}"
    )

    # environment = Environment(
    #     name=environment_name,
    #     image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest",
    #     conda_file=conda_path,
    # )
    # ml_client.environments.create_or_update(environment)

    dt_str = current_timestamp_str()
    # dset_path = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/workspaceblobstore/paths/datasets/lr_dsets/trimmed_r1_teleop_data_new"

    out_dir = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/workspaceblobstore/paths/datasets/hf_datasets"
    inputs = {
        # "dset_path": Input(
        #     type=AssetTypes.URI_FOLDER,
        #     path=dset_path,
        # )
    }
    outputs = {
        "out_dir": Output(
            type=AssetTypes.URI_FOLDER,
            path=out_dir,
        ),
    }

    cmd_str = (
        """
    export PYTHONPATH=${PYTHONPATH}:$(realpath ./lerobot/) && echo $PYTHONPATH && 
    python examples/download_dataset.py
    --dset_name=lerobot/berkeley_autolab_ur5
    --dst_dir=${{outputs.out_dir}}
    """.strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )
    # resume=False/True?
    # training.out_dir=${{outputs.out_dir}}

    print(f"Save location: {out_dir}")
    print(f"Executing command:\n{cmd_str}")

    job = command(
        description="downloaddset",
        experiment_name=experiment_name,
        compute=compute_name,
        inputs=inputs,
        outputs=outputs,
        code=CODE_PATH,
        environment=f"{environment_name}@latest",  # to use existing env
        # environment=environment, # to remake env
        resources=dict(instance_count=1),
        distribution=dict(type="PyTorch", process_count_per_instance=1),
        # command="python train/azure_tmp_test.py --data_dir ${{input.src_dir}} --output_dir ${{outputs.model}}",
        # command="python train/azure_tmp_test.py --output_dir ${{outputs.ddp_test}}",
        command=cmd_str,
    )
    job = ml_client.jobs.create_or_update(job)


if __name__ == "__main__":
    main()
