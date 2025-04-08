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
    compute_name = azure_config["compute_name"]
    # conda_path = os.path.join(dirname, azure_config["conda_path"])
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

    environment_name = "new_lerobot_env"
    # environment = Environment(
    #     name=environment_name,
    #     image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest",
    #     conda_file="/home/yondu/mnthome/yondu/lerobot/azure/conda.yml",
    # )
    # ml_client.environments.create_or_update(environment)

    # Notice that we specify that we want two nodes/instances, and 4 processes
    # per node/instance.
    # 2 instances * 4 processes per instance = 8 total processes.
    # Azure ML will set the MASTER_ADDR, MASTER_PORT, NODE_RANK, WORLD_SIZE
    # environment variables on each node, in addition to the process-level RANK
    # and LOCAL_RANK environment variables, that are needed for distributed
    # PyTorch training.

    src_dset_name = "new_yondu_teleop_data_revised"

    src_dir = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/workspaceblobstore/paths/datasets/{src_dset_name}"

    dst_dir = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/workspaceblobstore/paths/datasets/lr_dsets/{src_dset_name}_revisedmichaeldata"
    # stats_path = cfg.training.stats_path
    # 'azureml://westus.api.azureml.ms/mlflow/v1.0/subscriptions/9c586efc-2916-46e3-a00a-7a003f986647/resourceGroups/yondu_gnm_resource_group/providers/Microsoft.MachineLearningServices/workspaces/yondu_gnm_workspace'
    # azureml://subscriptions/9c586efc-2916-46e3-a00a-7a003f986647/resourcegroups/yondu_gnm_resource_group/workspaces/yondu_gnm_workspace/datastores/workspaceblobstore/paths/LocalUpload/5ed915e5b86da93d344a81ae3bf8ce31/dpvo_video_dataset/
    inputs = {
        "src_dir": Input(
            type=AssetTypes.URI_FOLDER,
            path=src_dir,
        )
    }
    outputs = {
        "dst_dir": Output(
            type=AssetTypes.URI_FOLDER,
            path=dst_dir,
        ),
    }

    cmd_str = (
        """
    export PYTHONPATH=${PYTHONPATH}:$(realpath ./lerobot/) && echo $PYTHONPATH && 
    python examples/port_datasets/r1_h5_dset.py 
    --src_dir ${{inputs.src_dir}}
    --dst_dir ${{outputs.dst_dir}}
    """.strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )
    # resume=False/True?
    # training.out_dir=${{outputs.out_dir}}

    print(f"Save location: {dst_dir}")
    print(f"Executing command:\n{cmd_str}")

    job = command(
        description="r1 convert to lerobot dset",
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
