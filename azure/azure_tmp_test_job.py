"""Creates and runs an Azure ML command job."""

import logging
from pathlib import Path
import argparse

from azure.ai.ml import MLClient, Input, Output, PyTorchDistribution, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, Model, ComputeInstance, Data
from azure.identity import DefaultAzureCredential
import os
import pdb
from yondu_gnm.utils.base import load_yaml

CONDA_PATH = Path(Path(__file__).parent, "conda.yml")
CODE_PATH = Path(Path(__file__).parent.parent.parent)
MODEL_NAME = "stablenav-distributed"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    dirname = os.path.abspath(os.path.dirname(__file__))
    azure_config = load_yaml(os.path.join(dirname, "azure_config.yml"))
    subscription_id = azure_config["subscription_id"]
    resource_group = azure_config["resource_group"]
    workspace_name = azure_config["workspace_name"]
    data_name = azure_config["data_name"]
    experiment_name = azure_config["experiment_name"]
    environment_name = azure_config["environment_name"]
    compute_name = azure_config["compute_name"]
    conda_path = os.path.join(dirname, azure_config["conda_path"])
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

    save_location = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/workspaceblobstore/paths/models/ddp_testing"
    # 'azureml://westus.api.azureml.ms/mlflow/v1.0/subscriptions/9c586efc-2916-46e3-a00a-7a003f986647/resourceGroups/yondu_gnm_resource_group/providers/Microsoft.MachineLearningServices/workspaces/yondu_gnm_workspace'
    outputs = {
        "ddp_test": Output(
            type=AssetTypes.URI_FOLDER,
            path=save_location,
        )
    }

    job = command(
        description="Stablenav training",
        experiment_name=experiment_name,
        compute=compute_name,
        inputs=dict(src_dir=Input(path=f"{data_name}@latest")),
        # outputs=dict(model=Output(type=AssetTypes.MLFLOW_MODEL, path=save_location)),
        outputs=outputs,
        code=CODE_PATH,
        environment=f"{environment_name}@latest",  # to use existing env
        # environment=environment, # to remake env
        resources=dict(instance_count=2),
        distribution=dict(type="PyTorch", process_count_per_instance=1),
        # command="python train/azure_tmp_test.py --data_dir ${{input.src_dir}} --output_dir ${{outputs.model}}",
        # command="python train/azure_tmp_test.py --output_dir ${{outputs.ddp_test}}",
        command="python yondu_gnm/train/azure_tmp_test.py --data_dir ${{inputs.src_dir}}",
    )
    job = ml_client.jobs.create_or_update(job)


if __name__ == "__main__":

    main()
