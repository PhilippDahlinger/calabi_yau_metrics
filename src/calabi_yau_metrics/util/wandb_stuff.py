import hydra
import wandb
from omegaconf import OmegaConf


def initialize_wandb(config):
    wandb_params = config.wandb
    project_name = wandb_params.get("project_name")
    groupname = wandb_params.get("group_name")
    groupname = groupname[-127:]
    runname = wandb_params.get("run_name")[-127:]
    job_type = wandb_params.get("job_type")[-64:]
    recording_directory: str = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    entity = wandb_params.get("entity")

    start_method = wandb_params.get("start_method")
    settings = (
        wandb.Settings(start_method=start_method)
        if start_method is not None
        else None
    )

    wandb_logger = wandb.init(
        project=project_name,  # name of the whole project
        job_type=job_type,  # name of your experiment
        group=groupname,  # group of identical hyperparameters for different seeds
        name=runname,  # individual repetitions
        dir=recording_directory,  # local directory for wandb recording
        config=OmegaConf.to_container(config, resolve=True),  # full file config
        reinit=False,
        entity=entity,
        settings=settings,
    )
    return wandb_logger