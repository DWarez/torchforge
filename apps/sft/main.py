import asyncio
import logging
import sys

from apps.sft.recipe import ForgeSFTRecipe
from forge.controller.provisioner import init_provisioner, shutdown
from forge.observability import get_or_create_metric_logger
from forge.types import ProvisionerConfig, LauncherConfig, ServiceConfig, ProcessConfig
from forge.util.config import parse
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run(cfg: DictConfig) -> None:
    logger.info("Spawning recipe...")

    if cfg.get("provisioner", None) is not None:
        provisioner_dict = OmegaConf.to_container(cfg.provisioner, resolve=True)

        services_dict = {}
        if cfg.get("services"):
            for name, svc_cfg in cfg.services.items():
                svc_container = OmegaConf.to_container(svc_cfg, resolve=True)
                services_dict[name] = ServiceConfig(**svc_container)

        actors_dict = {}
        if cfg.get("actors"):
            for name, actor_cfg in cfg.actors.items():
                actor_container = OmegaConf.to_container(actor_cfg, resolve=True)
                actors_dict[name] = ProcessConfig(**actor_container)

        launcher_config = LauncherConfig(
            launcher=provisioner_dict.get("launcher"),
            job_name=provisioner_dict.get("job_name", ""),
            services=services_dict,
            actors=actors_dict,
            slurm_args=provisioner_dict.get("slurm_args", {}),
            cpus_per_task=provisioner_dict.get("cpus_per_task"),
            mem=provisioner_dict.get("mem"),
            gpus_per_node=provisioner_dict.get("gpus_per_node", 8),
        )

        await init_provisioner(
            ProvisionerConfig(launcher_config=launcher_config)
        )
    else:
        await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    actor_cfg = OmegaConf.to_container(cfg.actors.sft_trainer, resolve=True)
    recipe = await ForgeSFTRecipe.options(**actor_cfg).as_actor(cfg)

    logger.info("Created recipe, running setup.")
    await recipe.setup.call()

    logger.info("Recipe has been setup. Training now.")
    await recipe.train.call()

    logger.info("Done training. Clean up")
    await recipe.cleanup.call()

    await recipe.mesh.stop()
    await shutdown()
    logger.info("All done!")


@parse
def main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(main())