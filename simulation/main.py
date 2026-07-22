import asyncio
import os
import shutil
import uuid

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from simulation.utils import (
    ModelWandbWrapper,
    WandbLogger,
    get_model,
    reserve_run_storage_path,
    set_seed,
)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    results_root = os.path.join(
        os.path.dirname(__file__),
        f"./results/{cfg.experiment.name}",
    )
    os.makedirs(results_root, exist_ok=True)
    run_name = reserve_run_storage_path(results_root, cfg)
    logger = WandbLogger(
        cfg.experiment.name,
        OmegaConf.to_object(cfg),
        debug=cfg.debug,
        run_name=run_name,
    )
    experiment_storage = os.path.join(results_root, run_name)

    def build_wrapper(llm_cfg):
        model = get_model(
            llm_cfg.path,
            seed=cfg.seed,
            backend_name=llm_cfg.backend,
        )
        return ModelWandbWrapper(
            model,
            render=llm_cfg.render,
            wanbd_logger=logger,
            temperature=llm_cfg.temperature,
            top_p=llm_cfg.top_p,
            seed=cfg.seed,
            is_api=True,
            reasoning=OmegaConf.select(llm_cfg, "reasoning", default=None),
        )

    if len(cfg.mix_llm) == 0:
        wrapper = build_wrapper(cfg.llm)
        wrappers = [wrapper] * cfg.experiment.personas.num
        wrapper_framework = wrapper
    else:
        if len(cfg.mix_llm) != cfg.experiment.personas.num:
            raise ValueError(
                f"Length of mix_llm should be equal to personas.num: {cfg.experiment.personas.num}"
            )
        unique_configs = {}
        wrappers = []

        for idx, llm_config in enumerate(cfg.mix_llm):
            llm_config = llm_config.llm
            config_key = (
                llm_config.path,
                llm_config.backend,
                llm_config.temperature,
                llm_config.top_p,
            )
            if config_key not in unique_configs:
                unique_configs[config_key] = build_wrapper(llm_config)

            wrappers.append(unique_configs[config_key])

        llm_framework_config = cfg.framework_model
        config_key = (
            llm_framework_config.path,
            llm_framework_config.backend,
            llm_framework_config.temperature,
            llm_framework_config.top_p,
        )
        if config_key not in unique_configs:
            wrapper_framework = build_wrapper(llm_framework_config)
            unique_configs[config_key] = wrapper_framework
        else:
            wrapper_framework = unique_configs[config_key]

    coding_llm_cfg = OmegaConf.select(cfg, "experiment.contracting.coding_llm", default=None)
    if coding_llm_cfg is not None:
        wrapper_coding = build_wrapper(coding_llm_cfg)
    else:
        wrapper_coding = wrapper_framework

    async def close_wrappers():
        seen = set()
        for wrapper in [*wrappers, wrapper_framework, wrapper_coding]:
            wrapper_id = id(wrapper)
            if wrapper_id in seen:
                continue
            seen.add(wrapper_id)
            await wrapper.aclose()

    async def run_and_close():
        try:
            if cfg.experiment.scenario == "fishing":
                from .scenarios.fishing.run import run as run_scenario_fishing

                await run_scenario_fishing(
                    cfg.experiment,
                    logger,
                    wrappers,
                    wrapper_framework,
                    wrapper_coding,
                    experiment_storage,
                )
            else:
                raise ValueError(f"Unknown experiment.scenario: {cfg.experiment.scenario}")
        finally:
            await close_wrappers()

    asyncio.run(run_and_close())

    # Persist per-run token usage next to the env log for cost accounting.
    try:
        import json as _json

        with open(os.path.join(experiment_storage, "token_usage.json"), "w") as _f:
            _json.dump(ModelWandbWrapper.usage_snapshot(), _f, indent=2)
    except Exception as _e:  # noqa: BLE001 - usage logging must never fail a run
        print(f"[token-usage] failed to write usage snapshot: {_e}")

    hydra_log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    shutil.copytree(f"{hydra_log_path}/.hydra/", f"{experiment_storage}/.hydra/")
    shutil.copy(f"{hydra_log_path}/main.log", f"{experiment_storage}/main.log")
    # shutil.rmtree(hydra_log_path)

    artifact = wandb.Artifact("hydra", type="log")
    artifact.add_dir(f"{experiment_storage}/.hydra/")
    artifact.add_file(f"{experiment_storage}/.hydra/config.yaml")
    artifact.add_file(f"{experiment_storage}/.hydra/hydra.yaml")
    artifact.add_file(f"{experiment_storage}/.hydra/overrides.yaml")
    wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
