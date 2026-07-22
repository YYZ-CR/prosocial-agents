import asyncio
import collections
import datetime
import json
import os
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from simulation.persona.common import PersonaActionHarvesting
from simulation.persona.common import PersonaEnvironment
from simulation.persona.common import PersonaIdentity
from simulation.persona.persona import PersonaAgent
from simulation.utils import ModelWandbWrapper

from .contracting import ContractingOrchestrator
from .environment import FishingConcurrentEnv, FishingPerturbationEnv


def _print_env_summary(env, agent_id, obs, stage: str) -> None:
    """Print a compact environment state snapshot for debugging."""
    state = getattr(env, "internal_global_state", {})
    stock_now = state.get("resource_in_pool")
    stock_before = state.get("resource_before_harvesting")
    catches = state.get("last_collected_resource", {}) or {}
    catches_by_name = {
        env.agent_id_to_name.get(pid, pid): amount for pid, amount in catches.items()
    }
    ordered_catches = ", ".join(
        f"{name}:{catches_by_name[name]}" for name in sorted(catches_by_name)
    ) or "none"
    next_agent_name = env.agent_id_to_name.get(agent_id, agent_id)
    print(
        "[env-summary]"
        f" stage={stage}"
        f" round={env.num_round}"
        f" phase={env.phase}"
        f" next_agent={next_agent_name}"
        f" observed_stock={obs.current_resource_num}"
        f" stock_now={stock_now}"
        f" stock_before_harvest={stock_before}"
        f" last_catches={ordered_catches}"
    )


def make_harvest_report(
    personas: dict[str, PersonaAgent],
    last_rounds_harvest_stats: dict[str, int],
    env,
) -> str:
  """Factual per-agent catch summary for harvest_report / leader context."""
  remaining_fish = env.internal_global_state["resource_in_pool"]
  report = f"There are {remaining_fish} tons of fish left in the lake.\n"
  adverse_event = env.internal_global_state.pop("recent_adverse_event", None)
  if adverse_event is not None:
    event_type = adverse_event.get("type", "adverse event")
    stock_before = adverse_event.get("stock_before", remaining_fish)
    stock_after = adverse_event.get("stock_after", remaining_fish)
    report += (
        "A drought hit the lake since the previous cycle. "
        f"Fish stock changed from {stock_before} to {stock_after} after the {event_type} shock.\n"
    )
  for _, persona in personas.items():
    name = persona.identity.name
    tons = last_rounds_harvest_stats.get(name, 0)
    report += f"\t{name} caught {tons} tons of fish\n"

  return report


def _configure_cognition(cfg) -> bool:
    """Configure cognition module globals from cfg. Returns the debug flag."""
    if cfg.agent.agent_package == "persona_v3":
        from .agents.persona_v3.cognition import utils as cognition_utils

        if cfg.agent.system_prompt == "v3":
            cognition_utils.SYS_VERSION = "v3"
        elif cfg.agent.system_prompt == "v3_p2":
            cognition_utils.SYS_VERSION = "v3_p2"
        elif cfg.agent.system_prompt == "v3_p1":
            cognition_utils.SYS_VERSION = "v3_p1"
        elif cfg.agent.system_prompt == "v3_p3":
            cognition_utils.SYS_VERSION = "v3_p3"
        elif cfg.agent.system_prompt == "v3_nocom":
            cognition_utils.SYS_VERSION = "v3_nocom"
        else:
            cognition_utils.SYS_VERSION = "v1"
        if cfg.agent.cot_prompt == "think_step_by_step":
            cognition_utils.REASONING = "think_step_by_step"
        elif cfg.agent.cot_prompt == "deep_breath":
            cognition_utils.REASONING = "deep_breath"
    else:
        raise ValueError(f"Unknown agent package: {cfg.agent.agent_package}")
    return bool(OmegaConf.select(cfg, "debug", default=False))


def _build_personas(cfg, wrappers, framework_wrapper, experiment_storage, log_to_file):
    """Create and initialise all personas with identities, SVO assignments and cross-references."""
    from .agents.persona_v3 import FishingPersona
    from .agents.persona_v3.cognition.leaders import FisherPopulationType, sample_fisher_svos

    num_personas = cfg.personas.num
    svo_population_str = OmegaConf.select(cfg, "personas.svo_population", default="none") or "none"
    svo_population = FisherPopulationType(svo_population_str)
    svo_angles, svo_types = sample_fisher_svos(svo_population, num_personas)
    log_to_file("svo_assignments", {
        f"persona_{i}": {
            "svo_type": svo_types[i].value,
            "svo_angle": svo_angles[i],
        }
        for i in range(num_personas)
    })

    personas = {
        f"persona_{i}": FishingPersona(
            cfg.agent,
            wrappers[i],
            framework_wrapper,
            os.path.join(experiment_storage, f"persona_{i}"),
            experiment_storage=experiment_storage,
            svo_angle=svo_angles[i],
            svo_type=svo_types[i],
        )
        for i in range(num_personas)
    }

    regen_mode_str = OmegaConf.select(cfg, "env.regen_mode", default="deterministic") or "deterministic"
    identities = {}
    for i in range(num_personas):
        persona_id = f"persona_{i}"
        identities[persona_id] = PersonaIdentity(
            agent_id=persona_id, **cfg.personas[persona_id]
        )
        identities[persona_id].env = PersonaEnvironment(
            regen_min_range=cfg.env.regen_factor_range[0],
            regen_max_range=cfg.env.regen_factor_range[1],
            regen_mode=regen_mode_str,
            num_agents=num_personas,
        )

    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    for persona in personas:
        personas[persona].init_persona(persona, identities[persona], social_graph=None)
    for persona in personas:
        for other_persona in personas:
            personas[persona].add_reference_to_other_persona(personas[other_persona])

    return personas, agent_name_to_id, agent_id_to_name


def _build_env(cfg, experiment_storage, agent_id_to_name, framework_wrapper, coding_wrapper):
    """Instantiate the fishing environment and contracting runtime."""
    env_class = (
        FishingPerturbationEnv
        if cfg.env.class_name == "fishing_perturbation_env"
        else FishingConcurrentEnv
    )
    env = env_class(cfg.env, experiment_storage, agent_id_to_name)
    contracting_runtime = ContractingOrchestrator.from_cfg(
        OmegaConf.select(cfg, "contracting", default=None),
        framework_wrapper,
        coding_wrapper,
        experiment_storage,
        cfg.env.max_num_rounds,
    )
    return env, contracting_runtime


async def _run_sim_loop(
    env,
    personas,
    contracting_runtime,
    agent_id,
    obs,
    curr_round,
    logger,
    debug,
    agent_name_to_id,
    num_personas,
    experiment_storage,
):
    """Run the main simulation loop. Returns round harvest stats."""
    round_harvest_stats = collections.defaultdict(lambda: collections.defaultdict(int))
    STATS_KEYS = [
        "conversation_resource_limit",
        *[f"persona_{i}_collected_resource" for i in range(num_personas)],
    ]

    while True:
        # CONCURRENT PHASES (lake / home): all agents act in parallel.
        if obs.phase in ("lake", "home"):
            agent_order = env._agent_selector.agent_order
            phase_obs = {agent_order[0]: obs, **{
                aid: env._observe(aid) for aid in agent_order[1:]
            }}

            async def _run_agent(aid):
                a = personas[aid]
                return aid, await a.aloop(phase_obs[aid], debug=debug)

            phase_actions = dict(await asyncio.gather(
                *(_run_agent(aid) for aid in agent_order)
            ))

            terminated = False
            for aid in agent_order:
                action = phase_actions[aid]
                agent_id, obs, rewards, termination = env.step(action)
                _print_env_summary(env, agent_id, obs, stage="step")
                if isinstance(action, PersonaActionHarvesting):
                    round_harvest_stats[curr_round][personas[aid].identity.name] = action.quantity
                stats = {}
                for s in STATS_KEYS:
                    if s in action.stats:
                        stats[s] = action.stats[s]
                if np.any(list(termination.values())):
                    logger.log_game(
                        {"num_resource": obs.current_resource_num, **stats},
                        last_log=True,
                    )
                    terminated = True
                    break
                else:
                    logger.log_game({"num_resource": obs.current_resource_num, **stats})
            if terminated:
                break
            if curr_round != env.num_round:
                curr_round = env.num_round
                if contracting_runtime is not None:
                    contracting_runtime.set_round(curr_round)
            logger.save(experiment_storage, agent_name_to_id)
            continue

        if (
            obs.current_location == "restaurant"
        ):
            report_text = make_harvest_report(
                personas,
                round_harvest_stats[curr_round],
                env,
            )
            for p in personas.values():
                p.update_harvest_report(report_text)

        agent = personas[agent_id]
        action = await agent.aloop(obs, debug=debug)

        agent_id, obs, rewards, termination = env.step(action)
        _print_env_summary(env, agent_id, obs, stage="step")

        if isinstance(action, PersonaActionHarvesting):
            round_harvest_stats[curr_round][agent.identity.name] = action.quantity
        stats = {}
        for s in STATS_KEYS:
            if s in action.stats:
                stats[s] = action.stats[s]

        if np.any(list(termination.values())):
            logger.log_game(
                {"num_resource": obs.current_resource_num, **stats},
                last_log=True,
            )
            break
        else:
            logger.log_game({"num_resource": obs.current_resource_num, **stats})
        if curr_round != env.num_round:
            curr_round = env.num_round
            if contracting_runtime is not None:
                contracting_runtime.set_round(curr_round)
        logger.save(experiment_storage, agent_name_to_id)

    return round_harvest_stats


async def run(
    cfg: DictConfig,
    logger: ModelWandbWrapper,
    wrappers: List[ModelWandbWrapper],
    framework_wrapper: ModelWandbWrapper,
    coding_wrapper: ModelWandbWrapper,
    experiment_storage: str,
):
    debug = _configure_cognition(cfg)

    os.makedirs(experiment_storage, exist_ok=True)
    consolidated_log_path = os.path.join(experiment_storage, "consolidated_results.json")

    def log_to_file(log_type, data):
        with open(consolidated_log_path, "a") as f:
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": log_type,
                "data": data,
            }
            f.write(json.dumps(entry) + "\n")

    log_to_file(
        "initialization",
        {
            "experiment_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "config": (
                OmegaConf.to_container(cfg)
                if hasattr(cfg, "to_container")
                else str(cfg)
            ),
        },
    )

    personas, agent_name_to_id, agent_id_to_name = _build_personas(
        cfg, wrappers, framework_wrapper, experiment_storage, log_to_file
    )
    env, contracting_runtime = _build_env(
        cfg, experiment_storage, agent_id_to_name, framework_wrapper, coding_wrapper
    )

    if contracting_runtime is not None:
        for persona in personas.values():
            persona.set_contracting_runtime(contracting_runtime)
        env.set_harvest_enforcer(contracting_runtime.enforce_catches)
        env.set_post_regen_hook(contracting_runtime.on_post_regen)
        env.set_contracting_enabled(True)
    else:
        env.set_contracting_enabled(False)

    agent_id, obs = env.reset()
    _print_env_summary(env, agent_id, obs, stage="reset")
    curr_round = env.num_round
    if contracting_runtime is not None:
        contracting_runtime.set_round(curr_round)
        # A seeded norm must be visible for round 0's harvest, not just from the
        # first restaurant phase onward. No-op unless FISHING_SEED_NORM is set.
        if contracting_runtime.prime_personas(list(personas.values())):
            print("[norm-opt] seeded norm primed into personas before round 0")

    round_harvest_stats = await _run_sim_loop(
        env, personas, contracting_runtime, agent_id, obs, curr_round,
        logger, debug, agent_name_to_id, cfg.personas.num, experiment_storage,
    )

    log_to_file("harvest", round_harvest_stats)
    env.save_log()
    for persona in personas:
        personas[persona].memory.save()
