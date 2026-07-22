"""Runtime orchestration for fishing formal contracts."""

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass, fields
from datetime import timedelta
from typing import Any

from simulation.persona import PersonaAgent
from simulation.persona.common import PersonaIdentity
from simulation.scenarios.common.environment.concurrent_env import (
    get_expiration_next_month,
)

from .coding_agent import CodingAgent
from .contract import (
    ContractMode,
    ContractType,
    EnforcementResult,
    FishingContractState,
    NegotiationProtocol,
)
from .managers import LawContractManager, NLContractManager, NoCommsManager
from .negotiation import (
    FreeChatNegotiationManager,
    MayoralVotingNegotiationManager,
    RoundRobinNegotiationManager,
)
from .tooling import ContractToolbox

try:
  from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional in lightweight smoke tests
  OmegaConf = None


@dataclass
class ContractingConfig:
  enabled: bool = False
  mode: str = ContractMode.NO_COMMUNICATION.value
  negotiation_protocol: str = NegotiationProtocol.ROUND_ROBIN.value
  max_turns: int = 10
  setting_context: str = ""
  # Default enactment rule: more than 3 unique agents agree.
  min_agree_agents: int | None = 4
  max_coding_retries: int = 2
  coding_temperature: float = 0.3
  judge_temperature: float = 0.3
  coding_llm: dict[str, Any] | None = None
  max_tool_calls_per_turn: int = 2


class ContractingOrchestrator:
  def __init__(
      self,
      cfg: Any,
      framework_model,
      coding_model,
      experiment_storage: str,
      max_rounds: int,
  ) -> None:
    self.cfg = self._coerce_cfg(cfg)
    self.framework_model = framework_model
    self.experiment_storage = experiment_storage
    self.max_rounds = max_rounds
    self.current_round = 0
    self.log_path = os.path.join(experiment_storage, "contracting_results.jsonl")
    self.framework_identity = PersonaIdentity("framework", "Framework")
    self.coding_identity = PersonaIdentity("coding_agent", "CodingAgent")
    self._contract_history: list[dict[str, Any]] = []
    # Norm-optimization hooks (see simulation/scenarios/fishing/optimization/).
    # FISHING_SEED_NORM: install this NL law as the active contract at round 0.
    # FISHING_FREEZE_NORM=1: skip all renegotiation so the seeded norm is held
    # fixed for the whole episode -- lets an outer optimizer cleanly attribute the
    # episode's outcome to exactly one norm, and skips (paid) negotiation calls.
    self._seed_norm = (os.getenv("FISHING_SEED_NORM") or "").strip() or None
    self._freeze_norm = os.getenv("FISHING_FREEZE_NORM", "0") == "1"
    self._manager = self._build_manager()
    self._coding_agent = CodingAgent(
        coding_model,
        temperature=self.cfg.coding_temperature,
    )
    self._install_default_contract()

  @classmethod
  def from_cfg(
      cls,
      cfg: Any,
      framework_model,
      coding_model,
      experiment_storage: str,
      max_rounds: int,
  ) -> "ContractingOrchestrator | None":
    if cfg is None:
      return None
    orchestrator = cls(
        cfg,
        framework_model,
        coding_model,
        experiment_storage,
        max_rounds,
    )
    if not orchestrator.enabled:
      return None
    return orchestrator

  @property
  def enabled(self) -> bool:
    return self.cfg.enabled and self.mode in {
        ContractMode.CODE_LAW,
        ContractMode.CODE_NL,
        ContractMode.FREE_CHAT,
    }

  @property
  def mode(self) -> ContractMode:
    return ContractMode(self.cfg.mode)

  @property
  def manager(self):
    return self._manager

  def set_round(self, round_number: int) -> None:
    self.current_round = round_number

  def on_post_regen(self, env: Any) -> None:
    """Called by the env after fish regeneration (contract ``on_round_end`` hook)."""
    mgr = self._manager
    if hasattr(mgr, "on_post_regen"):
      mgr.on_post_regen(env)

  async def aconduct_negotiation(
      self,
      personas: list[PersonaAgent],
      current_time,
      current_resource_num: int,
      sustainability_threshold: int,
      agent_resource_num: dict[str, int],
  ) -> tuple[list[tuple[PersonaIdentity, str]], str, int | None, list[str]]:
    if self._freeze_norm:
      return self._frozen_negotiation(personas)

    named_resource_num = self._normalize_agent_resource_num(personas, agent_resource_num)
    existing_contract = self._manager.get_contract()
    active_law_text = ""
    if existing_contract is not None:
      active_law_text = str(
          existing_contract.metadata.get("nl_contract", existing_contract.content)
      )
    state = FishingContractState(
        round_number=self.current_round,
        fish_population=float(current_resource_num),
        sustainability_threshold=float(sustainability_threshold),
        num_agents=len(personas),
        max_rounds=self.max_rounds,
        speaker_idx=self.current_round,
        setting_context=self.cfg.setting_context,
        negotiation_max_turns=self.cfg.max_turns,
    )
    self._manager.clear_conversation()

    if self.mode == ContractMode.FREE_CHAT:
      negotiation_manager = FreeChatNegotiationManager(
          personas,
          max_turns=self.cfg.max_turns,
      )
    elif self.cfg.negotiation_protocol == NegotiationProtocol.MAYORAL_VOTING.value:
      raise ValueError("Mayoral voting negotiation protocol is not supported in this version.")
      negotiation_manager = MayoralVotingNegotiationManager(
          personas,
          max_turns=self.cfg.max_turns,
          toolbox=ContractToolbox(self._contract_history),
          max_tool_calls=self.cfg.max_tool_calls_per_turn,
      )
    else:
      required_agreements = max(self.cfg.min_agree_agents or 0, 4)
      negotiation_manager = RoundRobinNegotiationManager(
          personas,
          max_turns=self.cfg.max_turns,
          min_agree_agents=required_agreements,
          toolbox=ContractToolbox(self._contract_history),
          max_tool_calls=self.cfg.max_tool_calls_per_turn,
      )

    result = await negotiation_manager.run_nl_negotiation(
        state=state,
        mode=self.mode,
        agent_resource_num=named_resource_num,
        active_law_text=active_law_text,
    )
    for turn in result.conversations:
      self._manager.add_conversation_turn(
          agent_name=turn["agent"],
          message=turn["message"],
          turn_number=turn.get("turn", 0),
          phase=turn.get("phase", "nl_negotiation"),
          html=turn.get("html", ""),
      )

    summary = ""
    final_contract_text = None
    agreed_to_keep_unchanged = False
    if self.mode == ContractMode.FREE_CHAT:
      summary = "Fishers had an open pre-harvest chat. No formal contract objective was applied."
    elif result.nl_contract:
      if "<UNCHANGED>" in result.nl_contract.upper() or result.nl_contract.strip().upper() == "UNCHANGED":
        agreed_to_keep_unchanged = True
      else:
        if self.mode == ContractMode.CODE_NL:
          final_contract_text = result.nl_contract
        elif self.mode == ContractMode.CODE_LAW:
          final_contract_text, coding_response = await self._run_coding_phase(
              personas=personas,
              state=state,
              nl_contract=result.nl_contract,
          )

    if self.mode == ContractMode.FREE_CHAT:
      self._sync_persona_contract_state(personas, "")
    elif final_contract_text:
      agent_names = [persona.identity.name for persona in personas]
      self._manager.set_contract(
          content=final_contract_text,
          agreed_at_round=self.current_round,
          agent_names=agent_names,
      )
      contract = self._manager.get_contract()
      assert contract is not None
      contract.metadata.update({
          "mode": self.mode.value,
          "nl_contract": result.nl_contract,
      })
      self._record_contract_history(contract, result.nl_contract or "")
      self._manager.set_voting_data(
          voting_scheme=self.cfg.negotiation_protocol,
          votes=result.votes,
          agreements=result.agreements,
          passed=True,
          consensus_threshold=max(self.cfg.min_agree_agents or 0, 4),
      )
      await self._store_contract_memories(
          personas=personas,
          current_time=current_time,
          nl_contract=result.nl_contract or "",
      )
      self._sync_persona_contract_state(personas, result.nl_contract or "")
      summary = self._contract_summary(contract.contract_type, result.nl_contract or "")
    else:
      active_contract = self._manager.get_contract()
      if active_contract is not None:
        active_nl_contract = str(
            active_contract.metadata.get("nl_contract", active_contract.content)
        )
        self._sync_persona_contract_state(personas, active_nl_contract)
        if agreed_to_keep_unchanged:
          summary = (
              "Fishers explicitly agreed to keep the active law unchanged. "
              f"The existing law remains in force: {active_nl_contract}"
          )
        elif existing_contract is not None:
          summary = (
              "No new formal contract was adopted. "
              f"The existing law remains in force: {active_nl_contract}"
          )
        else:
          summary = (
              "No new formal contract was adopted. "
              f"The active law remains in force: {active_nl_contract}"
          )
      else:
        self._sync_persona_contract_state(personas, "")
        if agreed_to_keep_unchanged:
          summary = "Fishers explicitly agreed to remain without a formal contract."
        else:
          summary = "No formal contract was adopted for the next round."




    transcript = self._build_transcript(self._manager._conversation)
    html_interactions = [turn.get("html", "") for turn in self._manager._conversation]
    transcript.append((self.framework_identity, summary))
    html_interactions.append(f"<div><strong>FRAMEWORK</strong>: {summary}</div>")

    if self.mode == ContractMode.FREE_CHAT:
      resource_limit = None
    else:
      limit_source = result.nl_contract
      if limit_source is None and self._manager.get_contract() is not None:
        limit_source = str(
            self._manager.get_contract().metadata.get(
                "nl_contract",
                self._manager.get_contract().content,
            )
        )
      resource_limit = self._extract_resource_limit(limit_source)
    limit_text = (
        f"Detected negotiated resource limit: {resource_limit}"
        if resource_limit is not None
        else "Detected negotiated resource limit: N/A"
    )
    transcript.append((self.framework_identity, limit_text))
    html_interactions.append(f"<div><strong>FRAMEWORK</strong>: {limit_text}</div>")

    self._log(
        "negotiation",
        {
            "round": self.current_round,
            "state": asdict(state),
            "summary": summary,
            "result": {
                "nl_contract": result.nl_contract,
                "votes": result.votes,
                "agreements": result.agreements,
            },
            "contract": (
                self._manager.get_contract().to_dict()
                if self._manager.get_contract()
                else None
            ),
        },
    )
    return transcript, summary, resource_limit, html_interactions

  def prime_personas(self, personas: list[PersonaAgent]) -> bool:
    """Push a seeded norm into the personas BEFORE round 0's harvest decisions.

    Without this, a seeded/frozen norm only reaches agents via
    ``_sync_persona_contract_state`` during the restaurant phase -- i.e. AFTER they
    have already fished in round 0 -- so round 0 is effectively norm-free anarchy.
    That is correct for a normal run (nothing is negotiated yet at round 0), but it
    silently invalidates seeded-norm evaluation, where the whole point is that the
    norm governs every round. Only primes when a norm was explicitly seeded.
    """
    if self._seed_norm is None:
      return False
    contract = self._manager.get_contract()
    if contract is None:
      return False
    active_nl = str(contract.metadata.get("nl_contract", contract.content))
    self._sync_persona_contract_state(personas, active_nl)
    self._log("norm_primed", {"round": self.current_round, "nl_contract": active_nl})
    return True

  def _frozen_negotiation(
      self,
      personas: list[PersonaAgent],
  ) -> tuple[list[tuple[PersonaIdentity, str]], str, int | None, list[str]]:
    """Held-fixed norm: no renegotiation. Keep the active (seeded) law in force.

    Returns the same 4-tuple as ``aconduct_negotiation`` so the env loop is
    unchanged, but makes zero negotiation LLM calls.
    """
    active = self._manager.get_contract()
    active_nl = ""
    if active is not None:
      active_nl = str(active.metadata.get("nl_contract", active.content))
    self._sync_persona_contract_state(personas, active_nl)
    summary = (
        "Norm frozen for evaluation (no renegotiation). Active law in force: "
        f"{active_nl}" if active_nl else
        "Norm frozen for evaluation; no formal law in force."
    )
    resource_limit = self._extract_resource_limit(active_nl)
    limit_text = (
        f"Detected negotiated resource limit: {resource_limit}"
        if resource_limit is not None
        else "Detected negotiated resource limit: N/A"
    )
    transcript = [
        (self.framework_identity, summary),
        (self.framework_identity, limit_text),
    ]
    html_interactions = [
        f"<div><strong>FRAMEWORK</strong>: {summary}</div>",
        f"<div><strong>FRAMEWORK</strong>: {limit_text}</div>",
    ]
    self._log(
        "negotiation_frozen",
        {"round": self.current_round, "summary": summary, "resource_limit": resource_limit},
    )
    return transcript, summary, resource_limit, html_interactions

  def enforce_catches(self, decisions: dict[str, float], env) -> EnforcementResult:
    named_decisions = {
        env.agent_id_to_name.get(agent_id, agent_id): amount
        for agent_id, amount in decisions.items()
    }
    if not self._manager.has_contract():
      return EnforcementResult(
          success=True,
          modified_catches=decisions,
          reasoning="No active contract for this round.",
      )
    state = FishingContractState(
        round_number=env.num_round,
        fish_population=float(env.internal_global_state["resource_before_harvesting"]),
        sustainability_threshold=float(env.internal_global_state["sustainability_threshold"]),
        num_agents=len(named_decisions),
        max_rounds=self.max_rounds,
        speaker_idx=env.num_round,
    )
    result = self._manager.enforce(named_decisions, state)
    result.modified_catches = self._normalize_catches(
        {
            agent_id: result.modified_catches.get(env.agent_id_to_name.get(agent_id, agent_id), amount)
            for agent_id, amount in decisions.items()
        },
        decisions,
    )
    result.reward_adjustments = {
        agent_id: float(result.reward_adjustments.get(env.agent_id_to_name.get(agent_id, agent_id), 0.0))
        for agent_id in decisions
    }
    self._log(
        "enforcement",
        {
            "round": env.num_round,
            "state": asdict(state),
            "contract": (
                self._manager.get_contract().to_dict()
                if self._manager.get_contract()
                else None
            ),
            "result": result.to_dict(),
        },
    )
    return result

  def _normalize_catches(
      self,
      modified: dict[str, float],
      original: dict[str, float],
  ) -> dict[str, int]:
    normalized = {}
    for agent, amount in original.items():
      value = modified.get(agent, amount)
      try:
        normalized[agent] = max(0, int(round(float(value))))
      except (TypeError, ValueError):
        normalized[agent] = int(amount)
    return normalized

  def _build_manager(self):
    if self.mode == ContractMode.CODE_LAW:
      return LawContractManager()
    if self.mode == ContractMode.CODE_NL:
      return NLContractManager()
    return NoCommsManager()

  async def _run_coding_phase(
      self,
      personas: list[PersonaAgent],
      state: FishingContractState,
      nl_contract: str,
  ) -> tuple[str | None, str | None]:
    feedback = None
    last_response = None
    active_contract = self._manager.get_contract()
    active_nl_law = ""
    active_code_law = ""
    if active_contract is not None:
      active_nl_law = str(
          active_contract.metadata.get("nl_contract", active_contract.content)
      )
      if active_contract.contract_type == ContractType.PYTHON_LAW:
        active_code_law = active_contract.content

    for retry_attempt in range(self.cfg.max_coding_retries + 1):
      coded_contract, response, html = await self._coding_agent.translate(
          nl_contract=nl_contract,
          mode=self.mode,
          state=state,
          active_nl_law=active_nl_law,
          active_code_law=active_code_law,
          feedback=feedback,
      )
      last_response = response
      self._manager.add_conversation_turn(
          agent_name=self.coding_identity.name,
          message=response,
          turn_number=len(self._manager._conversation),
          phase="coding_agent_translation",
          html=html,
      )
      if coded_contract is not None:
        self._log(
            "code_law_adopted",
            {
                "round": self.current_round,
                "retry_attempt": retry_attempt,
            },
        )
        return coded_contract, response
      feedback = response
    return None, last_response

  async def _store_contract_memories(
      self,
      personas: list[PersonaAgent],
      current_time,
      nl_contract: str,
  ) -> None:
    description = (
        "A formal contract was adopted for the next round: "
        f"{nl_contract}"
    )
    await asyncio.gather(*[
        persona.store.astore_thought(
            description=description,
            created=current_time,
            expiration_delta=timedelta(
                days=(get_expiration_next_month(current_time) - current_time).days
            ),
            always_include=True,
        )
        for persona in personas
    ])

  def _build_transcript(
      self,
      conversations: list[dict[str, Any]],
  ) -> list[tuple[PersonaIdentity, str]]:
    identities = {}
    transcript = []
    for turn in conversations:
      name = turn["agent"]
      if name not in identities:
        identities[name] = PersonaIdentity(name, name)
      transcript.append((identities[name], turn["message"]))
    return transcript

  def _contract_summary(
      self,
      contract_type: ContractType,
      nl_contract: str,
  ) -> str:
    if contract_type == ContractType.PYTHON_LAW:
      return f"Formal Python-law contract adopted for next round: {nl_contract}"
    if contract_type == ContractType.NATURAL_LANGUAGE:
      return f"Formal natural-language contract adopted for next round: {nl_contract}"
    return "No contract was adopted."

  def _sync_persona_contract_state(
      self,
      personas: list[PersonaAgent],
      nl_contract: str,
  ) -> None:
    contract_mode = self.mode.value if nl_contract else ""
    for persona in personas:
      persona.set_formal_contract(nl_contract, contract_mode)

  def _extract_resource_limit(self, text: str | None) -> int | None:
    if not text:
      return None
    patterns = [
        r"(?:at most|no more than|maximum of|limit of|cap of)\s+(\d+)",
        r"catch\s+no\s+more\s+than\s+(\d+)",
        r"catch\s+at\s+most\s+(\d+)",
        r"limit(?:ing)?\s+(?:our|their|the)?\s*catch(?:es)?\s+to\s+(\d+)",
        r"(\d+)\s+tons?\s+(?:each|per person|per fisher|per fisherman)",
    ]
    lowered = text.lower()
    for pattern in patterns:
      matches = re.findall(pattern, lowered)
      if matches:
        return int(matches[0])
    return None

  def _log(self, log_type: str, data: dict[str, Any]) -> None:
    with open(self.log_path, "a") as handle:
      handle.write(json.dumps({"type": log_type, "data": data}) + "\n")

  def _record_contract_history(self, contract, nl_contract: str) -> None:
    self._contract_history.append({
        "round_created": contract.round_created,
        "contract_type": contract.contract_type.value,
        "content": contract.content,
        "nl_contract": nl_contract,
        "agent_names": list(contract.agent_names),
        "metadata": dict(contract.metadata),
    })

  def _default_nl_law(self) -> str:
    if self._seed_norm is not None:
      return self._seed_norm
    return (
        "Unless and until replaced by a duly adopted law, each fisher shall remain "
        "free to choose and act according to their own judgment regarding harvest."
    )

  def _default_code_law(self) -> str:
    return (
        "class DefaultPassthrough(Contract):\n"
        "    VERSION = 1\n"
        "    def resolve(self, month, fish_population, submissions, ctx):\n"
        "        return dict(submissions)\n"
    )

  def _install_default_contract(self) -> None:
    if self.mode in {ContractMode.NO_COMMUNICATION, ContractMode.FREE_CHAT}:
      return
    default_nl_law = self._default_nl_law()
    if self.mode == ContractMode.CODE_LAW:
      self._manager.set_contract(
          content=self._default_code_law(),
          agreed_at_round=-1,
          agent_names=[],
      )
    elif self.mode == ContractMode.CODE_NL:
      self._manager.set_contract(
          content=default_nl_law,
          agreed_at_round=-1,
          agent_names=[],
      )
    else:
      return
    contract = self._manager.get_contract()
    if contract is None:
      return
    contract.proposer = self.framework_identity.name
    contract.enforcement_status = "active"
    contract.metadata.update({
        "mode": self.mode.value,
        "nl_contract": default_nl_law,
        "default_contract": True,
    })
    contract.passed = True
    self._record_contract_history(contract, default_nl_law)

  def _normalize_agent_resource_num(
      self,
      personas: list[PersonaAgent],
      agent_resource_num: dict[str, int],
  ) -> dict[str, int]:
    id_to_name = {
        persona.agent_id: persona.identity.name for persona in personas
    }
    normalized = {}
    for key, value in agent_resource_num.items():
      normalized[id_to_name.get(key, key)] = value
    return normalized

  def _coerce_cfg(self, cfg: Any) -> ContractingConfig:
    if OmegaConf is not None and OmegaConf.is_config(cfg):
      raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    else:
      raw_cfg = cfg
    valid_fields = {field.name for field in fields(ContractingConfig)}
    filtered_cfg = {
        key: value for key, value in raw_cfg.items() if key in valid_fields
    }
    return ContractingConfig(**filtered_cfg)
