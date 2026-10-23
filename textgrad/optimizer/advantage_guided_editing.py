"""Advantage-guided editing for text variables (TextSteer, Section 4.3).

This module is intentionally separate from ``optimizer.py`` so advantage-guided
editing does not change the TGD and TGD-LR implementations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from textgrad import logger
from textgrad.config import validate_engine_or_get_default
from textgrad.engine import EngineLM
from textgrad.variable import Variable

from .optimizer import Optimizer, get_gradient_and_context_text
from .lr_scheduler import NO_LIMIT, build_scheduler
from .optimizer_prompts import (
    CONSTRAINT_PROMPT_ADDITION,
    IN_CONTEXT_EXAMPLE_PROMPT_ADDITION,
    OPTIMIZER_SYSTEM_PROMPT,
    ADVANTAGE_GUIDED_UPDATE_PROMPT_TEMPLATE,
    EDIT_BUDGET_INSTRUCTION,
    EDIT_BUDGET_INSTRUCTION_AUTONOMOUS,
    CANDIDATE_PROPOSAL_PROMPT_TEMPLATE,
)


@dataclass
class ScoredCandidate:
    value: str
    reward: float
    metadata: Optional[Dict[str, Any]] = None
    advantage: Optional[float] = None


class AdvantageGuidedEditing(Optimizer):
    """Advantage-guided editing for discrete text variables.

    Candidate prompt rewrites are sampled from a frozen editor LLM, scored by
    the caller with scalar task rewards, and synthesized into the next prompt
    using their advantages over the current prompt. Like a REINFORCE update it
    weighs candidates by advantage, but it does not compute token
    log-probabilities or update model weights.
    """

    def __init__(
        self,
        parameters: List[Variable],
        verbose: int = 0,
        engine: Union[EngineLM, str] = None,
        constraints: List[str] = None,
        new_variable_tags: List[str] = None,
        optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
        in_context_examples: List[str] = None,
        candidates_per_step: int = 4,
        temperature: float = 0.7,
        baseline_momentum: float = 0.9,
        include_current: bool = True,
        deduplicate_candidates: bool = True,
        advantage_baseline: str = "mean_ema",
    ):
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]
        if candidates_per_step < 1:
            raise ValueError("candidates_per_step must be >= 1")
        if not 0 <= baseline_momentum <= 1:
            raise ValueError("baseline_momentum must be between 0 and 1")

        if advantage_baseline not in {"mean_ema", "current"}:
            raise ValueError("advantage_baseline must be mean_ema or current")
        if advantage_baseline == "current" and not include_current:
            raise ValueError("current advantage baseline requires include_current")
        self.advantage_baseline = advantage_baseline

        self.engine = validate_engine_or_get_default(engine)
        self.verbose = verbose
        self.constraints = constraints if constraints is not None else []
        self.do_constrained = bool(self.constraints)
        self.new_variable_tags = new_variable_tags
        self.optimizer_system_prompt = optimizer_system_prompt.format(
            new_variable_start_tag=new_variable_tags[0],
            new_variable_end_tag=new_variable_tags[1],
        )
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = bool(self.in_context_examples)
        self.candidates_per_step = candidates_per_step
        self.temperature = temperature
        self.baseline_momentum = baseline_momentum
        self.include_current = include_current
        self.deduplicate_candidates = deduplicate_candidates
        self.reward_baselines = {}
        self._sample_counter = 0

    @property
    def constraint_text(self):
        return "\n".join(
            f"Constraint {index + 1}: {constraint}"
            for index, constraint in enumerate(self.constraints)
        )

    def _engine_call(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            return self.engine(prompt, system_prompt=system_prompt, **kwargs)
        except TypeError:
            if "temperature" in kwargs:
                return self.engine(prompt, system_prompt=system_prompt)
            raise

    def _extract_tagged_variable(self, response: str, caller: str) -> str:
        try:
            return response.split(self.new_variable_tags[0])[1].split(
                self.new_variable_tags[1]
            )[0].strip()
        except IndexError:
            logger.error(
                f"{caller} response could not be indexed",
                extra={"optimizer.response": response},
            )
            raise IndexError(
                f"{caller} response could not be indexed. This can happen if the "
                f"optimizer model cannot follow the instructions. Response: {response}"
            )

    def _gradient_text(self, variable: Variable) -> str:
        if not variable.gradients:
            return "No textual gradients were provided for this policy-gradient step."
        gradient_text = get_gradient_and_context_text(variable)
        if isinstance(gradient_text, str):
            return gradient_text
        text_parts = [part for part in gradient_text if isinstance(part, str)]
        if not text_parts:
            return (
                "The gradients contained non-text context; only scalar rewards "
                "are used in this step."
            )
        return "\n".join(text_parts)

    def _constraint_prompt_text(self) -> str:
        if not self.do_constrained:
            return ""
        return CONSTRAINT_PROMPT_ADDITION.format(constraint_text=self.constraint_text)

    def _in_context_examples_text(self, variable: Variable = None) -> str:
        if not self.do_in_context_examples:
            return ""
        variable_desc = (
            variable.get_role_description() if variable is not None else "variable"
        )
        return IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(
            variable_desc=variable_desc,
            in_context_examples="\n".join(self.in_context_examples),
        )

    def _proposal_prompt(
        self,
        variable: Variable,
        sample_index: int,
        num_candidates: int,
        sample_nonce: int,
    ) -> str:
        return CANDIDATE_PROPOSAL_PROMPT_TEMPLATE.format(
            variable_desc=variable.get_role_description(),
            variable_value=variable.value,
            gradient_text=self._gradient_text(variable),
            constraint_text=self._constraint_prompt_text(),
            in_context_examples_text=self._in_context_examples_text(variable),
            sample_index=sample_index,
            num_candidates=num_candidates,
            sample_nonce=sample_nonce,
            new_variable_start_tag=self.new_variable_tags[0],
            new_variable_end_tag=self.new_variable_tags[1],
        )

    def sample_candidates(
        self,
        variable: Variable = None,
        num_candidates: int = None,
        include_current: Optional[bool] = None,
    ) -> List[str]:
        """Sample candidate variable values to be scored by the caller."""
        if variable is None:
            if len(self.parameters) != 1:
                raise ValueError(
                    "sample_candidates requires a variable when optimizing multiple parameters"
                )
            variable = self.parameters[0]
        if num_candidates is None:
            num_candidates = self.candidates_per_step
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if include_current is None:
            include_current = self.include_current

        candidates: List[str] = []
        seen = set()
        if include_current:
            candidates.append(variable.value)
            seen.add(variable.value)

        for sample_index in range(1, num_candidates + 1):
            self._sample_counter += 1
            prompt = self._proposal_prompt(
                variable,
                sample_index,
                num_candidates,
                self._sample_counter,
            )
            response = self._engine_call(
                prompt,
                system_prompt=self.optimizer_system_prompt,
                temperature=self.temperature,
            )
            logger.info(
                "AdvantageGuidedEditing proposal response",
                extra={"optimizer.response": response},
            )
            candidate = self._extract_tagged_variable(
                response,
                "AdvantageGuidedEditing proposal",
            )
            if self.deduplicate_candidates and candidate in seen:
                continue
            candidates.append(candidate)
            seen.add(candidate)
        return candidates

    def _normalize_candidate(self, candidate) -> ScoredCandidate:
        if isinstance(candidate, ScoredCandidate):
            return ScoredCandidate(
                value=candidate.value,
                reward=float(candidate.reward),
                metadata=dict(candidate.metadata or {}),
                advantage=candidate.advantage,
            )
        if isinstance(candidate, Mapping):
            if "value" not in candidate or "reward" not in candidate:
                raise ValueError(
                    "Each candidate dict must include 'value' and 'reward'"
                )
            metadata = dict(candidate.get("metadata") or {})
            for key, value in candidate.items():
                if key not in {"value", "reward", "metadata", "advantage"}:
                    metadata[key] = value
            advantage = candidate.get("advantage")
            return ScoredCandidate(
                value=str(candidate["value"]),
                reward=float(candidate["reward"]),
                metadata=metadata,
                advantage=None if advantage is None else float(advantage),
            )
        if (
            isinstance(candidate, Sequence)
            and not isinstance(candidate, (str, bytes))
            and len(candidate) >= 2
        ):
            metadata = (
                candidate[2]
                if len(candidate) > 2 and isinstance(candidate[2], Mapping)
                else None
            )
            return ScoredCandidate(
                value=str(candidate[0]),
                reward=float(candidate[1]),
                metadata=dict(metadata or {}),
            )
        raise TypeError(
            "Candidates must be dicts, tuples, or "
            "ScoredCandidate objects"
        )

    def _normalize_scored_candidates(
        self,
        scored_candidates,
    ) -> Dict[Variable, List[ScoredCandidate]]:
        if scored_candidates is None:
            raise ValueError(
                "AdvantageGuidedEditing.step requires scored candidates. Call "
                "sample_candidates(), evaluate each candidate with a reward, "
                "then pass them to step()."
            )
        if isinstance(scored_candidates, Mapping):
            if "value" in scored_candidates and "reward" in scored_candidates:
                if len(self.parameters) != 1:
                    raise ValueError(
                        "Pass a {variable: candidates} mapping when optimizing "
                        "multiple parameters"
                    )
                return {
                    self.parameters[0]: [self._normalize_candidate(scored_candidates)]
                }
            normalized = {}
            for variable, candidates in scored_candidates.items():
                if (
                    isinstance(candidates, Mapping)
                    and "value" in candidates
                    and "reward" in candidates
                ):
                    candidates = [candidates]
                normalized[variable] = [
                    self._normalize_candidate(candidate) for candidate in candidates
                ]
            return normalized
        if len(self.parameters) != 1:
            raise ValueError(
                "Pass a {variable: candidates} mapping when optimizing multiple parameters"
            )
        return {
            self.parameters[0]: [
                self._normalize_candidate(candidate) for candidate in scored_candidates
            ]
        }

    def _candidate_baseline_for(self, variable, candidates) -> float:
        """Resolve the baseline from this batch's scored current prompt or EMA."""
        if self.advantage_baseline == "current":
            current = [c for c in candidates if (c.metadata or {}).get("is_current")]
            if len(current) != 1 or current[0].value != variable.value:
                raise ValueError("current baseline requires exactly one matching current prompt")
            baseline = current[0].reward
            # Explicit current mode also replaces caller-supplied advantages.
            for candidate in candidates:
                candidate.advantage = candidate.reward - baseline
            return baseline
        return self._baseline_for(variable, [c.reward for c in candidates])

    def _baseline_for(self, variable: Variable, rewards: List[float]) -> float:
        if variable in self.reward_baselines:
            return self.reward_baselines[variable]
        return sum(rewards) / len(rewards)

    def _update_reward_baseline(self, variable: Variable, rewards: List[float]):
        if self.advantage_baseline == "current":
            return
        mean_reward = sum(rewards) / len(rewards)
        if variable not in self.reward_baselines:
            self.reward_baselines[variable] = mean_reward
            return
        self.reward_baselines[variable] = (
            self.baseline_momentum * self.reward_baselines[variable]
            + (1 - self.baseline_momentum) * mean_reward
        )

    @staticmethod
    def _format_metadata(metadata: Optional[Dict[str, Any]]) -> str:
        if not metadata:
            return ""
        lines = []
        for key in sorted(metadata):
            value = metadata[key]
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(item) for item in value)
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _format_rollouts(
        self,
        candidates: List[ScoredCandidate],
        baseline: float,
    ) -> str:
        ranked = sorted(
            enumerate(candidates, start=1),
            key=lambda item: (
                item[1].advantage if item[1].advantage is not None else 0.0
            ),
            reverse=True,
        )
        rollout_blocks = []
        for original_index, candidate in ranked:
            metadata_text = self._format_metadata(candidate.metadata)
            if metadata_text:
                metadata_text = f"\n<METADATA>\n{metadata_text}\n</METADATA>"
            rollout_blocks.append(
                f'<CANDIDATE index="{original_index}">\n'
                f"<REWARD>{candidate.reward:.6f}</REWARD>\n"
                f"<BASELINE>{baseline:.6f}</BASELINE>\n"
                f"<ADVANTAGE>{candidate.advantage:.6f}</ADVANTAGE>"
                f"{metadata_text}\n"
                f"<VARIABLE>\n{candidate.value}\n</VARIABLE>\n"
                f"</CANDIDATE>"
            )
        return "\n\n".join(rollout_blocks)

    def _update_prompt(
        self,
        variable: Variable,
        candidates: List[ScoredCandidate],
        baseline: float,
    ) -> str:
        prompt = ADVANTAGE_GUIDED_UPDATE_PROMPT_TEMPLATE.format(
            variable_desc=variable.get_role_description(),
            variable_value=variable.value,
            rollout_text=self._format_rollouts(candidates, baseline),
            gradient_text=self._gradient_text(variable),
            constraint_text=self._constraint_prompt_text(),
            in_context_examples_text=self._in_context_examples_text(variable),
            new_variable_start_tag=self.new_variable_tags[0],
            new_variable_end_tag=self.new_variable_tags[1],
        )
        logger.info(
            "AdvantageGuidedEditing update prompt",
            extra={"prompt": prompt},
        )
        return prompt

    def step(self, scored_candidates=None):
        """Apply an update from externally evaluated candidate prompts."""
        candidates_by_parameter = self._normalize_scored_candidates(scored_candidates)
        for parameter in self.parameters:
            candidates = candidates_by_parameter.get(parameter)
            if not candidates:
                logger.info(
                    "AdvantageGuidedEditing: no scored candidates for parameter; "
                    "skipping update"
                )
                continue
            rewards = [candidate.reward for candidate in candidates]
            baseline = self._candidate_baseline_for(parameter, candidates)
            for candidate in candidates:
                if candidate.advantage is None:
                    candidate.advantage = candidate.reward - baseline

            update_prompt = self._update_prompt(parameter, candidates, baseline)
            response = self._engine_call(
                update_prompt,
                system_prompt=self.optimizer_system_prompt,
            )
            logger.info(
                "AdvantageGuidedEditing optimizer response",
                extra={"optimizer.response": response},
            )
            parameter.set_value(
                self._extract_tagged_variable(response, "AdvantageGuidedEditing")
            )
            self._update_reward_baseline(parameter, rewards)
            logger.info(
                "AdvantageGuidedEditing updated text",
                extra={"parameter.value": parameter.value},
            )
            if self.verbose:
                print(
                    f"[AdvantageGuidedEditing] candidates={len(candidates)} "
                    f"mean_reward={sum(rewards) / len(rewards):.3f} "
                    f"best_reward={max(rewards):.3f} baseline={baseline:.3f}"
                )
                print("-----------------------AdvantageGuidedEditing-----------------------")
                print(parameter.value)


class AdvantageGuidedEditingwithLR(AdvantageGuidedEditing):
    """Advantage-guided editing with a textual learning rate (edit budget).

    Candidate sampling, grouped reward evaluation, reward baselines, and
    advantages are unchanged from :class:`AdvantageGuidedEditing`.  The textual
    learning rate only limits the number of distinct edits that the final
    synthesis call may apply to the current variable.
    """

    def __init__(
        self,
        parameters: List[Variable],
        lr_mode: str = "cosine",
        max_lr: int = 4,
        min_lr: int = 2,
        total_steps: int = 1,
        **kwargs,
    ):
        super().__init__(parameters=parameters, **kwargs)
        self.scheduler = build_scheduler(
            lr_mode,
            max_lr=max_lr,
            min_lr=min_lr,
            total_steps=total_steps,
        )
        self.last_learning_rate = None
        self.last_baselines = {}
        self.last_advantages = {}

    def _update_prompt(
        self,
        variable: Variable,
        candidates: List[ScoredCandidate],
        baseline: float,
    ) -> str:
        learning_rate = self.last_learning_rate
        if learning_rate is None:
            learning_rate = self.scheduler.get_lr()
        if learning_rate < NO_LIMIT:
            budget_instruction = EDIT_BUDGET_INSTRUCTION.format(
                learning_rate=learning_rate,
            )
        else:
            budget_instruction = (
                EDIT_BUDGET_INSTRUCTION_AUTONOMOUS
            )
        prompt = ADVANTAGE_GUIDED_UPDATE_PROMPT_TEMPLATE.format(
            variable_desc=variable.get_role_description(),
            variable_value=variable.value,
            rollout_text=self._format_rollouts(candidates, baseline),
            gradient_text=self._gradient_text(variable),
            constraint_text=budget_instruction + self._constraint_prompt_text(),
            in_context_examples_text=self._in_context_examples_text(variable),
            new_variable_start_tag=self.new_variable_tags[0],
            new_variable_end_tag=self.new_variable_tags[1],
        )
        logger.info(
            "AdvantageGuidedEditingwithLR update prompt",
            extra={"prompt": prompt, "textual_learning_rate": learning_rate},
        )
        return prompt

    def step(self, scored_candidates=None):
        self.last_learning_rate = self.scheduler.get_lr()
        candidates_by_parameter = self._normalize_scored_candidates(scored_candidates)
        self.last_baselines = {}
        self.last_advantages = {}
        for parameter, candidates in candidates_by_parameter.items():
            if not candidates:
                continue
            rewards = [candidate.reward for candidate in candidates]
            baseline = self._candidate_baseline_for(parameter, candidates)
            advantages = [
                candidate.advantage
                if candidate.advantage is not None
                else candidate.reward - baseline
                for candidate in candidates
            ]
            self.last_baselines[parameter] = baseline
            self.last_advantages[parameter] = advantages
            if self.verbose:
                print(
                    "[AdvantageGuidedEditingwithLR advantage] "
                    f"baseline={baseline:.3f} advantages={advantages}"
                )
        super().step(scored_candidates)
        if self.verbose:
            budget = (
                "unbounded"
                if self.last_learning_rate >= NO_LIMIT
                else str(self.last_learning_rate)
            )
            print(
                "[AdvantageGuidedEditingwithLR] "
                f"step={self.scheduler.current_step} edit_budget<={budget}"
            )
        self.scheduler.step()
