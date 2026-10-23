"""TextSteer prompt optimization runner.

TextSteer (paper Section 4) has four components, all on by default: group
inference, diagnostic preconditioning, advantage-guided editing and the
textual learning rate. Each ablation switches one off with
``--no-group-inference``, ``--no-diagnostic-preconditioning``,
``--no-advantage-guided-editing`` or ``--no-textual-learning-rate``. Switching
off group inference, advantage-guided editing and the textual learning rate
together gives TextGrad.
"""

import argparse
import concurrent.futures
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

tg = None
BackwardContext = None
load_task = None


DEFAULT_SEED = 12
DEFAULT_EVAL_ENGINE = "gpt-4.1-mini"
DEFAULT_TEST_ENGINE = "openrouter-openai/gpt-oss-20b"
DEFAULT_TEST_MAX_TOKENS = 4096
DEFAULT_FINQA_TEST_MAX_TOKENS = 2000
DEFAULT_TASK = "FinQA"
DEFAULT_BATCH_SIZE = 4
DEFAULT_GROUP_TEMPERATURE = 0.0
DEFAULT_MAX_STEPS = 3
DEFAULT_PASS_AT_K = 1
DEFAULT_EVAL_MAX_WORKERS = 2
DEFAULT_PASS_AT_K_TEMPERATURE = 0.7
DEFAULT_NUM_CANDIDATES = 4
DEFAULT_CANDIDATE_TEMPERATURE = 0.7
DEFAULT_CANDIDATE_RESPONSES = 4
DEFAULT_EDIT_BUDGET = 2
DEFAULT_VALIDATION_REVERT_TOLERANCE = 0.03
DEFAULT_FINQA_STARTING_SYSTEM_PROMPT = (
    "The following are financial numerical reasoning questions over report text "
    "and tables. Think step by step, identify the relevant financial figures, "
    "track units carefully, perform the needed calculations, and finish your "
    "answer with a final line exactly formatted as 'Answer: $VALUE' where "
    "$VALUE is the final answer."
)
DEFAULT_REASONING_GYM_STARTING_SYSTEM_PROMPT = (
    "Solve the procedural reasoning task exactly. "
    "Follow the output format requested in the question. "
    "Return only the final answer."
)

RG_ALGORITHMIC_PRESETS = {
    "ab": {
        "easy": {"length": 6},
        "medium": {"length": 10},
    },
    "count_primes": {
        "easy": {"min_n": 1, "max_n": 100},
        "medium": {"min_n": 1, "max_n": 1000},
    },
    "graph_color": {
        "easy": {"num_colors": 3, "min_num_vertices": 5, "max_num_vertices": 6, "edge_probability": 0.15},
        "medium": {"num_colors": 3, "min_num_vertices": 8, "max_num_vertices": 10, "edge_probability": 0.2},
    },
}


def get_default_starting_prompt(task_name):
    if task_name == "FinQA":
        return DEFAULT_FINQA_STARTING_SYSTEM_PROMPT
    if task_name.startswith("RG_"):
        return DEFAULT_REASONING_GYM_STARTING_SYSTEM_PROMPT
    raise ValueError(f"No default starting prompt for {task_name}; pass --starting-prompt")


def load_textgrad():
    global tg, BackwardContext, load_task
    if tg is not None:
        return

    import textgrad as _tg
    from textgrad.autograd.function import BackwardContext as _BackwardContext
    from textgrad.tasks import load_task as _load_task

    tg = _tg
    BackwardContext = _BackwardContext
    load_task = _load_task


RG_DIAGNOSTIC_MODES = {
    "RG_ab": "ab_top1",
    "RG_count_primes": "count_primes_process_top1",
    "RG_graph_color": "graph_color_top1",
}

# Diagnostic modes that read the target model's own reasoning trace.
RG_PROCESS_TRACE_MODES = frozenset({"count_primes_process_top1"})


def build_test_engine(args):
    if args.test_engine.startswith("openrouter-"):
        return tg.get_engine(
            engine_name=args.test_engine,
            max_tokens=args.test_max_tokens,
            request_timeout=args.openrouter_request_timeout,
            max_retries=args.openrouter_max_retries,
            reasoning_enabled={
                "provider-default": None, "on": True, "off": False,
            }.get(args.openrouter_reasoning),
            reasoning_effort=(
                args.openrouter_reasoning
                if args.openrouter_reasoning in {"low", "medium", "high"} else None
            ),
            transient_error_policy=args.openrouter_transient_error_policy,
            provider_sort=args.openrouter_provider_sort,
            provider=args.openrouter_provider,
        )
    return tg.get_engine(engine_name=args.test_engine)


def build_finqa_eval_fn(args):
    from textgrad.autograd.string_based_ops import StringBasedFunction
    from textgrad.tasks.finqa import finqa_equality_fn
    from textgrad.tasks.finqa_diagnoser import FinQADiagnoser

    if not args.diagnostic_preconditioning:
        return StringBasedFunction(
            finqa_equality_fn,
            function_purpose=(
                "Deterministically check whether the predicted FinQA final answer "
                "matches the ground truth, using generic TextGrad feedback without "
                "the FinQA diagnosis template."
            ),
        )
    return FinQADiagnoser(
        include_gold_program=True,
        include_gold_program_re=True,
        include_gold_inds=True,
    )


def build_task_kwargs(args):
    task_kwargs = {}
    if args.task.startswith("RG_"):
        rg_dataset_name = args.task[len("RG_"):]
        task_kwargs.update(
            dict(
                base_seed=args.seed,
                train_size=args.rg_train_size,
                val_size=args.rg_val_size,
                test_size=args.rg_test_size,
                use_projected_diagnostics=args.diagnostic_preconditioning,
                diagnostic_mode=RG_DIAGNOSTIC_MODES.get(args.task),
            )
        )
        if args.rg_difficulty in {"easy", "medium"} and rg_dataset_name in RG_ALGORITHMIC_PRESETS:
            task_kwargs.update(RG_ALGORITHMIC_PRESETS[rg_dataset_name][args.rg_difficulty])

        explicit_config = getattr(args, "rg_config", {}).get(rg_dataset_name)
        if explicit_config:
            # Explicit generator arguments override named difficulty presets.
            task_kwargs.update(explicit_config)
    return task_kwargs


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def call_eval_fn(x, y, response, eval_fn):
    """Call evaluators with the richest context they support."""
    if getattr(eval_fn, "uses_question_context", False):
        inputs = dict(question=x, prediction=response, ground_truth_answer=y)
        return eval_fn(inputs=inputs)

    try:
        return eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
    except Exception:
        eval_output_variable = eval_fn([x, y, response])
        return eval_output_variable


def score_response(x, y, response, eval_fn):
    eval_output_variable = call_eval_fn(x, y, response, eval_fn)
    try:
        return int(eval_output_variable.value)
    except Exception:
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


class SizedDatasetView:
    """Read-only prefix view used by the --train/--val/--test-size arguments."""

    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = min(int(size), len(dataset))

    def __getitem__(self, index):
        return self.dataset[int(index)]

    def __len__(self):
        return self.size

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def apply_split_size(dataset, size):
    if size is None:
        return dataset
    return SizedDatasetView(dataset, size)


def parse_json_object_argument(value, parser, argument_name):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        parser.error(f"{argument_name} must be valid JSON: {exc}")
    if not isinstance(parsed, dict):
        parser.error(f"{argument_name} must decode to a JSON object.")
    for key, item in parsed.items():
        if not isinstance(key, str):
            parser.error(f"{argument_name} keys must be strings.")
        if item is not None and not isinstance(item, dict):
            parser.error(
                f"{argument_name} value for {key!r} must be a JSON object."
            )
    return parsed


def diagnostic_mode_label(args):
    """Name the task diagnoser; recorded in the candidate metadata for the editor."""
    if not args.diagnostic_preconditioning:
        return "none"
    if args.task.startswith("RG_"):
        return f"reasoning_gym:{RG_DIAGNOSTIC_MODES[args.task]}"
    if args.task == "FinQA":
        return "finqa:top1"
    return "none"


def drop_cached_response(
    engine,
    prompt,
    system_prompt=None,
    temperature=0,
    max_tokens=None,
    top_p=0.99,
):
    """Best-effort cache busting using the active engine's cache-key schema."""
    if not hasattr(engine, "cache"):
        return
    sys_prompt = system_prompt if system_prompt else getattr(engine, "system_prompt", "")
    if hasattr(engine, "bypass_cache_once"):
        engine.bypass_cache_once(
            prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return
    if hasattr(engine, "_request_cache_key"):
        cache_key = engine._request_cache_key(
            prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    else:
        cache_key = sys_prompt + prompt
    try:
        if cache_key in engine.cache:
            del engine.cache[cache_key]
    except Exception:
        pass


def sampled_eval_response(
    model,
    x,
    sample_idx,
    pass_at_k,
    temperature,
    add_sample_tag=True,
    use_eval_cache=True,
):
    query = x.value
    if add_sample_tag and pass_at_k > 1:
        query = (
            f"{query}\n\n"
            f"Sampling note: solve independently as completion {sample_idx + 1}/{pass_at_k}. "
            "Do not mention this sampling note."
        )

    system_prompt_value = model.system_prompt.value if model.system_prompt else None
    if not use_eval_cache:
        drop_cached_response(
            model.engine,
            query,
            system_prompt_value,
            temperature=temperature,
        )

    response_text = model.engine(query, system_prompt=system_prompt_value, temperature=temperature)
    return tg.Variable(
        response_text,
        requires_grad=False,
        role_description="response from the language model",
    )


def sampled_training_response(
    model,
    x,
    sample_idx,
    group_size,
    temperature,
    add_sample_tag=True,
):
    """Sample a training completion while preserving the TextGrad backward path."""
    query = x.value
    if add_sample_tag and group_size > 1:
        query = (
            f"{query}\n\n"
            f"Sampling note: solve independently as completion {sample_idx + 1}/{group_size}. "
            "Do not mention this sampling note."
        )
    x_for_call = tg.Variable(
        query,
        requires_grad=False,
        role_description=x.get_role_description(),
    )

    system_prompt_value = model.system_prompt.value if model.system_prompt else None
    # Training groups must be freshly sampled. Some engine cache keys do not
    # include temperature or other sampling kwargs.
    drop_cached_response(
        model.engine,
        x_for_call.value,
        system_prompt_value,
        temperature=temperature,
    )
    trace = None
    if getattr(model.engine, "capture_traces", False) and hasattr(model.engine, "generate_with_trace"):
        response_text, trace = model.engine.generate_with_trace(
            x_for_call.value,
            system_prompt=system_prompt_value,
            temperature=temperature,
        )
    else:
        response_text = model.engine(
            x_for_call.value,
            system_prompt=system_prompt_value,
            temperature=temperature,
        )
    predecessors = [model.system_prompt, x_for_call] if model.system_prompt else [x_for_call]
    response = tg.Variable(
        value=response_text,
        predecessors=predecessors,
        role_description="sampled response from the language model",
    )
    if trace is not None:
        response.generation_trace = trace
    response.set_grad_fn(
        BackwardContext(
            backward_fn=model.llm_call.backward,
            response=response,
            prompt=x_for_call.value,
            system_prompt=system_prompt_value,
        )
    )
    return response


def parse_eval_score(eval_output_variable, eval_fn):
    try:
        return int(eval_output_variable.value)
    except Exception:
        return int(eval_fn.parse_output(eval_output_variable))


def sample_group(
    x_raw,
    y_raw,
    eval_fn,
    model,
    group_size,
    group_temperature,
    add_sample_tag=True,
):
    """Sample and score a fixed group of training responses for one question."""
    x = tg.Variable(x_raw, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y_raw, requires_grad=False, role_description="correct answer for the query")

    group_correct = []
    group_losses = []
    group_responses = []
    for group_sample_idx in range(group_size):
        response = sampled_training_response(
            model=model,
            x=x,
            sample_idx=group_sample_idx,
            group_size=group_size,
            temperature=group_temperature,
            add_sample_tag=add_sample_tag,
        )
        eval_output_variable = call_eval_fn(x, y, response, eval_fn)
        group_correct.append(parse_eval_score(eval_output_variable, eval_fn))
        group_losses.append(eval_output_variable)
        group_responses.append(response)

    return group_correct, group_losses, group_responses


def group_inference(
    x_raw,
    y_raw,
    eval_fn,
    model,
    group_temperature,
    add_sample_tag=True,
):
    """Group inference for one question (Section 4.1).

    Samples cumulative groups of 2, 4, 6 and 8 responses, stopping at the first
    group with both correct and wrong responses.
    """
    x = tg.Variable(x_raw, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y_raw, requires_grad=False, role_description="correct answer for the query")
    group_correct = []
    group_losses = []
    group_responses = []
    for size in (2, 4, 6, 8):
        for group_sample_idx in range(len(group_correct), size):
            response = sampled_training_response(
                model=model,
                x=x,
                sample_idx=group_sample_idx,
                group_size=8,
                temperature=group_temperature,
                add_sample_tag=add_sample_tag,
            )
            eval_output_variable = call_eval_fn(x, y, response, eval_fn)
            group_correct.append(parse_eval_score(eval_output_variable, eval_fn))
            group_losses.append(eval_output_variable)
            group_responses.append(response)
        if any(group_correct) and not all(group_correct):
            break
    return group_correct, group_losses, group_responses


def eval_sample(
    item,
    eval_fn,
    model,
    pass_at_k=DEFAULT_PASS_AT_K,
    eval_temperature=0.0,
    add_sample_tag=True,
    use_eval_cache=True,
):
    """Evaluate whether any of k model answers is correct for one sample."""
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")

    for sample_idx in range(pass_at_k):
        response = sampled_eval_response(
            model=model,
            x=x,
            sample_idx=sample_idx,
            pass_at_k=pass_at_k,
            temperature=eval_temperature,
            add_sample_tag=add_sample_tag,
            use_eval_cache=use_eval_cache,
        )
        if score_response(x, y, response, eval_fn):
            return 1
    return 0


def eval_dataset(
    test_set,
    eval_fn,
    model,
    max_samples=None,
    max_workers=2,
    pass_at_k=DEFAULT_PASS_AT_K,
    eval_temperature=0.0,
    add_sample_tag=True,
    use_eval_cache=True,
):
    if max_samples is None:
        max_samples = len(test_set)

    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            futures.append(
                executor.submit(
                    eval_sample,
                    sample,
                    eval_fn,
                    model,
                    pass_at_k,
                    eval_temperature,
                    add_sample_tag,
                    use_eval_cache,
                )
            )
            if len(futures) >= max_samples:
                break

        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"pass@{pass_at_k}: {np.mean(accuracy_list)}")

    return accuracy_list


def diagnose_current_prompt(
    model,
    batch_items,
    eval_fn,
    args,
):
    """Attach the task diagnoser's feedback to the current prompt.

    Used without group inference: one response per batch question is diagnosed.
    Candidate rewards remain deterministic and do not invoke the critic again.
    """
    if diagnostic_mode_label(args) == "none":
        return []

    losses = []
    records = []
    for sample_idx, (x_raw, y_raw) in enumerate(batch_items):
        x = tg.Variable(
            x_raw,
            requires_grad=False,
            role_description="query to the language model",
        )
        y = tg.Variable(
            y_raw,
            requires_grad=False,
            role_description="correct answer for the query",
        )
        response = model(x)
        eval_output = call_eval_fn(x, y, response, eval_fn)
        losses.append(eval_output)
        records.append(
            {
                "sample_idx": sample_idx,
                "x_raw": x_raw,
                "y_raw": y_raw,
                "score": parse_eval_score(eval_output, eval_fn),
                "response": response,
                "eval_output": eval_output,
            }
        )

    if hasattr(eval_fn, "begin_step"):
        eval_fn.begin_step()
    try:
        if losses:
            tg.sum(losses).backward()
    finally:
        if hasattr(eval_fn, "end_step"):
            eval_fn.end_step()
    return records


def score_candidates_single_response(
    optimizer,
    system_prompt,
    model,
    batch_items,
    eval_fn,
    args,
):
    """Sample candidates; score each by one response per batch question.

    Used without group inference.
    """
    old_prompt = system_prompt.get_value()
    candidates = optimizer.sample_candidates(
        system_prompt,
        num_candidates=args.num_candidates,
        include_current=True,
    )
    diagnostic_mode = diagnostic_mode_label(args)
    scored_candidates = []
    try:
        for candidate_idx, candidate_prompt in enumerate(candidates):
            system_prompt.set_value(candidate_prompt)
            batch_scores = [
                eval_sample(
                    item,
                    eval_fn,
                    model,
                    pass_at_k=1,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                )
                for item in batch_items
            ]
            reward = float(np.mean(batch_scores)) if batch_scores else 0.0
            metadata = {
                "candidate_index": candidate_idx,
                "is_current": candidate_idx == 0 and candidate_prompt == old_prompt,
                "batch_scores": batch_scores,
                "diagnostic_mode": diagnostic_mode,
            }
            scored_candidates.append(
                {
                    "value": candidate_prompt,
                    "reward": reward,
                    "metadata": metadata,
                }
            )
            if args.optimizer_verbose:
                current_marker = " current" if metadata["is_current"] else ""
                print(
                    f"[candidate reward] candidate {candidate_idx}{current_marker}: "
                    f"reward={reward:.3f}, scores={batch_scores}"
                )
    finally:
        system_prompt.set_value(old_prompt)
    compute_advantages(optimizer, system_prompt, scored_candidates)
    return scored_candidates


def score_candidates(
    optimizer,
    system_prompt,
    model,
    batch_items,
    eval_fn,
    args,
):
    """Sample K candidates and score each on the selected questions (Section 4.3).

    The current prompt is candidate 0. Each candidate's reward is the mean
    correctness of J freshly sampled responses per selected question.
    """
    old_prompt = system_prompt.get_value()
    candidates = optimizer.sample_candidates(
        system_prompt,
        num_candidates=args.num_candidates,
        include_current=True,
    )
    diagnostic_mode = diagnostic_mode_label(args)
    scored_candidates = []
    try:
        for candidate_idx, candidate_prompt in enumerate(candidates):
            system_prompt.set_value(candidate_prompt)
            batch_group_correct = []
            for x_raw, y_raw in batch_items:
                group_correct, _, _ = sample_group(
                    x_raw=x_raw,
                    y_raw=y_raw,
                    eval_fn=eval_fn,
                    model=model,
                    group_size=args.candidate_responses,
                    group_temperature=args.group_temperature,
                    add_sample_tag=args.add_sample_tag,
                )
                batch_group_correct.append(group_correct)

            flat_scores = [
                score
                for group_correct in batch_group_correct
                for score in group_correct
            ]
            reward = float(np.mean(flat_scores)) if flat_scores else 0.0
            batch_scores = [
                float(np.mean(group_correct))
                for group_correct in batch_group_correct
            ]
            metadata = {
                "candidate_index": candidate_idx,
                "is_current": candidate_idx == 0 and candidate_prompt == old_prompt,
                "batch_scores": batch_scores,
                "batch_group_correct": batch_group_correct,
                "group_size": args.candidate_responses,
                "diagnostic_mode": diagnostic_mode,
            }
            scored_candidates.append(
                {
                    "value": candidate_prompt,
                    "reward": reward,
                    "metadata": metadata,
                }
            )
            if args.optimizer_verbose:
                current_marker = " current" if metadata["is_current"] else ""
                print(
                    f"[candidate reward] candidate {candidate_idx}"
                    f"{current_marker}: reward={reward:.3f}, "
                    f"group_scores={batch_group_correct}"
                )
    finally:
        system_prompt.set_value(old_prompt)
    compute_advantages(optimizer, system_prompt, scored_candidates)
    return scored_candidates


def compute_advantages(optimizer, system_prompt, candidates):
    """Set A_k = R_k - R_0 (Eq. 5), even when the improvement gate skips editing."""
    current = [c for c in candidates if c.get("metadata", {}).get("is_current")]
    if len(current) != 1 or current[0]["value"] != system_prompt.value:
        raise ValueError("current baseline requires exactly one matching current prompt")
    baseline = float(current[0]["reward"])
    for candidate in candidates:
        candidate["advantage"] = float(candidate["reward"]) - baseline
    optimizer.last_baselines = {system_prompt: baseline}
    optimizer.last_advantages = {system_prompt: [c["advantage"] for c in candidates]}


def improvement_gate(scored_candidates):
    """Improvement gate h = 1[max_k A_k > 0] (Eq. 5): edit only if a candidate beats the current prompt."""
    current = [
        candidate
        for candidate in scored_candidates
        if candidate.get("metadata", {}).get("is_current", False)
    ]
    if len(current) != 1:
        raise ValueError("the improvement gate requires exactly one scored current prompt")

    current_reward = float(current[0]["reward"])
    generated = [
        candidate
        for candidate in scored_candidates
        if not candidate.get("metadata", {}).get("is_current", False)
    ]
    best_candidate_reward = (
        max(float(candidate["reward"]) for candidate in generated)
        if generated
        else None
    )
    return {
        "current_reward": current_reward,
        "best_candidate_reward": best_candidate_reward,
        "should_update": (
            best_candidate_reward is not None
            and best_candidate_reward > current_reward
        ),
    }


def validation_revert_decision(candidate_performance, validation_history, tolerance=0.0):
    """Decide whether a candidate falls outside the best-score trust band."""
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("validation revert tolerance must be between 0 and 1")
    if not validation_history:
        raise ValueError("validation history must contain at least one score")

    historical_scores = [float(np.mean(score)) for score in validation_history]
    reference_performance = max(historical_scores)
    performance_drop = reference_performance - float(candidate_performance)
    rejected = performance_drop > tolerance + 1e-12
    return rejected, reference_performance, performance_drop


def run_validation_revert(
    system_prompt,
    results,
    model,
    eval_fn,
    val_set,
    max_val_samples=None,
    pass_at_k=DEFAULT_PASS_AT_K,
    eval_temperature=0.0,
    add_sample_tag=True,
    use_eval_cache=True,
    max_workers=DEFAULT_EVAL_MAX_WORKERS,
    tolerance=DEFAULT_VALIDATION_REVERT_TOLERANCE,
):
    val_performance = np.mean(
        eval_dataset(
            val_set,
            eval_fn,
            model,
            max_samples=max_val_samples,
            max_workers=max_workers,
            pass_at_k=pass_at_k,
            eval_temperature=eval_temperature,
            add_sample_tag=add_sample_tag,
            use_eval_cache=use_eval_cache,
        )
    )
    previous_performance = np.mean(results["validation_acc"][-1])
    rejected, reference_performance, performance_drop = validation_revert_decision(
        val_performance,
        results["validation_acc"],
        tolerance=tolerance,
    )
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    print("validation_reference_performance: ", reference_performance)
    print("validation_performance_drop: ", performance_drop)
    print("validation_revert_tolerance: ", tolerance)
    print("validation_acceptance_threshold: ", reference_performance - tolerance)
    previous_prompt = results["prompt"][-1]

    if rejected:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)
    return rejected


def build_prompt_optimizer(args, system_prompt, eval_engine):
    """Construct the prompt optimizer for the enabled components.

    The backward model (``--eval-engine``) also writes the candidates and the
    update. Without advantage-guided editing the textual gradient is applied
    with the TextGrad update.
    """
    if args.advantage_guided_editing:
        print(
            f"Advantage-guided editing: editor={args.eval_engine} "
            f"diagnostics={diagnostic_mode_label(args)}"
        )
        editing_kwargs = dict(
            parameters=[system_prompt],
            engine=eval_engine,
            verbose=args.optimizer_verbose,
            candidates_per_step=args.num_candidates,
            temperature=args.candidate_temperature,
            include_current=True,
            advantage_baseline="current",
        )
        if args.textual_learning_rate:
            print(f"Textual learning rate: edit budget L={args.edit_budget}")
            return tg.AdvantageGuidedEditingwithLR(
                lr_mode="constant",
                max_lr=args.edit_budget,
                min_lr=args.edit_budget,
                total_steps=args.max_steps,
                **editing_kwargs,
            )
        return tg.AdvantageGuidedEditing(**editing_kwargs)
    if args.textual_learning_rate:
        print(f"Textual learning rate: edit budget L={args.edit_budget}")
        return tg.TextualGradientDescentwithLR(
            parameters=[system_prompt],
            lr_mode="constant",
            max_lr=args.edit_budget,
            min_lr=args.edit_budget,
            total_steps=args.max_steps,
            engine=eval_engine,
            verbose=args.optimizer_verbose,
        )
    return tg.TextualGradientDescent(
        parameters=[system_prompt],
        engine=eval_engine,
        verbose=args.optimizer_verbose,
    )


def build_models_and_data(args):
    load_textgrad()
    set_seed(args.seed)
    load_dotenv(override=True)
    if args.starting_prompt is None:
        args.starting_prompt = get_default_starting_prompt(args.task)

    llm_api_eval = tg.get_engine(engine_name=args.eval_engine)
    llm_api_test = build_test_engine(args)
    rg_diagnostic_mode = RG_DIAGNOSTIC_MODES.get(args.task)
    if args.diagnostic_preconditioning and rg_diagnostic_mode in RG_PROCESS_TRACE_MODES:
        if not hasattr(llm_api_test, "generate_with_trace"):
            raise ValueError(
                f"The {rg_diagnostic_mode} diagnoser needs a test engine that "
                "returns reasoning traces (openrouter-*)"
            )
        llm_api_test.capture_traces = True
    tg.set_backward_engine(llm_api_eval, override=True)
    print(f"Eval engine: {args.eval_engine} via OPENAI_API_KEY")
    if args.test_engine.startswith("openrouter-"):
        print(
            f"Test engine: {args.test_engine} via OPENROUTER_API_KEY "
            f"(max_tokens={args.test_max_tokens}, "
            f"reasoning={args.openrouter_reasoning}, "
            f"request_timeout={args.openrouter_request_timeout}s, "
            f"max_retries={args.openrouter_max_retries}, "
            f"provider_sort={args.openrouter_provider_sort}, "
            f"provider={args.openrouter_provider}, "
            f"transient_error_policy={args.openrouter_transient_error_policy})"
        )
    else:
        print(f"Test engine: {args.test_engine}")

    train_set, val_set, test_set, eval_fn = load_task(
        args.task,
        evaluation_api=llm_api_eval,
        **build_task_kwargs(args),
    )
    train_set = apply_split_size(train_set, getattr(args, "train_size", None))
    val_set = apply_split_size(val_set, getattr(args, "val_size", None))
    test_set = apply_split_size(test_set, getattr(args, "test_size", None))
    if args.task == "FinQA":
        eval_fn = build_finqa_eval_fn(args)
    print(f"Diagnostic preconditioning: {diagnostic_mode_label(args)}")
    print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
    print(args.starting_prompt)

    system_prompt = tg.Variable(
        args.starting_prompt,
        requires_grad=True,
        role_description=(
            "structured system prompt to a somewhat capable language model "
            "that specifies the behavior and strategies for the QA task"
        ),
    )
    model = tg.BlackboxLLM(llm_api_test, system_prompt)

    optimizer = build_prompt_optimizer(args, system_prompt, eval_engine=llm_api_eval)

    return train_set, val_set, test_set, eval_fn, system_prompt, model, optimizer


def initialize_results(args, val_set, test_set, eval_fn, model, system_prompt):
    results = {
        "test_acc": [],
        "prompt": [],
        "validation_acc": [],
        "optimizer_step_applied": [],
    }
    if args.group_inference:
        results.update(
            {
                "train_group_sample_counts": [],
                "train_group_acc": [],
                "train_acc": [],
                "optimized_group_indices": [],
            }
        )
    if args.textual_learning_rate:
        results["textual_learning_rate"] = []
    if args.advantage_guided_editing:
        results.update(
            {
                "candidates": [],
                "candidate_rewards": [],
                "baseline_rewards": [],
                "advantages": [],
                "improvement_gate": [],
            }
        )
    results["test_acc"].append(
        eval_dataset(
            test_set,
            eval_fn,
            model,
            max_samples=args.max_test_samples,
            max_workers=args.eval_max_workers,
            pass_at_k=args.pass_at_k,
            eval_temperature=args.eval_temperature,
            add_sample_tag=args.add_sample_tag,
            use_eval_cache=args.use_eval_cache,
        )
    )
    if args.skip_validation:
        results["validation_acc"].append([])
    else:
        results["validation_acc"].append(
            eval_dataset(
                val_set,
                eval_fn,
                model,
                max_samples=args.max_val_samples,
                max_workers=args.eval_max_workers,
                pass_at_k=args.pass_at_k,
                eval_temperature=args.eval_temperature,
                add_sample_tag=args.add_sample_tag,
                use_eval_cache=args.use_eval_cache,
            )
        )
    results["prompt"].append(system_prompt.get_value())
    return results


def prewarm_initial_prompt_cache(args):
    """Populate the initial-prompt response cache for every dataset split."""
    train_set, val_set, test_set, eval_fn, system_prompt, model, _ = build_models_and_data(args)
    results = {
        "train_acc": [],
        "validation_acc": [],
        "test_acc": [],
        "prompt": [system_prompt.get_value()],
        "optimizer_step_applied": [],
    }
    splits = (
        ("train", train_set, len(train_set), "train_acc"),
        ("validation", val_set, args.max_val_samples, "validation_acc"),
        ("test", test_set, args.max_test_samples, "test_acc"),
    )
    for split_name, dataset, max_samples, result_key in splits:
        if max_samples is None:
            max_samples = len(dataset)
        max_samples = min(max_samples, len(dataset))
        print(f"Prewarming initial prompt: split={split_name} samples={max_samples}")
        results[result_key] = eval_dataset(
            dataset,
            eval_fn,
            model,
            max_samples=max_samples,
            # Populate each key deterministically. Synthetic datasets can contain
            # duplicate questions; concurrent misses could otherwise issue two
            # model calls and record a response different from the one persisted.
            max_workers=1,
            pass_at_k=args.pass_at_k,
            eval_temperature=args.eval_temperature,
            add_sample_tag=args.add_sample_tag,
            use_eval_cache=True,
        )
    return results


def has_actionable_textual_gradient(variable):
    """Return whether a variable has at least one non-empty textual gradient."""
    for gradient in variable.gradients:
        value = getattr(gradient, "value", None)
        if isinstance(value, bytes):
            if value.strip():
                return True
        elif value is not None and str(value).strip():
            return True
    return False


def apply_optimizer_step_if_gradient(variable, optimizer, step):
    """Apply one optimizer step only when the variable has textual feedback."""
    applied = has_actionable_textual_gradient(variable)
    if applied:
        optimizer.step()
        print(f"optimizer_step: step={step} applied=True")
    else:
        print(
            f"optimizer_step: step={step} applied=False "
            "reason=no_actionable_prompt_gradient"
        )
    return applied


def train_without_group_inference(
    args,
    train_set,
    val_set,
    test_set,
    eval_fn,
    system_prompt,
    model,
    optimizer,
    results,
):
    """Advantage-guided editing without group inference.

    Every batch question is used with one response. With diagnostic
    preconditioning, one backward pass on the current prompt supplies the
    diagnoser's feedback before candidate proposal. Candidate rewards always
    come from the deterministic task checker.
    """
    train_loader = tg.tasks.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    global_step = 0
    for epoch in range(args.epochs):
        epoch_step = 0
        pbar = tqdm(train_loader, position=0)
        for batch_x, batch_y in pbar:
            if global_step >= args.max_steps:
                break
            if (
                args.max_steps_per_epoch is not None
                and epoch_step >= args.max_steps_per_epoch
            ):
                break

            pbar.set_description(f"Training step {global_step}. Epoch {epoch}")
            optimizer.zero_grad()
            old_prompt = system_prompt.get_value()
            batch_items = list(zip(batch_x, batch_y))

            diagnosis_records = diagnose_current_prompt(
                model,
                batch_items,
                eval_fn,
                args,
            )
            if diagnosis_records and hasattr(eval_fn, "state_dict"):
                diagnostic_state = eval_fn.state_dict()
                results.setdefault("diagnostic_state", []).append(diagnostic_state)
                selected = [
                    direction
                    for sample in diagnostic_state.get("last_step_info", [])
                    for direction in sample.get("selected_directions", [])
                ]
                print(
                    "preconditioned_diagnostic: "
                    f"step={global_step} selected_directions={selected or ['none']}"
                )

            if args.textual_learning_rate:
                results["textual_learning_rate"].append(
                    optimizer.scheduler.get_lr()
                )
            scored_candidates = score_candidates_single_response(
                optimizer,
                system_prompt,
                model,
                batch_items,
                eval_fn,
                args,
            )
            gate = improvement_gate(scored_candidates)
            results["improvement_gate"].append(gate)
            print(f"[improvement gate] step={global_step} {gate}")
            if gate["should_update"]:
                optimizer.step(scored_candidates)
            optimizer_step_applied = system_prompt.get_value() != old_prompt
            gate_reason = "" if gate["should_update"] else " reason=improvement_gate"
            print(
                f"optimizer_step: step={global_step} "
                f"applied={optimizer_step_applied}{gate_reason}"
            )
            results["optimizer_step_applied"].append(optimizer_step_applied)
            results["candidates"].append(
                [candidate["value"] for candidate in scored_candidates]
            )
            results["candidate_rewards"].append(
                [candidate["reward"] for candidate in scored_candidates]
            )
            results["baseline_rewards"].append(
                optimizer.last_baselines.get(system_prompt)
            )
            results["advantages"].append(
                optimizer.last_advantages.get(system_prompt, [])
            )

            if not optimizer_step_applied and args.use_validation_revert:
                print(
                    f"evaluation_step: step={global_step} skipped=True "
                    "reason=prompt_unchanged"
                )
                results["validation_acc"].append(results["validation_acc"][-1])
                test_acc = results["test_acc"][-1]
            elif not optimizer_step_applied and args.test_every == "step":
                print(
                    f"evaluation_step: step={global_step} skipped=True "
                    "reason=prompt_unchanged"
                )
                test_acc = results["test_acc"][-1]
                results["validation_acc"].append(results["validation_acc"][-1])
            elif args.use_validation_revert:
                rejected = run_validation_revert(
                    system_prompt,
                    results,
                    model,
                    eval_fn,
                    val_set,
                    max_val_samples=args.max_val_samples,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                    max_workers=args.eval_max_workers,
                    tolerance=args.validation_revert_tolerance,
                )
                test_acc = (
                    results["test_acc"][-1]
                    if rejected
                    else eval_dataset(
                        test_set,
                        eval_fn,
                        model,
                        max_samples=args.max_test_samples,
                        max_workers=args.eval_max_workers,
                        pass_at_k=args.pass_at_k,
                        eval_temperature=args.eval_temperature,
                        add_sample_tag=args.add_sample_tag,
                        use_eval_cache=args.use_eval_cache,
                    )
                )
            elif args.test_every == "step":
                test_acc = eval_dataset(
                    test_set,
                    eval_fn,
                    model,
                    max_samples=args.max_test_samples,
                    max_workers=args.eval_max_workers,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                )
                if args.skip_validation:
                    results["validation_acc"].append([])
                else:
                    results["validation_acc"].append(
                        results["validation_acc"][-1]
                    )

            if args.use_validation_revert or args.test_every == "step":
                results["test_acc"].append(test_acc)
                results["prompt"].append(system_prompt.get_value())

            global_step += 1
            epoch_step += 1

        if (
            args.test_every == "epoch"
            and not args.use_validation_revert
            and epoch_step > 0
        ):
            test_acc = eval_dataset(
                test_set,
                eval_fn,
                model,
                max_samples=args.max_test_samples,
                max_workers=args.eval_max_workers,
                pass_at_k=args.pass_at_k,
                eval_temperature=args.eval_temperature,
                add_sample_tag=args.add_sample_tag,
                use_eval_cache=args.use_eval_cache,
            )
            if args.skip_validation:
                results["validation_acc"].append([])
            else:
                results["validation_acc"].append(results["validation_acc"][-1])
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())

        if global_step >= args.max_steps:
            break
    return results


def train_textsteer(
    args,
    train_set,
    val_set,
    test_set,
    eval_fn,
    system_prompt,
    model,
    optimizer,
    results,
):
    """TextSteer training loop (Section 4).

    Group inference keeps the questions with mixed outcomes; their responses
    supply the (diagnostically preconditioned) textual feedback, and the same
    questions score the candidates of advantage-guided editing. With the
    textual learning rate the editor applies at most L edits.
    """
    train_loader = tg.tasks.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    global_step = 0
    for epoch in range(args.epochs):
        epoch_step = 0
        pbar = tqdm(train_loader, position=0)
        for batch_x, batch_y in pbar:
            if global_step >= args.max_steps:
                break
            if (
                args.max_steps_per_epoch is not None
                and epoch_step >= args.max_steps_per_epoch
            ):
                break

            pbar.set_description(f"Training step {global_step}. Epoch {epoch}")
            optimizer.zero_grad()

            optimization_losses = []
            optimized_group_indices = []
            batch_group_correct = []
            batch_group_means = []
            batch_items = list(zip(batch_x, batch_y))
            for sample_idx, (x_raw, y_raw) in enumerate(batch_items):
                group_correct, group_losses, group_responses = group_inference(
                    x_raw=x_raw,
                    y_raw=y_raw,
                    eval_fn=eval_fn,
                    model=model,
                    group_temperature=args.group_temperature,
                    add_sample_tag=args.add_sample_tag,
                )
                use_group_for_optimization = any(group_correct) and not all(group_correct)

                batch_group_correct.append(group_correct)
                batch_group_means.append(float(np.mean(group_correct)))
                print(
                    f"[step {global_step}] sample {sample_idx} group correctness: "
                    f"{group_correct} -> {sum(group_correct)}/{len(group_correct)} correct"
                )

                if use_group_for_optimization:
                    optimization_losses.extend(group_losses)
                    optimized_group_indices.append(sample_idx)
                elif all(group_correct):
                    print(
                        f"[step {global_step}] sample {sample_idx} is all-correct; "
                        "excluding it from prompt optimization"
                    )
                else:
                    print(
                        f"[step {global_step}] sample {sample_idx} is all-wrong; "
                        "excluding it from prompt optimization"
                    )

            total_correct = sum(sum(group) for group in batch_group_correct)
            total_generations = sum(len(group) for group in batch_group_correct)
            all_correct_samples = sum(all(group) for group in batch_group_correct)
            all_wrong_samples = sum(not any(group) for group in batch_group_correct)
            mixed_samples = len(batch_group_correct) - all_correct_samples - all_wrong_samples
            print(
                f"[step {global_step}] group correctness matrix: {batch_group_correct} -> "
                f"{total_correct}/{total_generations} generations correct"
            )
            print(
                f"[step {global_step}] group summary: "
                f"all_correct={all_correct_samples}, mixed={mixed_samples}, "
                f"all_wrong={all_wrong_samples}"
            )
            print(
                f"[step {global_step}] groups used for prompt optimization: "
                f"{optimized_group_indices}"
            )
            results["train_group_acc"].append(batch_group_correct)
            results["train_group_sample_counts"].append(
                [len(group) for group in batch_group_correct]
            )
            results["train_acc"].append(batch_group_means)
            results["optimized_group_indices"].append(optimized_group_indices)

            old_prompt = system_prompt.get_value()
            if hasattr(eval_fn, "begin_step"):
                eval_fn.begin_step()
            try:
                if optimization_losses:
                    tg.sum(optimization_losses).backward()
                else:
                    print(
                        f"[step {global_step}] no mixed groups -- "
                        "skipping backward/optimizer step"
                    )
            finally:
                if hasattr(eval_fn, "end_step"):
                    eval_fn.end_step()

            scored_candidates = []
            if args.textual_learning_rate:
                results["textual_learning_rate"].append(
                    optimizer.scheduler.get_lr()
                )
            selected_batch_items = [
                batch_items[index] for index in optimized_group_indices
            ]
            gate = None
            if has_actionable_textual_gradient(system_prompt) and selected_batch_items:
                scored_candidates = score_candidates(
                    optimizer,
                    system_prompt,
                    model,
                    selected_batch_items,
                    eval_fn,
                    args,
                )
                gate = improvement_gate(scored_candidates)
                print(f"[improvement gate] step={global_step} {gate}")
                if gate["should_update"]:
                    optimizer.step(scored_candidates)
                optimizer_step_applied = system_prompt.get_value() != old_prompt
                gate_reason = "" if gate["should_update"] else " reason=improvement_gate"
                print(
                    f"optimizer_step: step={global_step} "
                    f"applied={optimizer_step_applied}{gate_reason}"
                )
            else:
                optimizer_step_applied = False
                reason = (
                    "no_mixed_groups"
                    if not selected_batch_items
                    else "no_actionable_prompt_gradient"
                )
                print(
                    f"optimizer_step: step={global_step} applied=False "
                    f"reason={reason}"
                )
            results["improvement_gate"].append(gate)
            results["baseline_rewards"].append(
                optimizer.last_baselines.get(system_prompt)
                if scored_candidates
                else None
            )
            results["advantages"].append(
                optimizer.last_advantages.get(system_prompt, [])
                if scored_candidates
                else []
            )
            results["optimizer_step_applied"].append(optimizer_step_applied)
            results["candidates"].append(
                [candidate["value"] for candidate in scored_candidates]
            )
            results["candidate_rewards"].append(
                [candidate["reward"] for candidate in scored_candidates]
            )
            if hasattr(eval_fn, "state_dict"):
                diagnostic_state = eval_fn.state_dict()
                results.setdefault("diagnostic_state", []).append(diagnostic_state)
                selected = [
                    direction
                    for sample in diagnostic_state.get("last_step_info", [])
                    for direction in sample.get("selected_directions", [])
                ]
                print(
                    "preconditioned_diagnostic: "
                    f"step={global_step} selected_directions={selected or ['none']}"
                )

            if not optimizer_step_applied and args.use_validation_revert:
                print(
                    f"evaluation_step: step={global_step} skipped=True "
                    "reason=prompt_unchanged"
                )
                results["validation_acc"].append(results["validation_acc"][-1])
                test_acc = results["test_acc"][-1]
            elif not optimizer_step_applied and args.test_every == "step":
                print(
                    f"evaluation_step: step={global_step} skipped=True "
                    "reason=prompt_unchanged"
                )
                test_acc = results["test_acc"][-1]
                results["validation_acc"].append(results["validation_acc"][-1])
            elif args.use_validation_revert:
                rejected = run_validation_revert(
                    system_prompt,
                    results,
                    model,
                    eval_fn,
                    val_set,
                    max_val_samples=args.max_val_samples,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                    max_workers=args.eval_max_workers,
                    tolerance=args.validation_revert_tolerance,
                )
                test_acc = (
                    results["test_acc"][-1]
                    if rejected
                    else eval_dataset(
                        test_set,
                        eval_fn,
                        model,
                        max_samples=args.max_test_samples,
                        max_workers=args.eval_max_workers,
                        pass_at_k=args.pass_at_k,
                        eval_temperature=args.eval_temperature,
                        add_sample_tag=args.add_sample_tag,
                        use_eval_cache=args.use_eval_cache,
                    )
                )
            elif args.test_every == "step":
                test_acc = eval_dataset(
                    test_set,
                    eval_fn,
                    model,
                    max_samples=args.max_test_samples,
                    max_workers=args.eval_max_workers,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                )
                if args.skip_validation:
                    results["validation_acc"].append([])
                else:
                    results["validation_acc"].append(
                        results["validation_acc"][-1]
                    )

            if args.use_validation_revert or args.test_every == "step":
                results["test_acc"].append(test_acc)
                results["prompt"].append(system_prompt.get_value())

            global_step += 1
            epoch_step += 1

        if (
            args.test_every == "epoch"
            and not args.use_validation_revert
            and epoch_step > 0
        ):
            test_acc = eval_dataset(
                test_set,
                eval_fn,
                model,
                max_samples=args.max_test_samples,
                max_workers=args.eval_max_workers,
                pass_at_k=args.pass_at_k,
                eval_temperature=args.eval_temperature,
                add_sample_tag=args.add_sample_tag,
                use_eval_cache=args.use_eval_cache,
            )
            if args.skip_validation:
                results["validation_acc"].append([])
            else:
                results["validation_acc"].append(results["validation_acc"][-1])
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())

        if global_step >= args.max_steps:
            break
    return results


def train_textual_gradient(
    args,
    train_set,
    val_set,
    test_set,
    eval_fn,
    system_prompt,
    model,
    optimizer,
    results,
):
    """TextGrad training loop: one response per question, TextGrad update.

    Used without group inference and without advantage-guided editing. With
    the textual learning rate the update applies at most L edits.
    """
    train_loader = tg.tasks.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    global_step = 0
    for epoch in range(args.epochs):
        epoch_step = 0
        pbar = tqdm(train_loader, position=0)
        for batch_x, batch_y in pbar:
            if global_step >= args.max_steps:
                break
            if (
                args.max_steps_per_epoch is not None
                and epoch_step >= args.max_steps_per_epoch
            ):
                break

            pbar.set_description(f"Training step {global_step}. Epoch {epoch}")
            optimizer.zero_grad()

            losses = []
            for x_raw, y_raw in zip(batch_x, batch_y):
                x = tg.Variable(
                    x_raw,
                    requires_grad=False,
                    role_description="query to the language model",
                )
                y = tg.Variable(
                    y_raw,
                    requires_grad=False,
                    role_description="correct answer for the query",
                )
                response = model(x)
                eval_output_variable = call_eval_fn(x, y, response, eval_fn)
                losses.append(eval_output_variable)

            total_loss = tg.sum(losses)
            if hasattr(eval_fn, "begin_step"):
                eval_fn.begin_step()
            try:
                total_loss.backward()
            finally:
                if hasattr(eval_fn, "end_step"):
                    eval_fn.end_step()
            if args.textual_learning_rate:
                results["textual_learning_rate"].append(
                    optimizer.scheduler.get_lr()
                )
            optimizer_step_applied = apply_optimizer_step_if_gradient(
                system_prompt,
                optimizer,
                global_step,
            )
            results["optimizer_step_applied"].append(optimizer_step_applied)
            if hasattr(eval_fn, "state_dict"):
                diagnostic_state = eval_fn.state_dict()
                results.setdefault("diagnostic_state", []).append(diagnostic_state)
                selected = [
                    direction
                    for sample in diagnostic_state.get("last_step_info", [])
                    for direction in sample.get("selected_directions", [])
                ]
                print(
                    "preconditioned_diagnostic: "
                    f"step={global_step} selected_directions={selected or ['none']}"
                )

            if not optimizer_step_applied and args.use_validation_revert:
                print(
                    f"evaluation_step: step={global_step} skipped=True "
                    "reason=prompt_unchanged"
                )
                results["validation_acc"].append(results["validation_acc"][-1])
                test_acc = results["test_acc"][-1]
            elif not optimizer_step_applied and args.test_every == "step":
                print(
                    f"evaluation_step: step={global_step} skipped=True "
                    "reason=prompt_unchanged"
                )
                test_acc = results["test_acc"][-1]
                results["validation_acc"].append(results["validation_acc"][-1])
            elif args.use_validation_revert:
                rejected = run_validation_revert(
                    system_prompt,
                    results,
                    model,
                    eval_fn,
                    val_set,
                    max_val_samples=args.max_val_samples,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                    max_workers=args.eval_max_workers,
                    tolerance=args.validation_revert_tolerance,
                )
                test_acc = (
                    results["test_acc"][-1]
                    if rejected
                    else eval_dataset(
                        test_set,
                        eval_fn,
                        model,
                        max_samples=args.max_test_samples,
                        max_workers=args.eval_max_workers,
                        pass_at_k=args.pass_at_k,
                        eval_temperature=args.eval_temperature,
                        add_sample_tag=args.add_sample_tag,
                        use_eval_cache=args.use_eval_cache,
                    )
                )
            elif args.test_every == "step":
                test_acc = eval_dataset(
                    test_set,
                    eval_fn,
                    model,
                    max_samples=args.max_test_samples,
                    max_workers=args.eval_max_workers,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                )
                if args.skip_validation:
                    results["validation_acc"].append([])
                else:
                    results["validation_acc"].append(
                        results["validation_acc"][-1]
                    )

            if args.use_validation_revert or args.test_every == "step":
                results["test_acc"].append(test_acc)
                results["prompt"].append(system_prompt.get_value())

            global_step += 1
            epoch_step += 1

        if (
            args.test_every == "epoch"
            and not args.use_validation_revert
            and epoch_step > 0
        ):
            test_acc = eval_dataset(
                test_set,
                eval_fn,
                model,
                max_samples=args.max_test_samples,
                max_workers=args.eval_max_workers,
                pass_at_k=args.pass_at_k,
                eval_temperature=args.eval_temperature,
                add_sample_tag=args.add_sample_tag,
                use_eval_cache=args.use_eval_cache,
            )
            if args.skip_validation:
                results["validation_acc"].append([])
            else:
                results["validation_acc"].append(results["validation_acc"][-1])
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())

        if global_step >= args.max_steps:
            break
    return results


def train_without_advantage_guided_editing(
    args,
    train_set,
    val_set,
    test_set,
    eval_fn,
    system_prompt,
    model,
    optimizer,
    results,
):
    """Group inference without advantage-guided editing.

    The feedback from the mixed-outcome questions is applied with the TextGrad
    update (at most L edits with the textual learning rate).
    """
    train_loader = tg.tasks.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    global_step = 0
    for epoch in range(args.epochs):
        epoch_step = 0
        pbar = tqdm(train_loader, position=0)
        for batch_x, batch_y in pbar:
            if global_step >= args.max_steps:
                break
            if args.max_steps_per_epoch is not None and epoch_step >= args.max_steps_per_epoch:
                break

            pbar.set_description(f"Training step {global_step}. Epoch {epoch}")
            optimizer.zero_grad()

            optimization_losses = []
            optimized_group_indices = []
            batch_group_correct = []
            batch_group_means = []
            for sample_idx, (x_raw, y_raw) in enumerate(zip(batch_x, batch_y)):
                group_correct, group_losses, group_responses = group_inference(
                    x_raw=x_raw,
                    y_raw=y_raw,
                    eval_fn=eval_fn,
                    model=model,
                    group_temperature=args.group_temperature,
                    add_sample_tag=args.add_sample_tag,
                )
                use_group_for_optimization = any(group_correct) and not all(group_correct)

                batch_group_correct.append(group_correct)
                batch_group_means.append(float(np.mean(group_correct)))
                print(
                    f"[step {global_step}] sample {sample_idx} group correctness: "
                    f"{group_correct} -> {sum(group_correct)}/{len(group_correct)} correct"
                )

                if use_group_for_optimization:
                    optimization_losses.extend(group_losses)
                    optimized_group_indices.append(sample_idx)
                elif all(group_correct):
                    print(
                        f"[step {global_step}] sample {sample_idx} is all-correct; "
                        "excluding it from prompt optimization"
                    )
                else:
                    print(
                        f"[step {global_step}] sample {sample_idx} is all-wrong; "
                        "excluding it from prompt optimization"
                    )

            total_correct = sum(sum(group) for group in batch_group_correct)
            total_generations = sum(len(group) for group in batch_group_correct)
            all_correct_samples = sum(all(group) for group in batch_group_correct)
            all_wrong_samples = sum(not any(group) for group in batch_group_correct)
            mixed_samples = len(batch_group_correct) - all_correct_samples - all_wrong_samples
            print(
                f"[step {global_step}] group correctness matrix: {batch_group_correct} -> "
                f"{total_correct}/{total_generations} generations correct"
            )
            print(
                f"[step {global_step}] group summary: "
                f"all_correct={all_correct_samples}, mixed={mixed_samples}, "
                f"all_wrong={all_wrong_samples}"
            )
            print(
                f"[step {global_step}] groups used for prompt optimization: "
                f"{optimized_group_indices}"
            )
            results["train_group_acc"].append(batch_group_correct)
            results["train_group_sample_counts"].append(
                [len(group) for group in batch_group_correct]
            )
            results["train_acc"].append(batch_group_means)
            results["optimized_group_indices"].append(optimized_group_indices)

            if hasattr(eval_fn, "begin_step"):
                eval_fn.begin_step()
            try:
                if optimization_losses:
                    total_loss = tg.sum(optimization_losses)
                    total_loss.backward()
                else:
                    print(
                        f"[step {global_step}] no mixed groups -- "
                        "skipping backward/optimizer step"
                    )
            finally:
                if hasattr(eval_fn, "end_step"):
                    eval_fn.end_step()
            if args.textual_learning_rate:
                results["textual_learning_rate"].append(
                    optimizer.scheduler.get_lr()
                )
            optimizer_step_applied = apply_optimizer_step_if_gradient(
                system_prompt,
                optimizer,
                global_step,
            )
            results["optimizer_step_applied"].append(optimizer_step_applied)
            if hasattr(eval_fn, "state_dict"):
                diagnostic_state = eval_fn.state_dict()
                results.setdefault("diagnostic_state", []).append(diagnostic_state)
                selected = [
                    direction
                    for sample in diagnostic_state.get("last_step_info", [])
                    for direction in sample.get("selected_directions", [])
                ]
                print(
                    "preconditioned_diagnostic: "
                    f"step={global_step} selected_directions={selected or ['none']}"
                )

            if not optimizer_step_applied and args.use_validation_revert:
                print(
                    f"evaluation_step: step={global_step} skipped=True "
                    "reason=prompt_unchanged"
                )
                results["validation_acc"].append(results["validation_acc"][-1])
                test_acc = results["test_acc"][-1]
            elif not optimizer_step_applied and args.test_every == "step":
                print(
                    f"evaluation_step: step={global_step} skipped=True "
                    "reason=prompt_unchanged"
                )
                test_acc = results["test_acc"][-1]
                results["validation_acc"].append(results["validation_acc"][-1])
            elif args.use_validation_revert:
                rejected = run_validation_revert(
                    system_prompt,
                    results,
                    model,
                    eval_fn,
                    val_set,
                    max_val_samples=args.max_val_samples,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                    max_workers=args.eval_max_workers,
                    tolerance=args.validation_revert_tolerance,
                )
                test_acc = results["test_acc"][-1] if rejected else eval_dataset(
                    test_set,
                    eval_fn,
                    model,
                    max_samples=args.max_test_samples,
                    max_workers=args.eval_max_workers,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                )
            elif args.test_every == "step":
                test_acc = eval_dataset(
                    test_set,
                    eval_fn,
                    model,
                    max_samples=args.max_test_samples,
                    max_workers=args.eval_max_workers,
                    pass_at_k=args.pass_at_k,
                    eval_temperature=args.eval_temperature,
                    add_sample_tag=args.add_sample_tag,
                    use_eval_cache=args.use_eval_cache,
                )
                if args.skip_validation:
                    results["validation_acc"].append([])
                else:
                    results["validation_acc"].append(results["validation_acc"][-1])

            if args.use_validation_revert or args.test_every == "step":
                results["test_acc"].append(test_acc)
                results["prompt"].append(system_prompt.get_value())

            global_step += 1
            epoch_step += 1

        if args.test_every == "epoch" and not args.use_validation_revert and epoch_step > 0:
            test_acc = eval_dataset(
                test_set,
                eval_fn,
                model,
                max_samples=args.max_test_samples,
                max_workers=args.eval_max_workers,
                pass_at_k=args.pass_at_k,
                eval_temperature=args.eval_temperature,
                add_sample_tag=args.add_sample_tag,
                use_eval_cache=args.use_eval_cache,
            )
            if args.skip_validation:
                results["validation_acc"].append([])
            else:
                results["validation_acc"].append(results["validation_acc"][-1])
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())

        if global_step >= args.max_steps:
            break

    return results


def train(args):
    train_set, val_set, test_set, eval_fn, system_prompt, model, optimizer = build_models_and_data(args)
    results = initialize_results(args, val_set, test_set, eval_fn, model, system_prompt)
    if args.group_inference and args.advantage_guided_editing:
        loop = train_textsteer
    elif args.advantage_guided_editing:
        loop = train_without_group_inference
    elif args.group_inference:
        loop = train_without_advantage_guided_editing
    else:
        loop = train_textual_gradient
    return loop(
        args,
        train_set,
        val_set,
        test_set,
        eval_fn,
        system_prompt,
        model,
        optimizer,
        results,
    )


def save_results(results, path):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: [value.tolist() if hasattr(value, "tolist") else value for value in values]
        for key, values in results.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "TextSteer: prompt optimization with group inference, diagnostic "
            "preconditioning, advantage-guided editing and a textual learning rate."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-engine", default=DEFAULT_EVAL_ENGINE,
                        help="Backward (critic) and prompt-editor model.")
    parser.add_argument("--test-engine", default=DEFAULT_TEST_ENGINE,
                        help="Target model whose system prompt is optimized.")
    parser.add_argument(
        "--test-max-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens for the target model. Defaults to 2000 for FinQA and 4096 otherwise.",
    )
    parser.add_argument(
        "--eval-max-workers",
        type=int,
        default=DEFAULT_EVAL_MAX_WORKERS,
        help="Maximum concurrent validation/test model requests.",
    )
    parser.add_argument(
        "--openrouter-reasoning",
        choices=("provider-default", "off", "on", "low", "medium", "high"),
        default="provider-default",
        help=(
            "OpenRouter reasoning mode: low/medium/high send an explicit effort; "
            "provider-default omits the reasoning object. Supported modes depend "
            "on the model/provider; GPT-OSS on Groq does not support off."
        ),
    )
    parser.add_argument(
        "--openrouter-request-timeout", type=float, default=300.0,
        help="Per-attempt OpenRouter API timeout in seconds (must be positive).",
    )
    parser.add_argument(
        "--openrouter-max-retries", type=int, default=5,
        help="OpenAI SDK retries for OpenRouter transport/429/5xx errors (>= 0).",
    )
    parser.add_argument(
        "--openrouter-transient-error-policy",
        choices=("raise", "score-zero"), default=None,
        help=(
            "After OpenRouter SDK retries are exhausted, raise or return an "
            "uncached empty answer. Defaults to score-zero for FinQA, raise otherwise."
        ),
    )
    parser.add_argument(
        "--openrouter-provider-sort",
        choices=("provider-default", "throughput", "latency", "price"),
        default="provider-default",
        help="OpenRouter upstream routing priority.",
    )
    parser.add_argument(
        "--openrouter-provider", default="auto",
        help=(
            "Pin one lowercase OpenRouter provider slug (e.g. groq, coreweave/fp8) "
            "with cross-provider fallback disabled; auto keeps automatic routing."
        ),
    )
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="FinQA or RG_<dataset> (RG_ab, RG_count_primes, RG_graph_color).")
    parser.add_argument("--train-size", type=int, default=None, help="Training split size.")
    parser.add_argument("--val-size", type=int, default=None, help="Validation split size.")
    parser.add_argument("--test-size", type=int, default=None, help="Test split size.")
    parser.add_argument(
        "--rg-config-json",
        default="{}",
        help="Reasoning Gym generator keyword arguments, keyed by dataset name.",
    )
    parser.add_argument("--starting-prompt", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Optional cap on optimizer steps inside each epoch. --max-steps remains a global cap.",
    )
    parser.add_argument("--pass-at-k", type=int, default=DEFAULT_PASS_AT_K)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument(
        "--test-every",
        default="step",
        choices=["step", "epoch"],
        help="Evaluate the test set after every optimizer step or once after each epoch.",
    )
    parser.add_argument("--no-sample-tag", dest="add_sample_tag", action="store_false")
    parser.add_argument("--refresh-eval-cache", dest="use_eval_cache", action="store_false")
    parser.add_argument(
        "--prewarm-initial-cache-only",
        action="store_true",
        help=(
            "Evaluate the unchanged initial prompt on train, validation, and test to populate "
            "the test-engine cache, save optional results, and exit before optimization."
        ),
    )
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--optimizer-verbose", type=int, default=1)
    components = parser.add_argument_group(
        "TextSteer components (all on by default; each ablation switches one off)"
    )
    components.add_argument(
        "--no-group-inference",
        dest="group_inference",
        action="store_false",
        help="Use one response for every batch question instead of selecting mixed-outcome questions.",
    )
    components.add_argument(
        "--no-diagnostic-preconditioning",
        dest="diagnostic_preconditioning",
        action="store_false",
        help="Use generic TextGrad feedback instead of the task diagnoser.",
    )
    components.add_argument(
        "--no-advantage-guided-editing",
        dest="advantage_guided_editing",
        action="store_false",
        help="Apply the textual feedback with the TextGrad update instead of scoring candidates.",
    )
    components.add_argument(
        "--no-textual-learning-rate",
        dest="textual_learning_rate",
        action="store_false",
        help="Do not limit the number of edits per update.",
    )
    hyperparameters = parser.add_argument_group("TextSteer hyperparameters")
    hyperparameters.add_argument(
        "--group-temperature",
        type=float,
        default=DEFAULT_GROUP_TEMPERATURE,
        help="Sampling temperature of the training responses.",
    )
    hyperparameters.add_argument(
        "--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES,
        help="K: candidate rewrites per update.",
    )
    hyperparameters.add_argument(
        "--candidate-temperature", type=float, default=DEFAULT_CANDIDATE_TEMPERATURE,
        help="Sampling temperature of the candidate rewrites.",
    )
    hyperparameters.add_argument(
        "--candidate-responses", type=int, default=DEFAULT_CANDIDATE_RESPONSES,
        help="J: responses per selected question when scoring a candidate.",
    )
    hyperparameters.add_argument(
        "--edit-budget", type=int, default=DEFAULT_EDIT_BUDGET,
        help="L: textual learning rate, the maximum number of distinct edits per update.",
    )
    parser.add_argument("--use-validation-revert", action="store_true")
    parser.add_argument(
        "--validation-revert-tolerance",
        type=float,
        default=DEFAULT_VALIDATION_REVERT_TOLERANCE,
        help=(
            "Maximum absolute validation-accuracy drop allowed from the best score so far. "
            "For example, 0.03 allows a three-percentage-point drop."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation-set evaluation entirely. Incompatible with --use-validation-revert.",
    )
    parser.add_argument("--save-results", default=None)
    parser.add_argument("--rg-train-size", type=int, default=50)
    parser.add_argument("--rg-val-size", type=int, default=10)
    parser.add_argument("--rg-test-size", type=int, default=100)
    parser.add_argument(
        "--rg-difficulty",
        default="custom",
        choices=["custom", "easy", "medium"],
        help="Named difficulty preset for RG_ab, RG_count_primes and RG_graph_color.",
    )
    parser.set_defaults(
        add_sample_tag=True,
        use_eval_cache=True,
    )
    args = parser.parse_args(argv)
    args.rg_config = parse_json_object_argument(
        args.rg_config_json,
        parser,
        "--rg-config-json",
    )
    if (
        args.task.startswith("RG_")
        and args.diagnostic_preconditioning
        and args.task not in RG_DIAGNOSTIC_MODES
    ):
        parser.error(f"{args.task} has no diagnoser; pass --no-diagnostic-preconditioning")
    if args.test_max_tokens is None:
        args.test_max_tokens = (
            DEFAULT_FINQA_TEST_MAX_TOKENS
            if args.task == "FinQA"
            else DEFAULT_TEST_MAX_TOKENS
        )
    if args.openrouter_transient_error_policy is None:
        args.openrouter_transient_error_policy = (
            "score-zero" if args.task == "FinQA" else "raise"
        )
    if args.task.startswith("RG_"):
        if args.train_size is not None:
            args.rg_train_size = args.train_size
        if args.val_size is not None:
            args.rg_val_size = args.val_size
        if args.test_size is not None:
            args.rg_test_size = args.test_size
    for name in ("train_size", "val_size", "test_size"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.test_max_tokens < 1:
        parser.error("--test-max-tokens must be >= 1")
    if args.eval_max_workers < 1:
        parser.error("--eval-max-workers must be >= 1")
    if (
        not np.isfinite(args.openrouter_request_timeout)
        or args.openrouter_request_timeout <= 0
    ):
        parser.error("--openrouter-request-timeout must be > 0")
    if args.openrouter_max_retries < 0:
        parser.error("--openrouter-max-retries must be >= 0")
    if not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)*", args.openrouter_provider
    ):
        parser.error("--openrouter-provider must be auto or a lowercase provider slug")
    if args.group_temperature < 0:
        parser.error("--group-temperature must be >= 0")
    if args.num_candidates < 1:
        parser.error("--num-candidates must be >= 1")
    if args.candidate_temperature < 0:
        parser.error("--candidate-temperature must be >= 0")
    if args.candidate_responses < 1:
        parser.error("--candidate-responses must be >= 1")
    if args.edit_budget < 1:
        parser.error("--edit-budget must be >= 1")
    if args.pass_at_k < 1:
        parser.error("--pass-at-k must be >= 1")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch < 1:
        parser.error("--max-steps-per-epoch must be >= 1")
    if not 0.0 <= args.validation_revert_tolerance <= 1.0:
        parser.error("--validation-revert-tolerance must be between 0 and 1")
    if args.skip_validation and args.use_validation_revert:
        parser.error("--skip-validation is incompatible with --use-validation-revert")
    if args.eval_temperature is None:
        args.eval_temperature = DEFAULT_PASS_AT_K_TEMPERATURE if args.pass_at_k > 1 else 0.0
    return args


def main():
    args = parse_args()
    if args.prewarm_initial_cache_only:
        results = prewarm_initial_prompt_cache(args)
        save_results(results, args.save_results)
        print("Initial-prompt cache prewarm complete.")
        return
    results = train(args)
    save_results(results, args.save_results)
    print("Final prompt:")
    print(results["prompt"][-1])


if __name__ == "__main__":
    main()
