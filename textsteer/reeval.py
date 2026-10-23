"""Re-evaluate a saved prompt on the test split under different decoding settings.

Pass the run's own ``run.py`` arguments followed by the overrides; later flags
win. The paper uses this for Count Primes (``--test-max-tokens 8192``) and A::B
(``--openrouter-reasoning medium --test-max-tokens 4096``).

Usage:
    python textsteer/reeval.py --results RUN/results.json --output OUT.json \
        <run.py arguments> <overrides>
    python textsteer/reeval.py --prompt initial --output OUT.json \
        <run.py arguments> <overrides>
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textsteer import run  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=None, help="results.json written by a training run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", choices=["final", "initial"], default="final")
    own, runner_argv = parser.parse_known_args(argv)
    args = run.parse_args(runner_argv)
    run.load_textgrad()
    tg = run.tg

    if own.results is not None:
        saved = json.loads(Path(own.results).read_text(encoding="utf-8"))
        prompt = saved["prompt"][-1 if own.prompt == "final" else 0]
    elif own.prompt == "initial":
        prompt = args.starting_prompt or run.get_default_starting_prompt(args.task)
    else:
        parser.error("--results is required for --prompt final")
    _, _, test_set, eval_fn = run.load_task(args.task, evaluation_api=None, **run.build_task_kwargs(args))
    if args.task == "FinQA":
        eval_fn = run.build_finqa_eval_fn(args)
    test_set = run.apply_split_size(test_set, getattr(args, "test_size", None))
    limit = len(test_set) if args.max_test_samples is None else min(args.max_test_samples, len(test_set))
    items = [test_set[i] for i in range(limit)]
    engine = run.build_test_engine(args)

    def one(item):
        question, answer = item
        text = engine(question, system_prompt=prompt, temperature=args.eval_temperature)
        x = tg.Variable(question, requires_grad=False, role_description="query to the language model")
        y = tg.Variable(answer, requires_grad=False, role_description="correct answer for the query")
        response = tg.Variable(text, requires_grad=False, role_description="response from the language model")
        return int(run.score_response(x, y, response, eval_fn)), not str(text).strip()

    with ThreadPoolExecutor(max_workers=args.eval_max_workers) as pool:
        rows = list(pool.map(one, items))
    scores = [score for score, _ in rows]
    report = {
        "results": own.results,
        "prompt": own.prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "test_engine": args.test_engine,
        "openrouter_reasoning": args.openrouter_reasoning,
        "max_tokens": args.test_max_tokens,
        "n": len(items),
        "accuracy": statistics.mean(scores) if scores else 0.0,
        "empty": sum(empty for _, empty in rows),
        "scores_in_question_order": scores,
        "skipped_transient_requests": getattr(engine, "skipped_transient_requests", None),
    }
    output = Path(own.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{own.prompt} prompt: test accuracy {report['accuracy']:.3f} "
          f"({report['empty']} empty) at max_tokens={args.test_max_tokens}")


if __name__ == "__main__":
    main()
