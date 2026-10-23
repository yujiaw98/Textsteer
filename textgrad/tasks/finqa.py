import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import platformdirs
import textgrad as tg

from textgrad.autograd.function import BackwardContext, Function
from .base import Dataset


FINQA_RAW_BASE_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
FINQA_SPLIT_FILES = {
    "train": "train.json",
    "val": "dev.json",
    "test": "test.json",
}
FINQA_DEFAULT_SPLIT_SIZES = {
    "train": 50,
    "val": 100,
    "test": 100,
}
FINQA_CRITIC_SYSTEM_PROMPT = (
    "You are a careful FinQA error diagnoser. The deterministic answer checker "
    "has already decided whether the model's final answer is correct. Do not "
    "override that score. Your job is to explain incorrect answers and produce "
    "a reusable textual gradient for improving the system prompt."
)
FINQA_DIAGNOSIS_PROMPT_TEMPLATE = """We are optimizing a system prompt for FinQA.

The deterministic answer checker marked this example as incorrect.

Question and table/text evidence:
{question}

Gold answer:
{gold_answer}

Gold reasoning program:
{gold_program}

Gold nested program:
{gold_program_re}

Gold supporting facts:
{gold_inds}

Model response:
{prediction}

Model extracted final answer:
{predicted_answer}

Return a compact JSON object with this exact schema:
{{
  "is_correct": false,
  "error_type": "evidence_selection_error | operation_selection_error | calculation_error | scale_or_unit_error | answer_format_error | other",
  "evidence_error": true/false,
  "operation_error": true/false,
  "calculation_error": true/false,
  "format_error": true/false,
  "diagnosis": "...",
  "prompt_gradient": "..."
}}

The prompt_gradient must be general guidance for future FinQA questions. It
must not mention this example id, memorize literal table values, or reveal the
gold final answer as a rule.
"""

_NUMBER_WITH_UNIT_RE = re.compile(
    r"(?P<number>\(?-?\$?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?\)?)"
    r"(?P<percent>\s*%)?"
    r"(?P<unit>\s*(?:thousand|million|billion|trillion))?",
    flags=re.IGNORECASE,
)


def _clean_answer_text(answer: str) -> str:
    answer = str(answer).strip()
    answer = answer.strip().strip("`").strip()
    return answer.strip(" \t\r\n.。")


def extract_finqa_answer(text: str) -> str:
    """Extract a concise final answer from a FinQA model response."""
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    answer_patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)$",
        r"therefore,?\s*(.+)$",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return _clean_answer_text(matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _clean_answer_text(lines[-1]) if lines else ""


def _coerce_number(number_text: str) -> Optional[float]:
    number_text = str(number_text).strip()
    is_parenthesized = number_text.startswith("(") and number_text.endswith(")")
    cleaned = number_text.replace("$", "").replace(",", "")
    cleaned = cleaned.strip("()")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if is_parenthesized and value > 0:
        value = -value
    return value


def _scale_for_unit(unit_text: str) -> float:
    unit_text = (unit_text or "").strip().lower()
    if unit_text == "thousand":
        return 1_000.0
    if unit_text == "million":
        return 1_000_000.0
    if unit_text == "billion":
        return 1_000_000_000.0
    if unit_text == "trillion":
        return 1_000_000_000_000.0
    return 1.0


def extract_numeric_candidates(answer: str) -> List[float]:
    """Return plausible numeric interpretations of the final FinQA answer."""
    answer = extract_finqa_answer(answer)
    matches = list(_NUMBER_WITH_UNIT_RE.finditer(answer))
    if not matches:
        return []

    match = matches[-1]
    value = _coerce_number(match.group("number"))
    if value is None:
        return []

    scale = _scale_for_unit(match.group("unit"))
    values = [value]
    if scale != 1.0:
        values.append(value * scale)
    if match.group("percent"):
        values.append(value / 100.0)
        if scale != 1.0:
            values.append(value * scale / 100.0)

    deduped = []
    for item in values:
        if not any(math.isclose(item, existing, rel_tol=0.0, abs_tol=0.0) for existing in deduped):
            deduped.append(item)
    return deduped


def _normalized_text(answer: str) -> str:
    answer = extract_finqa_answer(answer).lower()
    answer = answer.replace("$", "").replace(",", "").replace("%", "")
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip(" .")


def finqa_answers_match(prediction: str, ground_truth_answer: str) -> bool:
    pred_values = extract_numeric_candidates(prediction)
    truth_values = extract_numeric_candidates(ground_truth_answer)

    if pred_values and truth_values:
        for pred_value in pred_values:
            for truth_value in truth_values:
                if math.isclose(pred_value, truth_value, rel_tol=1e-4, abs_tol=1e-4):
                    return True
        return False

    return bool(_normalized_text(prediction)) and _normalized_text(prediction) == _normalized_text(ground_truth_answer)


def finqa_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable) -> int:
    truth = parse_finqa_ground_truth(ground_truth_answer.value)
    return int(finqa_answers_match(str(prediction.value), truth["answer"]))


def _truncate(text: str, max_chars: int = 8000) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return f"{text[:keep]}\n...[truncated]...\n{text[-keep:]}"


def parse_finqa_ground_truth(value) -> Dict[str, str]:
    """Parse a FinQA target value.

    Newer FinQA examples store answer/program metadata as JSON. Plain answer
    strings are still accepted so older cached data or small tests keep working.
    """
    if isinstance(value, dict):
        data = value
    else:
        text = str(value).strip()
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            data = {"answer": text}
        if not isinstance(data, dict):
            data = {"answer": text}

    gold_inds = data.get("gold_inds", "")
    if gold_inds in ("", None):
        gold_inds_text = ""
    else:
        gold_inds_text = json.dumps(gold_inds, ensure_ascii=False)

    return {
        "answer": str(data.get("answer", data.get("exe_ans", ""))).strip(),
        "program": str(data.get("program", "")).strip(),
        "program_re": str(data.get("program_re", "")).strip(),
        "gold_inds": gold_inds_text,
        "id": str(data.get("id", "")).strip(),
    }


class FinQAEvaluation(Function):
    """Deterministic FinQA scoring plus LLM-based textual diagnosis.

    Forward pass returns only ``"1"`` or ``"0"`` for stable accuracy reporting.
    Backward pass calls an LLM critic only for incorrect examples and attaches
    the critic's diagnosis as feedback to the model response.
    """

    def __init__(
        self,
        critic_system_prompt: str = FINQA_CRITIC_SYSTEM_PROMPT,
        include_gold_program: bool = True,
        include_gold_program_re: bool = True,
        include_gold_inds: bool = True,
    ):
        super().__init__()
        self.uses_question_context = True
        self.critic_system_prompt = critic_system_prompt
        self.include_gold_program = include_gold_program
        self.include_gold_program_re = include_gold_program_re
        self.include_gold_inds = include_gold_inds
        self.function_purpose = (
            "Deterministically check FinQA final-answer correctness, then use "
            "LLM diagnosis only to explain incorrect answers and suggest prompt "
            "updates."
        )

    def forward(self, inputs: Dict[str, tg.Variable], response_role_description: str = None) -> tg.Variable:
        prediction = inputs["prediction"]
        ground_truth = inputs["ground_truth_answer"]
        truth = parse_finqa_ground_truth(ground_truth.value)
        is_correct = finqa_answers_match(str(prediction.value), truth["answer"])

        if response_role_description is None:
            response_role_description = "deterministic FinQA answer correctness, 1 if correct and 0 if incorrect"

        response = tg.Variable(
            int(is_correct),
            predecessors=list(inputs.values()),
            role_description=response_role_description,
        )
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                inputs=inputs,
                is_correct=is_correct,
                truth=truth,
            )
        )
        return response

    def backward(
        self,
        response: tg.Variable,
        inputs: Dict[str, tg.Variable],
        is_correct: bool,
        truth: Dict[str, str],
        backward_engine,
    ):
        if is_correct:
            return

        prediction = inputs["prediction"]
        if not prediction.requires_grad:
            return

        question = inputs.get("question")
        question_value = question.value if question is not None else ""
        gold_program = truth["program"] if self.include_gold_program else "REMOVED_FOR_ABLATION"
        gold_program_re = truth["program_re"] if self.include_gold_program_re else "REMOVED_FOR_ABLATION"
        gold_inds = truth["gold_inds"] if self.include_gold_inds else "REMOVED_FOR_ABLATION"
        critic_prompt = FINQA_DIAGNOSIS_PROMPT_TEMPLATE.format(
            question=_truncate(question_value, 8000),
            gold_answer=truth["answer"],
            gold_program=gold_program or "N/A",
            gold_program_re=gold_program_re or "N/A",
            gold_inds=gold_inds or "N/A",
            prediction=_truncate(prediction.value, 4000),
            predicted_answer=extract_finqa_answer(prediction.value) or "N/A",
        )
        diagnosis = backward_engine(critic_prompt, system_prompt=self.critic_system_prompt)
        gradient = tg.Variable(
            diagnosis,
            role_description="FinQA critic diagnosis and textual gradient for the model response",
        )
        prediction.gradients.add(gradient)
        prediction.gradients_context[gradient] = {
            "context": critic_prompt,
            "response_desc": response.get_role_description(),
            "variable_desc": prediction.get_role_description(),
        }


def _format_text_lines(items: Iterable[str]) -> str:
    lines = []
    for item in items:
        text = " ".join(str(item).split())
        if text:
            lines.append(text)
    return "\n".join(lines)


def _format_text_block(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _format_text_lines([value])
    return _format_text_lines(value)


def _format_table(table) -> str:
    rows = []
    for row in table or []:
        if isinstance(row, (list, tuple)):
            cells = [" ".join(str(cell).split()) for cell in row]
        else:
            cells = [" ".join(str(row).split())]
        rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    return "\n".join(" | ".join(row) for row in padded_rows)


class FinQA(Dataset):
    """FinQA numerical reasoning over financial report text and tables.

    The official dataset is stored in JSON files in the FinQA GitHub repository.
    This wrapper uses compact deterministic splits for prompt optimization:
        - train: first 50 examples from train.json
        - val: first 100 examples from dev.json
        - test: first 100 examples from test.json

    Pass ``max_samples=-1`` to use a full split.
    """

    def __init__(
        self,
        root: str = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        raw_base_url: str = FINQA_RAW_BASE_URL,
        *args,
        **kwargs,
    ):
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        assert split in FINQA_SPLIT_FILES

        self.root = Path(root) / "finqa"
        self.split = split
        self.raw_base_url = raw_base_url.rstrip("/")
        self._task_description = (
            "You will answer financial numerical reasoning questions using the "
            "given report text and table. Think step by step, track units "
            "carefully, perform the needed calculations, and finish your answer "
            "with a final line exactly formatted as 'Answer: $VALUE' where "
            "$VALUE is the final answer."
        )

        self._check_or_download_dataset()
        raw_path = self.root / FINQA_SPLIT_FILES[split]
        with raw_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples is None:
            max_samples = FINQA_DEFAULT_SPLIT_SIZES[split]
        if max_samples is not None and max_samples >= 0:
            data = data[:max_samples]
        self.data = data

    def _check_or_download_dataset(self):
        self.root.mkdir(parents=True, exist_ok=True)
        filename = FINQA_SPLIT_FILES[self.split]
        output_path = self.root / filename
        if output_path.exists():
            return

        url = f"{self.raw_base_url}/{filename}"
        try:
            urllib.request.urlretrieve(url, output_path)
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Could not download FinQA split {self.split!r} from {url}") from exc

    @staticmethod
    def _format_ground_truth(example):
        qa = example.get("qa", {})
        return json.dumps(
            {
                "answer": str(qa.get("exe_ans", "")).strip(),
                "program": qa.get("program", ""),
                "program_re": qa.get("program_re", ""),
                "gold_inds": qa.get("gold_inds", ""),
                "id": example.get("id", ""),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _format_example(example):
        qa = example.get("qa", {})
        question = str(qa.get("question", "")).strip()

        sections = []
        pre_text = _format_text_block(example.get("pre_text"))
        table = _format_table(example.get("table"))
        post_text = _format_text_block(example.get("post_text"))

        if pre_text:
            sections.append(f"Pre-table text:\n{pre_text}")
        if table:
            sections.append(f"Table:\n{table}")
        if post_text:
            sections.append(f"Post-table text:\n{post_text}")
        sections.append(f"Question: {question}")
        return "\n\n".join(sections), FinQA._format_ground_truth(example)

    def __getitem__(self, index):
        return self._format_example(self.data[int(index)])

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return self._task_description

    def get_default_task_instruction(self):
        return self._task_description
