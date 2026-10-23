import re

import platformdirs
import textgrad as tg

from .base import Dataset


MATH_CATEGORIES = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _read_balanced_braces(text: str, start: int) -> str:
    depth = 0
    chars = []
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
            chars.append(char)
        else:
            chars.append(char)
    return ""


def extract_boxed_answer(text: str) -> str:
    """Extract the last LaTeX boxed answer from a MATH solution/prediction."""
    if text is None:
        return ""
    text = str(text)
    boxed_positions = [match.start() for match in re.finditer(r"\\(?:boxed|fbox)", text)]
    for pos in reversed(boxed_positions):
        brace_start = text.find("{", pos)
        if brace_start == -1:
            continue
        answer = _read_balanced_braces(text, brace_start)
        if answer:
            return answer.strip()
    return ""


def extract_math_answer(text: str) -> str:
    """Extract a final short answer from MATH-style free-form text."""
    if text is None:
        return ""
    text = str(text).strip()
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer:
        return boxed_answer

    answer_patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)$",
        r"therefore,?\s*(.+)$",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].strip().strip(".")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].strip().strip(".") if lines else ""


def _normalize_latex_fraction(answer: str) -> str:
    frac_pattern = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while previous != answer:
        previous = answer
        answer = frac_pattern.sub(r"\1/\2", answer)
    return answer


def normalize_math_answer(answer: str) -> str:
    """Normalize common presentation-only differences in short math answers."""
    if answer is None:
        return ""
    answer = str(answer).strip()
    answer = extract_math_answer(answer)
    answer = answer.strip().strip("$").strip()
    answer = _normalize_latex_fraction(answer)

    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\;": "",
        "\\:": "",
        "\\cdot": "*",
        "\\times": "*",
        "^\\circ": "",
        "\\%": "%",
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)

    answer = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\s+", "", answer)
    answer = answer.replace(",", "")
    return answer.lower()


def math_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable) -> int:
    pred = normalize_math_answer(str(prediction.value))
    truth = normalize_math_answer(str(ground_truth_answer.value))
    return int(bool(pred) and pred == truth)


class MATH(Dataset):
    """Hendrycks MATH competition mathematics benchmark.

    <https://arxiv.org/abs/2103.03874>

    For prompt optimization this wrapper exposes compact deterministic splits:
        - train: first 50 examples from the official train split
        - val: next 100 examples from the official train split
        - test: first 100 examples from the official test split

    Set ``category`` to one of ``MATH_CATEGORIES`` or ``"all"``.
    """

    def __init__(
        self,
        category: str = "all",
        root: str = None,
        split: str = "train",
        dataset_name: str = "EleutherAI/hendrycks_math",
        *args,
        **kwargs,
    ):
        from datasets import concatenate_datasets, load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        assert split in ["train", "val", "test"]

        self.root = root
        self.category = category
        self.split = split
        self.dataset_name = dataset_name

        categories = MATH_CATEGORIES if category == "all" else (category,)
        invalid_categories = sorted(set(categories) - set(MATH_CATEGORIES))
        if invalid_categories:
            raise ValueError(
                f"Unknown MATH category/categories: {invalid_categories}. "
                f"Available categories: {', '.join(MATH_CATEGORIES)} or all."
            )

        hf_split = "test" if split == "test" else "train"
        datasets = [
            load_dataset(dataset_name, category_name, cache_dir=root, split=hf_split)
            for category_name in categories
        ]
        full = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
        if category == "all":
            full = full.shuffle(seed=0)

        if split == "train":
            indices = range(0, min(50, len(full)))
        elif split == "val":
            indices = range(50, min(150, len(full)))
        else:
            indices = range(0, min(100, len(full)))
        self.data = full.select(indices)

        self._task_description = (
            "You will answer a competition mathematics problem. Think step by step. "
            "The last line of your response should be of the following format: "
            "'Answer: $VALUE' where $VALUE is the final mathematical answer."
        )

    def __getitem__(self, index):
        row = self.data[int(index)]
        problem = str(row["problem"]).strip()
        answer = extract_boxed_answer(row["solution"]) or extract_math_answer(row["solution"])
        return f"Problem: {problem}", answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return self._task_description

    def get_default_task_instruction(self):
        return self._task_description
