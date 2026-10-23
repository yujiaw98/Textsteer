import platformdirs
import re
import textgrad as tg
from .base import Dataset


def _valid_letters(n: int):
    return tuple(chr(ord("A") + i) for i in range(n))


def parse_mc_letter_answer(answer: str, max_choices: int = 10) -> str:
    """Extract a multiple-choice letter (A..) from a model's free-form answer.

    ``max_choices`` controls how many letters are valid; default 10 covers A-J for MMLU-Pro.
    """
    if answer is None:
        return ""
    valid = _valid_letters(max_choices)
    last_letter = valid[-1]
    letter_class = f"A-{last_letter}a-{last_letter.lower()}"
    text = str(answer).strip()

    match = re.search(rf"Answer\s*[:=]\s*\(?\*?\*?([{letter_class}])\)?\*?\*?", text)
    if match:
        return match.group(1).upper()

    tail = text.splitlines()[-1] if text else ""
    match = re.search(rf"\(?([{letter_class}])\)?\s*$", tail.strip())
    if match:
        return match.group(1).upper()

    for token in reversed(text.split()):
        cleaned = "".join(c for c in token if c.isalpha()).upper()
        if cleaned in valid:
            return cleaned
    return ""


def mmlu_pro_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable) -> int:
    pred = parse_mc_letter_answer(str(prediction.value), max_choices=10)
    truth = str(ground_truth_answer.value).strip().upper()
    truth = truth[0] if truth else ""
    return int(pred == truth and len(truth) > 0)


class MMLUPro(Dataset):
    """MMLU-Pro multiple-choice reasoning benchmark.

    <https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro>

    The official dataset ships only ``test`` (~12k) and ``validation`` (~70) splits.
    For prompt optimization we filter to a single ``category`` and carve a
    50/100/100 train/val/test split out of the test slice (matches AQuA/BBH).
    """

    def __init__(self, category: str = "math", root: str = None, split: str = "train", *args, **kwargs):
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        self.root = root
        self.category = category
        assert split in ["train", "val", "test"]
        self.split = split

        full = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=root, split="test")
        full = full.filter(lambda ex: ex["category"] == category)
        if len(full) < 250:
            raise ValueError(
                f"Not enough samples in category={category!r}: only {len(full)} (< 250 needed for 50/100/100 split)."
            )

        if split == "train":
            self.data = full.select(range(0, 50))
        elif split == "val":
            self.data = full.select(range(50, 150))
        else:
            self.data = full.select(range(150, 250))

        self._task_description = (
            "You will answer a multiple-choice question. Think step by step. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' where $LETTER is the letter (one of A-J) of the correct option."
        )

    def get_task_description(self):
        return self._task_description

    def __getitem__(self, index):
        row = self.data[int(index)]
        question = str(row["question"]).strip()
        options = row["options"]
        options_block = "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))
        x = f"{question}\nOptions:\n{options_block}"
        y = str(row["answer"]).strip().upper()
        return x, y

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self):
        return self._task_description
