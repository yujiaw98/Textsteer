import os
import re
import json
import pandas as pd
import subprocess
import platformdirs
import textgrad as tg
from .base import Dataset


VALID_LETTERS = ("A", "B", "C", "D", "E")


def parse_letter_answer(answer: str) -> str:
    """Extract the multiple-choice letter (A-E) from a model's free-form answer."""
    if answer is None:
        return ""
    text = str(answer).strip()

    match = re.search(r"Answer\s*[:=]\s*\(?\*?\*?([A-Ea-e])\)?\*?\*?", text)
    if match:
        return match.group(1).upper()

    tail = text.splitlines()[-1] if text else ""
    match = re.search(r"\(?([A-Ea-e])\)?\s*$", tail.strip())
    if match:
        return match.group(1).upper()

    for token in reversed(text.split()):
        cleaned = "".join(c for c in token if c.isalpha()).upper()
        if cleaned in VALID_LETTERS:
            return cleaned
    return ""


def aqua_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable) -> int:
    pred = parse_letter_answer(str(prediction.value))
    truth = str(ground_truth_answer.value).strip().upper()
    truth = truth[0] if truth else ""
    return int(pred == truth and pred in VALID_LETTERS)


class AQuA(Dataset):
    def __init__(self, root: str = None, split: str = "train", *args, **kwargs):
        """
        AQuA-RAT multiple-choice algebraic word problems.
        <https://github.com/google-deepmind/AQuA>

        Train/val/test are built from AQuA's ``dev.json`` (254 examples) and
        ``test.json`` (254 examples) so we avoid downloading the 30MB+
        ``train.json``:
            - train: dev[:50]
            - val:   dev[50:150]
            - test:  test[:100]

        Args:
            root (string): Root directory of the dataset cache.
            split (string, optional): One of ``"train"``, ``"val"``, ``"test"``.
        """
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        self.root = root
        self.split = split
        self._dirname = "aqua"
        self._check_or_download_dataset()
        assert split in ["train", "val", "test"]
        data_path = os.path.join(self.root, self._dirname, f"{split}.csv")
        self.data = pd.read_csv(data_path, index_col=0)
        self._task_description = (
            "You will answer a multiple-choice math word problem. Think step by step. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' where $LETTER is one of A, B, C, D, or E."
        )

    def get_task_description(self):
        return self._task_description

    @staticmethod
    def _format_example(ex):
        """Build the prompt input and target letter for one raw AQuA example."""
        question = ex["question"].strip()
        options = "\n".join(opt.strip() for opt in ex["options"])
        x = f"{question}\nOptions:\n{options}"
        y = str(ex["correct"]).strip().upper()
        return {"x": x, "y": y}

    @staticmethod
    def _load_jsonl(path):
        examples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
        return examples

    def _check_or_download_dataset(self):
        data_path = os.path.join(self.root, self._dirname, f"{self.split}.csv")
        if os.path.exists(data_path):
            return

        os.makedirs(os.path.join(self.root, self._dirname), exist_ok=True)
        base_url = "https://raw.githubusercontent.com/google-deepmind/AQuA/master"
        raw_dir = os.path.join(self.root, self._dirname, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        for fname in ("dev.json", "test.json"):
            local = os.path.join(raw_dir, fname)
            if not os.path.exists(local):
                subprocess.call(["wget", f"{base_url}/{fname}", "-O", local])

        dev_examples = self._load_jsonl(os.path.join(raw_dir, "dev.json"))
        test_examples = self._load_jsonl(os.path.join(raw_dir, "test.json"))

        train_rows = [self._format_example(ex) for ex in dev_examples[:50]]
        val_rows = [self._format_example(ex) for ex in dev_examples[50:150]]
        test_rows = [self._format_example(ex) for ex in test_examples[:100]]

        for split_name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
            out_path = os.path.join(self.root, self._dirname, f"{split_name}.csv")
            with open(out_path, "w") as f:
                pd.DataFrame(rows).to_csv(f)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row["x"], row["y"]

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self):
        return self._task_description
