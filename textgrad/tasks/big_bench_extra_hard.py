import os
import json
import pandas as pd
import subprocess
import platformdirs
from .base import Dataset
from .big_bench_hard import parse_integer_answer, string_based_equality_fn  # re-exported for callers


class BigBenchExtraHard(Dataset):
    def __init__(self, task_name: str, root: str = None, split: str = "train", *args, **kwargs):
        """
        Tasks from BIG-Bench Extra Hard (BBEH)
        <https://github.com/google-deepmind/bbeh>

        The train, val, test splits are constructed from 50/50/100 examples
        (BBEH ships 200 examples per task).

        Args:
            task_name (string): BBEH task name without the ``bbeh_`` prefix
                (e.g. ``"object_counting"``).
            root (string): Root directory of the dataset cache.
            split (string, optional): The dataset split, supports ``"train"``
                (default), ``"val"`` and ``"test"``.
        """
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        self.root = root
        self.split = split
        self.task_name = task_name
        self._dirname = f"bbeh_{task_name}"
        self._check_or_download_dataset()
        assert split in ["train", "val", "test"]
        data_path = os.path.join(self.root, self._dirname, f"{split}.csv")
        self.data = pd.read_csv(data_path, index_col=0)
        self._task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."

    def get_task_description(self):
        return self._task_description

    def _check_or_download_dataset(self):
        data_path = os.path.join(self.root, self._dirname, f"{self.split}.csv")
        if os.path.exists(data_path):
            return

        os.makedirs(os.path.join(self.root, self._dirname), exist_ok=True)
        json_path = os.path.join(self.root, f"{self._dirname}.json")
        subprocess.call(
            [
                "wget",
                f"https://raw.githubusercontent.com/google-deepmind/bbeh/main/bbeh/benchmark_tasks/{self._dirname}/task.json",
                "-O",
                json_path,
            ]
        )
        data = json.load(open(json_path))
        examples = data["examples"]
        train_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[:50]]
        val_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[50:100]]
        test_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[100:]]
        train_path = os.path.join(self.root, self._dirname, "train.csv")
        with open(train_path, "w") as f:
            pd.DataFrame(train_examples).to_csv(f)
        val_path = os.path.join(self.root, self._dirname, "val.csv")
        with open(val_path, "w") as f:
            pd.DataFrame(val_examples).to_csv(f)
        test_path = os.path.join(self.root, self._dirname, "test.csv")
        with open(test_path, "w") as f:
            pd.DataFrame(test_examples).to_csv(f)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row["x"], row["y"]

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self):
        return self._task_description
