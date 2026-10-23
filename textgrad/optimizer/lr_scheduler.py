"""Textual learning-rate schedulers, ported from SkillOpt (microsoft/SkillOpt).

In SkillOpt the "textual learning rate" L_t is an *edit budget*: the maximum
number of edits a single optimization step may apply to the variable, which
controls how far one version is allowed to move from the previous one. These
schedulers reproduce that budget over the course of training (constant, linear,
cosine, autonomous). Step indexing is 1-based, results are rounded and floored
at ``min_lr``, matching the reference implementation.
"""

import math

# Sentinel returned by the autonomous schedule: effectively unbounded edits.
NO_LIMIT = 999


class LRScheduler:
    """Base class for textual learning-rate schedulers.

    :param max_lr: the largest edit budget (used at the start of training)
    :param min_lr: the smallest edit budget / floor the schedule decays toward
    :param total_steps: the total number of optimization steps the schedule spans
    """

    def __init__(self, max_lr: int = 4, min_lr: int = 2, total_steps: int = 1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self._step = 1  # 1-based, matches SkillOpt

    def _compute_lr(self, step: int) -> int:
        raise NotImplementedError

    def get_lr(self) -> int:
        """Edit budget for the current step."""
        return self._compute_lr(self._step)

    def step(self):
        """Advance to the next step. Call once per completed optimization step."""
        self._step += 1

    @property
    def current_step(self) -> int:
        return self._step


class ConstantScheduler(LRScheduler):
    def _compute_lr(self, step: int) -> int:
        return self.max_lr


class LinearScheduler(LRScheduler):
    def _compute_lr(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_lr
        t = min(step, self.total_steps) / self.total_steps
        lr = self.max_lr + (self.min_lr - self.max_lr) * t
        return max(self.min_lr, round(lr))


class CosineScheduler(LRScheduler):
    def _compute_lr(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_lr
        t = min(step, self.total_steps) / self.total_steps
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        return max(self.min_lr, round(lr))


class AutonomousScheduler(LRScheduler):
    """Unbounded edits: the optimizer decides how much to change on its own."""

    def _compute_lr(self, step: int) -> int:
        return NO_LIMIT


_SCHEDULERS = {
    "constant": ConstantScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
    "autonomous": AutonomousScheduler,
}


def build_scheduler(mode: str = "cosine", max_lr: int = 4, min_lr: int = 2, total_steps: int = 1) -> LRScheduler:
    """Instantiate a scheduler by mode string ('constant', 'linear', 'cosine', 'autonomous')."""
    mode = (mode or "cosine").lower()
    if mode not in _SCHEDULERS:
        raise ValueError(f"Unknown lr schedule '{mode}'. Choose from {sorted(_SCHEDULERS)}.")
    return _SCHEDULERS[mode](max_lr=max_lr, min_lr=min_lr, total_steps=total_steps)
