"""Linear-warmup, cosine-decay learning rate schedule (task spec §22)."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_schedule(optimizer: Optimizer, total_steps: int, warmup_frac: float = 0.05) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
