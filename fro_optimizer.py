"""
Fractal Resonant Optimization (FRO).

Memory-conscious optimizer for ZetaGrid Soul training. FRO keeps first
momentum, structured second moments, and per-parameter multi-scale resonance
statistics. It avoids AdamW's full second-moment tensor for matrix weights by
tracking row-wise gradient energy.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


class FRO(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        scales: Sequence[float] = (0.1, 0.01, 0.001),
        alpha: float = 0.1,
        gamma: float = 0.5,
        weight_decay: float = 0.0,
        resonance_floor: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not scales:
            raise ValueError("scales must not be empty")
        for scale in scales:
            if not 0 < scale <= 1:
                raise ValueError("all scales must be in (0, 1]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            scales=tuple(float(s) for s in scales),
            alpha=float(alpha),
            gamma=float(gamma),
            weight_decay=float(weight_decay),
            resonance_floor=float(resonance_floor),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            alpha = group["alpha"]
            gamma = group["gamma"]
            weight_decay = group["weight_decay"]
            scales = group["scales"]
            resonance_floor = group["resonance_floor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FRO does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if grad.ndim >= 2:
                        state["exp_avg_sq"] = torch.zeros(
                            grad.shape[:-1], dtype=torch.float32, device=grad.device
                        )
                        state["structured_second"] = True
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                        state["structured_second"] = False
                    state["rho_mu"] = torch.zeros(len(scales), dtype=torch.float32, device=grad.device)
                    state["rho_sq"] = torch.zeros(len(scales), dtype=torch.float32, device=grad.device)
                    state["last_resonance"] = torch.tensor(0.0, dtype=torch.float32, device=grad.device)
                    state["last_rho"] = torch.tensor(0.0, dtype=torch.float32, device=grad.device)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                grad_f = grad.float()
                avg_f = exp_avg.float()
                dot = torch.sum(grad_f * avg_f)
                denom_rho = grad_f.norm() * avg_f.norm() + eps
                rho = torch.clamp(dot / denom_rho, min=-1.0, max=1.0)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                for idx, lam in enumerate(scales):
                    state["rho_mu"][idx].mul_(1.0 - lam).add_(rho, alpha=lam)
                    state["rho_sq"][idx].mul_(1.0 - lam).add_(rho * rho, alpha=lam)

                resonance_terms = (state["rho_mu"].pow(2) / (state["rho_sq"] + eps)).clamp_min(eps)
                resonance = torch.exp(torch.mean(torch.log(resonance_terms))).clamp(0.0, 1.0)
                if resonance_floor > 0:
                    resonance = resonance.clamp_min(resonance_floor)
                state["last_resonance"].copy_(resonance)
                state["last_rho"].copy_(rho)

                if state["structured_second"]:
                    row_energy = grad_f.pow(2).mean(dim=-1)
                    exp_avg_sq.mul_(beta2).add_(row_energy, alpha=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().unsqueeze(-1).to(p.dtype).add_(eps)
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().to(p.dtype).add_(eps)

                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                adaptive = alpha + (1.0 - alpha) * gamma * float(resonance)
                step_size = lr * adaptive / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def resonance_summary(self) -> dict[str, float]:
        values = []
        rhos = []
        for state in self.state.values():
            if "last_resonance" in state:
                values.append(float(state["last_resonance"].detach().cpu()))
                rhos.append(float(state["last_rho"].detach().cpu()))
        if not values:
            return {"resonance": 0.0, "rho": 0.0}
        return {
            "resonance": math.fsum(values) / len(values),
            "rho": math.fsum(rhos) / len(rhos),
        }
