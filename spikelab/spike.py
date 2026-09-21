"""The spike nonlinearity: a Heaviside step, with an optional surrogate gradient.

Forward is always the true step. Backward is either zero (plain simulation, STDP)
or the fast-sigmoid surrogate from Zenke & Ganguli's SuperSpike:
d s / d v  ~  1 / (slope * |v - th| + 1)^2.
"""
from __future__ import annotations

import torch


def heaviside(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0).to(x.dtype)


class _FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (ctx.slope * x.abs() + 1.0) ** 2, None


def fast_sigmoid(slope: float):
    """A spike function whose backward pass is the fast-sigmoid surrogate."""
    return lambda x: _FastSigmoid.apply(x, slope)
