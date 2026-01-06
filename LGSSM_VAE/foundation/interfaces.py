from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass(frozen=True)
class StepOutput:
    """训练 step 的统一输出。

    设计目标：trainer 不依赖具体模型类型，只依赖该结构中的最小字段。

    约定：
    - 所有标量字段均为 0-dim Tensor（可用于反向传播或日志统计）。
    - recon 代表“观测点上的重构损失”（VAE: Gaussian NLL；AE: MSE）。
    - kl/kl_used：VAE 才有意义；AE 返回 0。
    - logvar_x：若模型不输出不确定性，则为 None（推理侧可用 fixed sigma 补齐）。
    """

    loss: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor
    kl_used: torch.Tensor
    mean: torch.Tensor
    logvar_x: Optional[torch.Tensor] = None


class TrainableModel(Protocol):
    """训练/推理可用的模型协议（不要求继承某个基类）。"""

    def training_step(
        self,
        x_input: torch.Tensor,
        mask_keep: torch.Tensor,
        x_target: torch.Tensor,
        *,
        beta: Optional[float] = None,
    ) -> StepOutput: ...

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_mean: bool = True,
        *,
        return_logvar: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]: ...


__all__ = ["StepOutput", "TrainableModel"]

