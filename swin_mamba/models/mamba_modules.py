
# Mamba相关模块实现
import torch
import torch.nn as nn
from functools import partial
from typing import Optional

from mamba_ssm.modules.mamba_simple import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MambaBlock(nn.Module):
    """
    Mamba模块，用于替换Swin Transformer中的WindowAttention
    """
    def __init__(
        self, 
        dim, 
        d_state=16,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="v2",
        if_divide_out=False,
        init_layer_scale=None,
        drop_path=0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # 创建Mamba模块
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}

        self.mixer = Mamba(
            d_model=dim, 
            d_state=d_state, 
            layer_idx=layer_idx, 
            bimamba_type=bimamba_type, 
            if_divide_out=if_divide_out, 
            init_layer_scale=init_layer_scale, 
            **ssm_cfg, 
            **factory_kwargs
        )

        # 归一化层
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(
            dim, eps=norm_epsilon, **factory_kwargs
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None, inference_params=None
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual
