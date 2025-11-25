# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba2 import Mamba2
# from mamba_ssm.modules.mha import MHA
# from mamba_ssm.modules.mlp import GatedMLP
# from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,# 模型维度（隐藏层大小）
    d_intermediate,# MLP中间层维度，为0时不使用MLP
    ssm_cfg=None, # Mamba SSM配置
    attn_layer_idx=None, # 使用注意力机制的层索引列表
    attn_cfg=None, # 注意力机制配置
    norm_epsilon=1e-5,# 归一化层的epsilon值
    rms_norm=False,# 是否使用RMSNorm（替代LayerNorm）
    residual_in_fp32=False,# 残差连接是否使用FP32精度
    fused_add_norm=False,# 是否使用融合的add+norm操作
    layer_idx=None,# 当前层索引
    device=None,# 设备（如'cuda'或'cpu'）
    dtype=None, # 数据类型
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        # 创建mamba混合器
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        # # 创建多头注意力混合器
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    # 根据配置选择标准LayerNorm或RMSNorm。
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    # 如果d_intermediate为0，则不使用MLP；否则创建GatedMLP
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    # 组装并返回Block ：
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,  # 需要初始化的模块
    n_layer, # 模型总层数
    initializer_range=0.02,  # Now only used for embedding layer.# 初始化范围（现在仅用于嵌入层）
    rescale_prenorm_residual=True,# 是否重新缩放预归一化残差路径上的权重
    n_residuals_per_layer=1,  # Change to 2 if we have MLP# 每层的残差连接数量（有MLP时改为2）
):
    '''
    - 检查模块是否为线性层
    - 如果线性层有偏置项（bias）
    - 且偏置项没有设置 _no_reinit 属性（这是一个避免重复初始化的标记）
    - 则将偏置项初始化为全零
    - 这是一种常见做法，因为线性层的权重会被其他初始化方法处理，而偏置项初始化为零有助于训练稳定性
    '''
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    '''
    - 如果模块是嵌入层
    - 使用正态分布初始化嵌入权重，标准差为 initializer_range （默认为0.02）
    - 这是语言模型中嵌入层的标准初始化方式，源自 GPT 系列模型
    '''
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    # 残差路径权重的重新缩放 （如果启用）：
    if rescale_prenorm_residual:
        # 当 rescale_prenorm_residual 为 True 时（默认启用），对残差路径上的特定权重进行重新缩放
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        '''
        - 遍历模块的所有命名参数
        - 仅对特定名称的权重进行处理：
        - out_proj.weight ：通常是多头注意力机制的输出投影层权重
        - fc2.weight ：通常是 MLP 层的第二个全连接层权重
        - 这些层位于残差连接路径上，需要特殊处理
        '''
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                # - 参数 a=math.sqrt(5) 是 PyTorch 线性层的默认值，对应于 ReLU 激活函数，这一步确保权重有适当的初始范围，避免梯度消失或爆炸问题
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    # MixerModel 是 Mamba 架构中的核心模型类，继承自 PyTorch 的 nn.Module 。
    # 该类实现了一个基于状态空间模型(SSM)的序列处理网络，可以混合使用 Mamba 层和传统的注意力层，
    # 具有极高的序列处理效率和灵活性。
    def __init__(
        self,
        d_model: int, # 模型维度（隐藏层大小）
        n_layer: int,# 模型层数
        d_intermediate: int,# MLP中间层维度，为0时不使用MLP
        vocab_size: int, # 词汇表大小
        ssm_cfg=None,# Mamba SSM配置
        attn_layer_idx=None, # 使用注意力机制的层索引列表
        attn_cfg=None, # 注意力机制配置
        norm_epsilon: float = 1e-5, # 归一化层的epsilon值
        rms_norm: bool = False,# 是否使用RMSNorm（替代LayerNorm）
        initializer_cfg=None,# 权重初始化配置
        fused_add_norm=False,# 是否使用融合的add+norm操作
        residual_in_fp32=False, # 残差连接是否使用FP32精度
        device=None,# 设备（如'cuda'或'cpu'）
        dtype=None,# 数据类型
    ) -> None:
        # 创建工厂参数字典，用于后续层的初始化
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类初始化,也就是上面的mamba类
        super().__init__()
        # 保存残差连接精度配置
        self.residual_in_fp32 = residual_in_fp32
        # 创建词汇表嵌入层，将词元索引转换为高维向量表示
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        '''
        - 实现了与标准 Transformer 不同的残差连接顺序：将 Add 操作移到 LN 之前
        - 这种改变不会影响模型功能，但允许融合 add 和 layer_norm 操作以提高性能
        - 如果启用融合操作，确保已导入相应的 Triton 内核函数
        '''
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        '''
        - 使用 nn.ModuleList 创建模型层的堆叠
        - 通过之前分析的 create_block 函数动态创建每一层
        - 根据配置，某些层可能使用 Mamba，而其他层可能使用注意力机制
        - 传递所有必要的配置参数，确保每一层的一致性
        '''
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        # - 创建模型的最终归一化层, 根据配置选择标准 LayerNorm 或 RMSNorm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        # - 应用之前分析的 _init_weights 函数对所有参数进行初始化
        # - 动态设置 n_residuals_per_layer ：如果有 MLP 则为 2，否则为 1
        # - 允许通过 initializer_cfg 传入额外的初始化配置
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        # - 将输入的 token 索引转换为嵌入向量, 初始化残差连接变量为 None
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
