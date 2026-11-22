import torch
import torch.nn as nn

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
except ImportError:
    LigerRMSNormFunction = None


class RMSNorm(nn.RMSNorm):
    """RMSNorm：把最后一维缩放到均方根为 1，再逐维乘一个可学的系数（初始为 1）。

    系数只有 dim 个，推理时可以并进后面的线性层，不花额外的算力。
    """

    def __init__(self, dim: int):
        super().__init__(dim)

    def forward(self, x):
        if x.is_cuda and x.dtype != torch.float64 and LigerRMSNormFunction is not None:
            if torch.compiler.is_compiling():
                # dynamo 追踪 Liger 的 autograd.Function 会生成损坏的 FX 图（多出引用不存在参数的
                # _cuda_maybe_exchange_device 节点）。gemma 模式加的 offset 是 0，数学上就是普通
                # RMSNorm，Liger 只是在 fp32 里算中间结果，这里照搬即可。
                return super().forward(x.float()).type_as(x)
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            return LigerRMSNormFunction.apply(x, self.weight, eps, 0.0, "gemma", False)
        return super().forward(x)
