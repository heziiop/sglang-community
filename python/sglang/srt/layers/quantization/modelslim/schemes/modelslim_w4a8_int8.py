"""ModelSlim W4A8_DYNAMIC dense linear scheme for Ascend NPU."""

from typing import Any, Dict, List

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW4A8Int8DynamicLinearMethod,
)
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme


class ModelSlimW4A8Int8(ModelSlimLinearScheme):
    """W4A8_DYNAMIC linear weights exported by msModelSlim.

    Version 1.0.0 packs two int4 values along the output dimension. Older
    exports keep one int4 value per int8 element and are converted to the NPU
    int4 layout during post-loading.
    """

    def __init__(self, quant_config: Dict[str, Any], prefix: str):
        self.quant_config = quant_config
        self.prefix = prefix
        self.group_size = int(quant_config.get("group_size", 0) or 0)
        self.new_quant_version = quant_config.get("version", "0") == "1.0.0"
        self.kernel = NPUW4A8Int8DynamicLinearMethod(
            quant_config=quant_config,
            group_size=self.group_size,
            new_quant_version=self.new_quant_version,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        stored_output = (
            output_size_per_partition // 2
            if self.new_quant_version
            else output_size_per_partition
        )

        if self.new_quant_version:
            weight = PackedvLLMParameter(
                packed_factor=2,
                packed_dim=0,
                data=torch.empty(
                    stored_output, input_size_per_partition, dtype=torch.int8
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            weight = ModelWeightParameter(
                data=torch.empty(
                    stored_output, input_size_per_partition, dtype=torch.int8
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, 1, dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        weight_offset = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, 1, dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_offset", weight_offset)

        if self.group_size > 0:
            second_shape = (
                output_size_per_partition,
                input_size_per_partition // self.group_size,
            )
            scale_second = GroupQuantScaleParameter(
                data=torch.empty(second_shape, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale_second", scale_second)
            offset_second = GroupQuantScaleParameter(
                data=torch.empty(second_shape, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_offset_second", offset_second)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias=None):
        return self.kernel.apply(layer, x, bias=bias)
