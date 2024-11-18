# credit to https://github.com/andmev/llama-to-coreml/blob/master/src/converter.py

# todo: look at https://forums.developer.apple.com/forums/thread/729942
# and https://machinelearning.apple.com/research/core-ml-on-device-llama to see
# if i can get this working on neural engine

import coremltools as ct
import torch
import numpy as np

class LlamaCoreMLConverter:
    def __init__(
        self,
        model,
        batch_size: int = 1,
        context_size: int = 2048
    ):
        self.model = model
        self.batch_size = batch_size
        self.context_size = context_size

    def convert(self, quantize: bool = True) -> ct.models.MLModel:
        # Initialize and trace PyTorch model
        example_inputs = (
            torch.zeros((1, 2), dtype=torch.int32),
            torch.zeros((1, 1, 2, 5), dtype=torch.float32)
        )
        traced_model = torch.jit.trace(
            self.model.eval(),
            example_inputs=example_inputs
        )

        # Define input shapes with RangeDim for dynamic dimensions
        inputs = [
            ct.TensorType(
                shape=(self.batch_size, ct.RangeDim(1, 2048, default=1)),
                dtype=np.int32,
                name="inputIds"
            ),
            ct.TensorType(
                shape=(self.batch_size, 1, ct.RangeDim(1, 2048, default=1), ct.RangeDim(1, 2048, default=1)),
                dtype=np.float16,
                name="causalMask"
            ),
        ]
        
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=self.model.kv_cache_shape,
                    dtype=np.float16
                ),
                name="keyCache"
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=self.model.kv_cache_shape,
                    dtype=np.float16
                ),
                name="valueCache"
            ),
        ]

        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=[ct.TensorType(dtype=np.float16, name="logits")],
            states=states,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,  # Enable Neural Engine
            skip_model_load=True,
        )

        if quantize:
            mlmodel = self._quantize_model(mlmodel)

        return mlmodel

    def _quantize_model(self, mlmodel: ct.models.MLModel) -> ct.models.MLModel:
        # Use int4 quantization as recommended in the Apple ML blog
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",  # Changed from int8 to int4
            granularity="per_block",  # Changed from per_channel to per_block
            block_size=32,
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        return ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config) 