# Custom FP8 Quantization Examples

This directory contains examples demonstrating how to use the custom FP8 quantization type (E5M2 format) with the compressed-tensors quantization pipeline.

## Overview

The custom FP8 E5M2 format provides:
- **5 exponent bits** + **2 mantissa bits** + **1 sign bit** = 8 bits total
- **Wider dynamic range** compared to E4M3 format
- **Lower precision** but suitable for certain model architectures
- **Extensible design** - easy to integrate custom CUDA/C++ operators

## Implementation Details

### Core Changes

The implementation adds custom FP8 support to `compressed-tensors` through:

1. **New Data Class** (`CUSTOM_FP8_E5M2_DATA`):
   - Defines the E5M2 format parameters
   - Provides a `cast_to_custom_fp8()` method (placeholder for custom operators)
   - Located in: `src/compressed_tensors/quantization/quant_args.py`

2. **Extended QuantizationArgs**:
   - New `custom_format` field to specify format type (e.g., "e5m2")
   - Updated `pytorch_dtype()` method to handle custom formats
   - Backward compatible with existing code

3. **Modified Quantization Function**:
   - `round_to_quantized_type()` now checks for custom format
   - Routes to appropriate casting function based on format

## Files in This Directory

- **`custom_fp8_recipe.yaml`**: YAML recipe for custom FP8 quantization
- **`custom_fp8_example.py`**: Simple PTQ example using Python API
- **`README.md`**: This file

## Usage

### Method 1: Using YAML Recipe

```python
from llmcompressor import oneshot

oneshot(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    recipe="custom_fp8_recipe.yaml",
    output_dir="./quantized_model"
)
```

### Method 2: Using Python API

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme={
        "input_activations": {
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "dynamic": True,
            "custom_format": "e5m2",  # Custom format identifier
        },
        "weights": {
            "num_bits": 8,
            "type": "float",
            "strategy": "channel",
            "symmetric": True,
            "custom_format": "e5m2",  # Custom format identifier
        },
    },
    ignore=["lm_head", "model.norm"],
)

oneshot(model=model, recipe=recipe)
```

## Integrating Custom Operators

The current implementation uses a placeholder casting function. To integrate your custom CUDA/C++ operator:

### Step 1: Register Your Custom Operator

```python
import torch

# Load your custom operator library
torch.ops.load_library("path/to/your/custom_fp8_ops.so")
```

### Step 2: Update the Casting Function

Edit `src/compressed_tensors/quantization/quant_args.py`:

```python
class CUSTOM_FP8_E5M2_DATA(FloatArgs):
    # ... existing code ...
    
    @staticmethod
    @torch.compile
    def cast_to_custom_fp8(x):
        # Replace placeholder with your custom operator
        return torch.ops.custom_ops.fp8_e5m2_cast(x)
```

### Step 3: Test Your Integration

```python
import torch
from compressed_tensors.quantization import CUSTOM_FP8_E5M2_DATA

# Test the custom casting
x = torch.randn(10, 10, dtype=torch.float32)
quantized = CUSTOM_FP8_E5M2_DATA.cast_to_custom_fp8(x)
print(f"Original dtype: {x.dtype}")
print(f"Quantized dtype: {quantized.dtype}")
```

## Running the Examples

### Prerequisites

```bash
# Install required packages
pip install llmcompressor transformers torch

# For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Run Simple PTQ Example

```bash
cd third_party/compressed-tensors/examples/custom_fp8_quantization
python custom_fp8_example.py
```

## Configuration Options

### Quantization Strategies

- **`tensor`**: Per-tensor quantization (one scale for entire tensor)
- **`channel`**: Per-channel quantization (one scale per output channel)
- **`group`**: Group quantization (one scale per group of elements)

### Dynamic vs Static Quantization

- **Dynamic** (`dynamic: true`): Quantization parameters computed at runtime
  - Better for activations
  - No calibration needed
  - Slight runtime overhead

- **Static** (`dynamic: false`): Quantization parameters pre-computed
  - Better for weights
  - Requires calibration data
  - Faster inference

### Symmetric vs Asymmetric

- **Symmetric** (`symmetric: true`): Zero-point is always 0
  - Simpler, faster
  - Good for weights

- **Asymmetric** (`symmetric: false`): Zero-point can be non-zero
  - Better accuracy
  - Good for activations

## Layers to Ignore

Typically, you should ignore:
- **Embedding layers**: `embed_tokens`, `wte`, etc.
- **Normalization layers**: `LayerNorm`, `RMSNorm`, etc.
- **Output heads**: `lm_head`, `classifier`, etc.
- **Activation functions**: `SiLU`, `GELU`, etc.
- **Positional encodings**: `RotaryEmbedding`, etc.

## Troubleshooting

### Issue: "extra='forbid'" Error

If you see an error about extra fields being forbidden:

```python
# Make sure custom_format is properly defined in QuantizationArgs
# Check that you're using the updated version of quant_args.py
```

### Issue: Custom Format Not Applied

```python
# Verify the format string matches exactly
custom_format: "e5m2"  # Correct
custom_format: "E5M2"  # Wrong (case-sensitive)
```

### Issue: Model Size Not Reduced

The placeholder implementation doesn't actually compress the model. To see compression:
1. Integrate a real custom operator
2. Or use the model with an inference engine that supports the format

## Performance Considerations

### Memory Usage

- Custom FP8 format: **8 bits per parameter**
- FP16 format: **16 bits per parameter**
- Expected compression: **~2x** reduction in model size

### Inference Speed

- Speed depends on hardware support for custom format
- With proper CUDA kernels: **1.5-2x** faster than FP16
- Without hardware support: May be slower due to conversion overhead

### Accuracy

- E5M2 format has wider range but lower precision than E4M3
- Suitable for models with large activation ranges
- May require calibration for best accuracy

## Next Steps

1. **Test with your model**: Try quantizing your specific model architecture
2. **Integrate custom operator**: Replace placeholder with real implementation
3. **Benchmark performance**: Measure speed and accuracy improvements
4. **Deploy with vLLM**: Use quantized model in production

## Additional Resources

- [Compressed Tensors Documentation](https://github.com/vllm-project/compressed-tensors)
- [LLM Compressor Documentation](https://github.com/vllm-project/llm-compressor)
- [FP8 Quantization Paper](https://arxiv.org/abs/2209.05433)
- [vLLM Documentation](https://docs.vllm.ai/)

## Support

For issues or questions:
1. Check the main implementation plan: `CUSTOM_FP8_IMPLEMENTATION_PLAN_DETAILED.md`
2. Review the code changes in `src/compressed_tensors/quantization/quant_args.py`
3. Open an issue in the repository

## License

This implementation follows the same license as the compressed-tensors library (Apache 2.0).
