# Custom FP8 Quantization Implementation

This directory contains the implementation and examples for custom FP8 quantization using the E5M2 format (5-bit exponent, 2-bit mantissa).

## Overview

This implementation adds support for custom FP8 quantization formats to the llm-compressor/compressed-tensors pipeline. The key feature is the ability to use custom FP8 formats beyond the standard E4M3 format, enabling experimentation with different precision trade-offs.

## Architecture

### Core Changes

The implementation modifies the following file:
- `third_party/compressed-tensors/src/compressed_tensors/quantization/quant_args.py`

Key additions:
1. **CUSTOM_FP8_E5M2_DATA class**: Defines the E5M2 format with custom casting function
2. **custom_format field**: Added to `QuantizationArgs` to specify format identifier
3. **Updated pytorch_dtype()**: Handles custom format selection
4. **Updated round_to_quantized_type_args()**: Routes to custom casting function

### Custom Format Identifier

The `custom_format` field accepts string identifiers like:
- `"e5m2"` - 5-bit exponent, 2-bit mantissa (implemented)
- `"e4m3"` - 4-bit exponent, 3-bit mantissa (default, can be explicit)

## Files

### Core Implementation
- `third_party/compressed-tensors/src/compressed_tensors/quantization/quant_args.py` - Modified core file

### Examples
- `custom_fp8_example.py` - Simple PTQ example using Python API
- `custom_fp8_recipe.yaml` - YAML recipe for quantization configuration

### Documentation
- `README.md` - This file

## Usage

### Quick Start

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure custom FP8 quantization
recipe = QuantizationModifier(
    targets="Linear",
    scheme={
        "input_activations": {
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "dynamic": True,
            "custom_format": "e5m2",  # Custom format
        },
        "weights": {
            "num_bits": 8,
            "type": "float",
            "strategy": "channel",
            "symmetric": True,
            "custom_format": "e5m2",  # Custom format
        },
    },
    ignore=["lm_head", "model.norm"],
)

# Apply quantization
oneshot(model=model, recipe=recipe)
```

### Running Examples

1. **Simple PTQ Example**:
   ```bash
   python custom_fp8_example.py
   ```
   This will quantize TinyLlama-1.1B using custom FP8 format.

2. **Using YAML Recipe**:
   ```bash
   # Modify the recipe file as needed
   # Then use with llmcompressor CLI or Python API
   ```

## Custom Operator Integration

The current implementation includes a placeholder casting function. To integrate your custom CUDA operator:

### Option 1: Using torch.ops

```python
@staticmethod
@torch.compile
def cast_to_custom_fp8(x):
    return torch.ops.custom_ops.fp8_e5m2_cast(x)
```

### Option 2: Using Python Binding

```python
import custom_fp8_ops

@staticmethod
@torch.compile
def cast_to_custom_fp8(x):
    return custom_fp8_ops.cast_to_e5m2(x)
```

### Option 3: Direct C++ Extension

```python
from custom_fp8_extension import fp8_e5m2_cast

@staticmethod
@torch.compile
def cast_to_custom_fp8(x):
    return fp8_e5m2_cast(x)
```

## Configuration Options

### Quantization Arguments

- `num_bits`: Number of bits (8 for FP8)
- `type`: "float" for floating-point quantization
- `strategy`: "tensor", "channel", "group", etc.
- `dynamic`: True for dynamic quantization, False for static
- `symmetric`: True for symmetric quantization
- `custom_format`: "e5m2" for custom E5M2 format

### Layers to Ignore

Typically ignore:
- Embedding layers
- Normalization layers (LayerNorm, RMSNorm)
- Activation functions
- Output projection (lm_head)

## Testing

### Unit Tests

```bash
# Run tests for quantization args
pytest tests/test_quantization/test_quant_args.py
```

### Integration Tests

```bash
# Test with a small model
python custom_fp8_example.py
```

### Validation

1. Check quantization config in saved model:
   ```python
   import json
   with open("model/config.json") as f:
       config = json.load(f)
   print(config["quantization_config"])
   ```

2. Verify model outputs:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("./quantized_model")
   # Test generation
   ```

## Deployment

### vLLM Integration

```python
from vllm import LLM

# Load quantized model
model = LLM("./quantized_model")

# Run inference
outputs = model.generate("Hello, world!", max_tokens=50)
```

### Evaluation

```bash
# Using lm-eval-harness
lm_eval --model hf \
    --model_args pretrained=./quantized_model \
    --tasks hellaswag,arc_easy,arc_challenge \
    --batch_size 8
```

## Performance Considerations

### E5M2 vs E4M3

| Format | Exponent | Mantissa | Range | Precision |
|--------|----------|----------|-------|-----------|
| E4M3   | 4 bits   | 3 bits   | ±448  | Higher    |
| E5M2   | 5 bits   | 2 bits   | ±57344| Lower     |

**E5M2 Advantages**:
- Larger dynamic range
- Better for values with wide distribution

**E4M3 Advantages**:
- Higher precision
- Better for values requiring fine granularity

## Troubleshooting

### Issue: Custom format not recognized

**Solution**: Ensure `custom_format` field is properly set in quantization config:
```python
scheme={
    "weights": {
        "custom_format": "e5m2",  # Must be set
        ...
    }
}
```

### Issue: Quantization fails during forward pass

**Solution**: Check that custom casting function is properly implemented and handles edge cases:
```python
def cast_to_custom_fp8(x):
    # Add proper error handling
    if x.numel() == 0:
        return x
    # Your casting logic
    ...
```

### Issue: Model accuracy degradation

**Solutions**:
1. Use calibration data for better quantization parameters
2. Adjust which layers to quantize (ignore more layers)
3. Try different quantization strategies (channel vs tensor)
4. Consider mixed precision (some layers in FP8, others in FP16)

## Future Enhancements

1. **Additional Formats**: Support for other custom FP8 formats (E3M4, E6M1, etc.)
2. **Mixed Precision**: Automatic selection of format per layer
3. **Calibration**: Advanced calibration strategies for custom formats
4. **Hardware Support**: Integration with custom hardware accelerators
5. **Benchmarking**: Comprehensive performance and accuracy benchmarks

## References

- [Compressed Tensors Documentation](https://github.com/vllm-project/compressed-tensors)
- [LLM Compressor Documentation](https://github.com/vllm-project/llm-compressor)
- [FP8 Formats Specification](https://arxiv.org/abs/2209.05433)

## Contributing

To add support for additional custom formats:

1. Define a new FloatArgs class (e.g., `CUSTOM_FP8_E3M4_DATA`)
2. Implement the casting function
3. Add format identifier to `custom_format` handling
4. Update documentation and examples
5. Add tests

## License

This implementation follows the same license as the parent project (Apache 2.0).

## Contact

For questions or issues, please open an issue in the repository.
