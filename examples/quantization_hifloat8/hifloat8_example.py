"""
HiFloat8 Quantization Example - Simple PTQ Approach
=====================================================

This example demonstrates how to quantize a model using custom HiFloat8 format
with the Python API (no YAML recipe required).

Usage:
    python custom_fp8_example.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
MODEL_ID = "/data0/models/QwQ-32B"  # Using a small model for testing
OUTPUT_DIR = "/data0/models/QwQ-32B-hifloat8-quantized"

# Quantization configuration
CUSTOM_FORMAT = "hifloat8"  # Your custom HiFloat8 format identifier
QUANTIZE_WEIGHTS = True
QUANTIZE_ACTIVATIONS = True

# Device configuration
DEVICE = "npu:0" if torch.npu.is_available() else "cpu"

# ============================================================================
# STEP 1: LOAD MODEL AND TOKENIZER
# ============================================================================

print("=" * 80)
print("STEP 1: Loading Model and Tokenizer")
print("=" * 80)

# Load the model in the appropriate dtype
# For FP8 quantization, we typically start with bfloat16 or float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # or "auto"
    device_map=DEVICE,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print(f"✓ Model loaded: {MODEL_ID}")
print(f"✓ Model dtype: {model.dtype}")
print(f"✓ Model device: {next(model.parameters()).device}")

# ============================================================================
# STEP 2: CONFIGURE HIFLOAT8 QUANTIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Configuring HiFloat8 Quantization")
print("=" * 80)

# Define which layers to ignore
IGNORE_LAYERS = [
    "lm_head",              # Output projection
]

# Create the quantization recipe using Python API
# This is equivalent to the YAML recipe but more flexible
recipe = QuantizationModifier(
    targets="Linear",  # Target all Linear layers
    scheme={
        "input_activations": {
            "num_bits": 8,
            "type": "float",            # quantize to low-precision float
            "strategy": "token",        # Per-token quantization, if supported
            "dynamic": True,            # Dynamic quantization (computed at runtime)
            "custom_format": CUSTOM_FORMAT,  # Your custom format
        } if QUANTIZE_ACTIVATIONS else None,
        "weights": {
            "num_bits": 8,
            "type": "float",            # quantize to low-precision float
            "strategy": "channel",     # Per-channel quantization for weights
            "symmetric": True,         # Symmetric quantization
            "custom_format": CUSTOM_FORMAT,  # Your custom format
        } if QUANTIZE_WEIGHTS else None,
    },
    ignore=IGNORE_LAYERS,
)

print(f"✓ Quantization scheme configured:")
print(f"  - Target layers: Linear")
print(f"  - Custom format: {CUSTOM_FORMAT}")
print(f"  - Weight quantization: {QUANTIZE_WEIGHTS}")
print(f"  - Activation quantization: {QUANTIZE_ACTIVATIONS}")
print(f"  - Ignored layers: {IGNORE_LAYERS}")

# ============================================================================
# STEP 3: APPLY QUANTIZATION (PTQ - No Calibration Data Needed)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Applying Quantization")
print("=" * 80)

# For simple PTQ with HiFloat8, we don't need calibration data
# The quantization is applied directly to the weights
oneshot(
    model=model,
    recipe=recipe,
)

print("✓ Quantization applied successfully")

# ============================================================================
# STEP 4: TEST QUANTIZED MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Testing Quantized Model")
print("=" * 80)

# Prepare the model for generation
dispatch_for_generation(model)

# Test prompts
test_prompts = [
    "Hello, my name is",
    "The capital of France is",
    "In machine learning, quantization refers to",
]

print("\nGenerating sample outputs:\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"Prompt {i}: {prompt}")
    
    # Tokenize input
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(model.device)
    
    # Generate output
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,  # Greedy decoding for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and print
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Output: {generated_text}\n")
    print("-" * 80 + "\n")

# ============================================================================
# STEP 5: SAVE QUANTIZED MODEL
# ============================================================================

print("=" * 80)
print("STEP 5: Saving Quantized Model")
print("=" * 80)

# Save the quantized model in compressed-tensors format
model.save_pretrained(
    OUTPUT_DIR,
    save_compressed=True,  # Save in compressed format
)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✓ Model saved to: {OUTPUT_DIR}")
print(f"✓ Model is ready for deployment with vLLM or other inference engines")

# ============================================================================
# STEP 6: VERIFY SAVED MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Verifying Saved Model")
print("=" * 80)

# Check the saved files
import os
saved_files = os.listdir(OUTPUT_DIR)
print(f"✓ Saved files: {', '.join(saved_files)}")

# Check quantization config
config_path = os.path.join(OUTPUT_DIR, "config.json")
if os.path.exists(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if "quantization_config" in config:
        print(f"✓ Quantization config found:")
        print(f"  {json.dumps(config['quantization_config'], indent=2)}")
    else:
        print("⚠ Warning: No quantization_config found in config.json")

print("\n" + "=" * 80)
print("QUANTIZATION COMPLETE!")
print("=" * 80)
print(f"\nNext steps:")
print(f"1. Load the model in vLLM:")
print(f"   from vllm import LLM")
print(f"   model = LLM('{OUTPUT_DIR}')")
print(f"2. Run inference:")
print(f"   outputs = model.generate('Hello, world!')")
print(f"3. Evaluate accuracy with lm-eval-harness")

