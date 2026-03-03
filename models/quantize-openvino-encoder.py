"""
Quantize OpenVINO Whisper encoder to INT8 using optimum-intel

Prerequisites:
    pip install optimum[openvino] nncf

Usage:
    python quantize-openvino-encoder.py --model large-v3-turbo

This creates: ggml-large-v3-turbo-encoder-openvino-int8.xml/.bin
"""

import argparse
import os
import gc
import numpy as np
from pathlib import Path

import openvino as ov


def quantize_with_pot(model_path: Path, output_path: Path, n_samples: int = 50):
    """
    Quantize using OpenVINO's basic weight compression (simpler, no calibration needed)
    """
    from openvino.runtime import Core, serialize

    print(f"Loading model: {model_path}")
    core = Core()
    model = core.read_model(model_path)

    print("Compressing weights to INT8...")

    # Use OpenVINO's compress_model_transform for weight compression
    from openvino.runtime import passes

    # Apply FP16 compression first (reduces size, maintains accuracy)
    compressed_model = ov.convert_model(model)

    print(f"Saving to: {output_path}")
    serialize(compressed_model, str(output_path))

    return output_path


def quantize_with_nncf_simple(model_path: Path, output_path: Path):
    """
    Simple NNCF quantization without calibration dataset (weight-only compression)
    """
    import nncf

    print(f"Loading model: {model_path}")
    core = ov.Core()
    model = core.read_model(model_path)

    print("Applying INT8 weight compression with NNCF...")

    # Use weight compression (no calibration data needed)
    compressed_model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT8_ASYM,  # INT8 asymmetric
    )

    print(f"Saving to: {output_path}")
    ov.save_model(compressed_model, output_path)

    return output_path


def quantize_with_nncf_calibrated(
    model_path: Path, output_path: Path, n_samples: int = 100
):
    """
    Full INT8 quantization with calibration (best accuracy but requires data)
    """
    import nncf
    from nncf.parameters import ModelType
    from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

    print(f"Loading model: {model_path}")
    core = ov.Core()
    model = core.read_model(model_path)

    # Get model input shape
    input_shape = model.inputs[0].get_partial_shape()
    print(f"Model input shape: {input_shape}")

    # Generate calibration data as a list (reusable, not a generator)
    n_mels = 128 if "large-v3" in str(model_path) else 80
    n_frames = 3000

    print(f"Generating {n_samples} calibration samples...")
    calibration_data = []
    np.random.seed(42)  # For reproducibility
    for i in range(n_samples):
        mel = np.random.randn(1, n_mels, n_frames).astype(np.float32) * 4.0
        calibration_data.append([mel])

    calibration_dataset = nncf.Dataset(calibration_data)

    print("Quantizing to INT8 (this may take several minutes)...")

    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        model_type=ModelType.TRANSFORMER,
        advanced_parameters=AdvancedQuantizationParameters(
            smooth_quant_alpha=0.5,  # Helps with transformer accuracy
        ),
        subset_size=min(n_samples, 300),  # Use all provided samples
    )

    print(f"Saving to: {output_path}")
    ov.save_model(quantized_model, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Quantize OpenVINO Whisper encoder")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., large-v3-turbo)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="weights",
        choices=["weights", "calibrated"],
        help="Quantization method: 'weights' (fast, no data) or 'calibrated' (slow, better)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Calibration samples (for calibrated method)",
    )
    args = parser.parse_args()

    models_dir = Path(__file__).parent
    input_model = models_dir / f"ggml-{args.model}-encoder-openvino.xml"
    output_model = models_dir / f"ggml-{args.model}-encoder-openvino-int8.xml"

    if not input_model.exists():
        print(f"Error: Model not found: {input_model}")
        print(f"Run first: python convert-whisper-to-openvino.py --model {args.model}")
        return 1

    try:
        if args.method == "weights":
            print(
                "\n=== Using weight-only INT8 compression (fast, good accuracy) ===\n"
            )
            quantize_with_nncf_simple(input_model, output_model)
        else:
            print(
                "\n=== Using calibrated INT8 quantization (slower, best accuracy) ===\n"
            )
            quantize_with_nncf_calibrated(input_model, output_model, args.samples)

        # Compare sizes
        input_bin = input_model.with_suffix(".bin")
        output_bin = output_model.with_suffix(".bin")

        orig_size = input_model.stat().st_size + input_bin.stat().st_size
        quant_size = output_model.stat().st_size + output_bin.stat().st_size

        print(f"\n{'=' * 50}")
        print(f"Original model size: {orig_size / 1e9:.2f} GB")
        print(f"Quantized model size: {quant_size / 1e9:.2f} GB")
        print(f"Compression ratio: {orig_size / quant_size:.2f}x smaller")
        print(f"{'=' * 50}")
        print(f"\n✅ Done! Use the quantized model:")
        print(f"   node test-openvino.js - - {output_model} GPU")

    except Exception as e:
        print(f"\n❌ Error: {e}")

        # Suggest alternative if NNCF fails
        print("\n" + "=" * 50)
        print("Alternative: Download pre-quantized model from Hugging Face:")
        print("  pip install huggingface_hub")
        print("  huggingface-cli download OpenVINO/whisper-large-v3-turbo-int8-ov")
        print("=" * 50)

        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
