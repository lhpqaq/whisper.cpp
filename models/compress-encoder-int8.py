"""Simple INT8 weight compression for OpenVINO encoder"""
import openvino as ov
import nncf
from pathlib import Path

model_path = Path(__file__).parent / 'ggml-large-v3-turbo-encoder-openvino.xml'
output_path = Path(__file__).parent / 'ggml-large-v3-turbo-encoder-openvino-int8.xml'

print(f'Loading model: {model_path}')
core = ov.Core()
model = core.read_model(str(model_path))

print('Applying INT8 weight compression...')
compressed_model = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT8_ASYM,
)

print(f'Saving to: {output_path}')
ov.save_model(compressed_model, str(output_path))

# Compare sizes
orig_size = model_path.stat().st_size + model_path.with_suffix('.bin').stat().st_size
quant_size = output_path.stat().st_size + output_path.with_suffix('.bin').stat().st_size
print(f'Original: {orig_size/1e9:.2f} GB')
print(f'Quantized: {quant_size/1e9:.2f} GB') 
print(f'Compression: {orig_size/quant_size:.1f}x smaller')
print('Done!')
