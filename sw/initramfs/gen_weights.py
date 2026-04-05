#!/usr/bin/env python3
"""Generate mnist_data.h from tpu_demo weights for embedding in init.c"""

import json
import numpy as np
import os
import sys

WEIGHTS_DIR = "/home/test/fpga/tpu_demo/weights"

def pad4(n):
    return ((n + 3) // 4) * 4

def arr_to_c(name, data, dtype, cols=16):
    """Convert numpy array to C initializer."""
    lines = []
    if dtype == 'int8':
        type_str = "const int8_t"
        fmt = lambda v: f"{int(v)}"
    elif dtype == 'int32':
        type_str = "const int32_t"
        fmt = lambda v: f"{int(v)}"
    elif dtype == 'uint8':
        type_str = "const uint8_t"
        fmt = lambda v: f"{int(v)}"

    lines.append(f"static {type_str} {name}[{len(data)}] = {{")
    for i in range(0, len(data), cols):
        chunk = data[i:i+cols]
        lines.append("  " + ", ".join(fmt(v) for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)

def main():
    with open(os.path.join(WEIGHTS_DIR, "model_meta.json")) as f:
        meta = json.load(f)

    out = []
    out.append("/* Auto-generated MNIST model data — do not edit */")
    out.append("#ifndef MNIST_DATA_H")
    out.append("#define MNIST_DATA_H")
    out.append("")

    # Number of layers
    out.append(f"#define NUM_LAYERS 3")
    out.append(f"#define NUM_SAMPLES 10")
    out.append("")

    # Layer metadata
    out.append("")

    for i, lm in enumerate(meta['layers']):
        in_dim = lm['in_features']
        out_dim = lm['out_features']
        in_pad = pad4(in_dim)
        out_pad = pad4(out_dim)

        # Load and pad weights
        W_raw = np.fromfile(os.path.join(WEIGHTS_DIR, lm['w_file']),
                           dtype=np.int8).reshape(out_dim, in_dim)
        W = np.zeros((out_pad, in_pad), dtype=np.int8)
        W[:out_dim, :in_dim] = W_raw

        # Load and pad bias
        b_raw = np.fromfile(os.path.join(WEIGHTS_DIR, lm['b_file']),
                           dtype=np.int32)
        b = np.zeros(out_pad, dtype=np.int32)
        b[:out_dim] = b_raw

        out.append(f"/* Layer {i}: {in_dim}→{out_dim} (padded {in_pad}→{out_pad}) */")
        out.append(arr_to_c(f"layer{i}_weights", W.flatten(), 'int8'))
        out.append(arr_to_c(f"layer{i}_bias", b, 'int32'))
        out.append("")

    # Layer metadata (no pointers — PIE binary has no CRT to relocate them)
    out.append("/* Layer dimensions — no pointers in static data (PIE relocation issue) */")
    out.append("typedef struct {")
    out.append("    int in_dim, out_dim, in_dim_pad, out_dim_pad;")
    out.append("    int has_relu, requant_mult;")
    out.append("} layer_dims_t;")
    out.append("")
    out.append("static const layer_dims_t layer_dims[NUM_LAYERS] = {")
    for i, lm in enumerate(meta['layers']):
        in_dim = lm['in_features']
        out_dim = lm['out_features']
        in_pad = pad4(in_dim)
        out_pad = pad4(out_dim)
        has_relu = 1 if lm['activation'] == 'relu' else 0
        requant_mult = min(int(round(lm['requant_scale'] * (1 << 16))), 65535)
        out.append(f"  {{ {in_dim}, {out_dim}, {in_pad}, {out_pad}, "
                   f"{has_relu}, {requant_mult} }},")
    out.append("};")
    out.append("")

    # Accessor functions (uses PC-relative addressing, works with PIE)
    out.append("static const int8_t *get_layer_weights(int layer) {")
    out.append("    switch (layer) {")
    for i in range(len(meta['layers'])):
        out.append(f"        case {i}: return layer{i}_weights;")
    out.append("        default: return (const int8_t *)0;")
    out.append("    }")
    out.append("}")
    out.append("")
    out.append("static const int32_t *get_layer_bias(int layer) {")
    out.append("    switch (layer) {")
    for i in range(len(meta['layers'])):
        out.append(f"        case {i}: return layer{i}_bias;")
    out.append("        default: return (const int32_t *)0;")
    out.append("    }")
    out.append("}")
    out.append("")

    # Test samples (uint8, 0-255 range — will be quantized to int8 0-127)
    samples = np.fromfile(os.path.join(WEIGHTS_DIR, "test_samples.bin"),
                         dtype=np.uint8).reshape(10, 784)
    labels = np.fromfile(os.path.join(WEIGHTS_DIR, "test_labels.bin"),
                        dtype=np.uint8)

    out.append(f"/* {len(labels)} test samples, 784 pixels each (uint8 0-255) */")
    out.append(arr_to_c("test_labels", labels, 'uint8'))
    out.append("")

    n_samples = 10
    out.append(f"#undef NUM_SAMPLES")
    out.append(f"#define NUM_SAMPLES {n_samples}")
    out.append(f"static const int test_sample_indices[NUM_SAMPLES] = {{{', '.join(str(i) for i in range(n_samples))}}};")
    for s in range(n_samples):
        out.append(arr_to_c(f"test_sample_{s}", samples[s], 'uint8'))
    out.append("")
    out.append("static const uint8_t *get_test_sample(int idx) {")
    out.append("    switch (idx) {")
    for s in range(n_samples):
        out.append(f"        case {s}: return test_sample_{s};")
    out.append("        default: return (const uint8_t *)0;")
    out.append("    }")
    out.append("}")
    out.append("")

    out.append("#endif /* MNIST_DATA_H */")

    header = "\n".join(out)

    outpath = os.path.join(os.path.dirname(__file__), "mnist_data.h")
    with open(outpath, 'w') as f:
        f.write(header)

    # Print stats
    total = 0
    for i, lm in enumerate(meta['layers']):
        in_pad = pad4(lm['in_features'])
        out_pad = pad4(lm['out_features'])
        w_size = in_pad * out_pad
        b_size = out_pad * 4
        total += w_size + b_size
        print(f"  Layer {i}: weights={w_size} bias={b_size}")
    total += n_samples * 784
    print(f"  Test samples: {n_samples} × 784 = {n_samples * 784}")
    print(f"  Total data: {total} bytes")
    print(f"  Written to: {outpath}")

if __name__ == "__main__":
    main()
