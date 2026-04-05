#!/usr/bin/env python3
"""Generate cnn_data.h from tpu_demo/cnn weights for embedding in init.c"""

import json
import numpy as np
import os

CNN_WEIGHTS_DIR = "/home/test/fpga/tpu_demo/cnn/weights"

def pad4(n):
    return ((n + 3) // 4) * 4

def arr_to_c(name, data, dtype, cols=16):
    lines = []
    if dtype == 'int8':
        type_str = "const int8_t"
    elif dtype == 'int32':
        type_str = "const int32_t"
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
    with open(os.path.join(CNN_WEIGHTS_DIR, "cnn_meta.json")) as f:
        meta = json.load(f)

    out = []
    out.append("/* Auto-generated CNN model data — do not edit */")
    out.append("#ifndef CNN_DATA_H")
    out.append("#define CNN_DATA_H")
    out.append("")

    layers = meta['layers']
    conv_info = meta['conv_info']
    conv_layers = [l for l in layers if l['type'] == 'conv']
    fc_layers = [l for l in layers if l['type'] == 'fc']

    out.append(f"#define CNN_NUM_CONV {len(conv_layers)}")
    out.append(f"#define CNN_NUM_FC {len(fc_layers)}")
    out.append(f"#define CNN_NUM_SAMPLES 10")
    out.append("")

    # Conv layer weights (padded to F_pad × K_pad)
    for i, lm in enumerate(conv_layers):
        F = lm['out_channels']
        K = lm['patch_size']
        K_pad = pad4(K)
        F_pad = pad4(F)

        W_raw = np.fromfile(os.path.join(CNN_WEIGHTS_DIR, lm['w_file']),
                           dtype=np.int8).reshape(F, K)
        W = np.zeros((F_pad, K_pad), dtype=np.int8)
        W[:F, :K] = W_raw

        b_raw = np.fromfile(os.path.join(CNN_WEIGHTS_DIR, lm['b_file']),
                           dtype=np.int32)
        b = np.zeros(F_pad, dtype=np.int32)
        b[:F] = b_raw

        ci = conv_info[i]
        out.append(f"/* Conv{i}: {ci['in_channels']}ch {ci['in_h']}x{ci['in_w']}"
                   f" -> {F}ch {ci['out_h']}x{ci['out_w']}"
                   f" (k={ci['kH']}x{ci['kW']}, K={K} pad={K_pad}) */")
        out.append(arr_to_c(f"cnn_conv{i}_w", W.flatten(), 'int8'))
        out.append(arr_to_c(f"cnn_conv{i}_b", b, 'int32'))
        out.append("")

    # FC layer weights (padded to out_pad × in_pad)
    for i, lm in enumerate(fc_layers):
        in_dim = lm['in_features']
        out_dim = lm['out_features']
        in_pad = pad4(in_dim)
        out_pad = pad4(out_dim)

        W_raw = np.fromfile(os.path.join(CNN_WEIGHTS_DIR, lm['w_file']),
                           dtype=np.int8).reshape(out_dim, in_dim)
        W = np.zeros((out_pad, in_pad), dtype=np.int8)
        W[:out_dim, :in_dim] = W_raw

        b_raw = np.fromfile(os.path.join(CNN_WEIGHTS_DIR, lm['b_file']),
                           dtype=np.int32)
        b = np.zeros(out_pad, dtype=np.int32)
        b[:out_dim] = b_raw

        out.append(f"/* FC{i}: {in_dim}->{out_dim} (padded {in_pad}->{out_pad}) */")
        out.append(arr_to_c(f"cnn_fc{i}_w", W.flatten(), 'int8'))
        out.append(arr_to_c(f"cnn_fc{i}_b", b, 'int32'))
        out.append("")

    # Conv layer metadata
    out.append("typedef struct {")
    out.append("    int in_ch, out_ch, kH, kW, in_h, in_w, out_h, out_w, pool;")
    out.append("    int patch_size, patch_size_pad, out_ch_pad;")
    out.append("    int requant_mult;")
    out.append("} cnn_conv_t;")
    out.append("")
    out.append(f"static const cnn_conv_t cnn_conv_dims[CNN_NUM_CONV] = {{")
    for i, lm in enumerate(conv_layers):
        ci = conv_info[i]
        K = lm['patch_size']
        K_pad = pad4(K)
        F = lm['out_channels']
        F_pad = pad4(F)
        mult = lm['requant_mult']
        out.append(f"  {{ {ci['in_channels']}, {F}, {ci['kH']}, {ci['kW']}, "
                   f"{ci['in_h']}, {ci['in_w']}, {ci['out_h']}, {ci['out_w']}, "
                   f"{ci['pool']}, {K}, {K_pad}, {F_pad}, {mult} }},")
    out.append("};")
    out.append("")

    # FC layer metadata (same as MLP)
    out.append("typedef struct {")
    out.append("    int in_dim, out_dim, in_pad, out_pad;")
    out.append("    int has_relu, requant_mult;")
    out.append("} cnn_fc_t;")
    out.append("")
    out.append(f"static const cnn_fc_t cnn_fc_dims[CNN_NUM_FC] = {{")
    for i, lm in enumerate(fc_layers):
        in_dim = lm['in_features']
        out_dim = lm['out_features']
        in_pad = pad4(in_dim)
        out_pad = pad4(out_dim)
        has_relu = 1 if lm['activation'] == 'relu' else 0
        mult = lm['requant_mult']
        out.append(f"  {{ {in_dim}, {out_dim}, {in_pad}, {out_pad}, "
                   f"{has_relu}, {mult} }},")
    out.append("};")
    out.append("")

    # Accessor functions (PC-relative, works with PIE)
    out.append("static const int8_t *cnn_get_conv_w(int i) {")
    out.append("    switch (i) {")
    for i in range(len(conv_layers)):
        out.append(f"        case {i}: return cnn_conv{i}_w;")
    out.append("        default: return (const int8_t *)0;")
    out.append("    }")
    out.append("}")
    out.append("static const int32_t *cnn_get_conv_b(int i) {")
    out.append("    switch (i) {")
    for i in range(len(conv_layers)):
        out.append(f"        case {i}: return cnn_conv{i}_b;")
    out.append("        default: return (const int32_t *)0;")
    out.append("    }")
    out.append("}")
    out.append("static const int8_t *cnn_get_fc_w(int i) {")
    out.append("    switch (i) {")
    for i in range(len(fc_layers)):
        out.append(f"        case {i}: return cnn_fc{i}_w;")
    out.append("        default: return (const int8_t *)0;")
    out.append("    }")
    out.append("}")
    out.append("static const int32_t *cnn_get_fc_b(int i) {")
    out.append("    switch (i) {")
    for i in range(len(fc_layers)):
        out.append(f"        case {i}: return cnn_fc{i}_b;")
    out.append("        default: return (const int32_t *)0;")
    out.append("    }")
    out.append("}")
    out.append("")

    # Test samples (int8, already [0,127])
    samples = np.fromfile(os.path.join(CNN_WEIGHTS_DIR, "test_samples.bin"),
                         dtype=np.int8).reshape(10, 784)
    labels = np.fromfile(os.path.join(CNN_WEIGHTS_DIR, "test_labels.bin"),
                        dtype=np.uint8)

    out.append(f"/* {len(labels)} test samples, 784 pixels each (int8 0-127) */")
    out.append(arr_to_c("cnn_test_labels", labels, 'uint8'))
    out.append("")

    n_samples = 10
    for s in range(n_samples):
        out.append(arr_to_c(f"cnn_test_sample_{s}", samples[s], 'int8'))
    out.append("")
    out.append("static const int8_t *cnn_get_test_sample(int idx) {")
    out.append("    switch (idx) {")
    for s in range(n_samples):
        out.append(f"        case {s}: return cnn_test_sample_{s};")
    out.append("        default: return (const int8_t *)0;")
    out.append("    }")
    out.append("}")
    out.append("")

    out.append("#endif /* CNN_DATA_H */")

    header = "\n".join(out)
    outpath = os.path.join(os.path.dirname(__file__), "cnn_data.h")
    with open(outpath, 'w') as f:
        f.write(header)

    # Stats
    total = 0
    for i, lm in enumerate(conv_layers):
        K_pad = pad4(lm['patch_size'])
        F_pad = pad4(lm['out_channels'])
        w_size = F_pad * K_pad
        b_size = F_pad * 4
        total += w_size + b_size
        print(f"  Conv{i}: weights={w_size} bias={b_size}")
    for i, lm in enumerate(fc_layers):
        in_pad = pad4(lm['in_features'])
        out_pad = pad4(lm['out_features'])
        w_size = out_pad * in_pad
        b_size = out_pad * 4
        total += w_size + b_size
        print(f"  FC{i}: weights={w_size} bias={b_size}")
    total += n_samples * 784
    print(f"  Test samples: {n_samples} x 784 = {n_samples * 784}")
    print(f"  Total data: {total} bytes")
    print(f"  Written to: {outpath}")

if __name__ == "__main__":
    main()
