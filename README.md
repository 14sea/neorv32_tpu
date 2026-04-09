# NEORV32 + NPU — Linux on RISC-V with Hardware Neural Accelerator

NEORV32 RV32IMAC soft-core running **nommu Linux** with a **4×4 INT8 systolic array NPU** on the Heijin AX301 board (Altera Cyclone IV EP4CE6).

The NPU is memory-mapped on the Wishbone (XBUS) bus and accessible from Linux userspace via `/dev/npu` ioctl interface.

## Status

| Phase | Status |
|-------|--------|
| Phase 1: Bare-metal NEORV32 + TPU | ✅ 21/21 tests passed |
| Phase 2: Linux + /dev/npu driver  | ✅ 4/4 tests passed |
| Phase 2: MNIST inference on NPU   | ✅ 10/10 correct (452 ms/sample via mmap) |
| Phase 2: CNN inference on NPU     | ✅ 10/10 correct (986 ms/sample via mmap) |

## Resource Usage (EP4CE6, 50 MHz)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| Logic Elements | 5,431 | 6,272 | 87% |
| Memory bits | 166,912 | 276,480 | 60% |
| DSP 9-bit elements | 15 | 30 | 50% |

## Architecture

```
ax301_top.vhd
├── neorv32_top (RV32IMAC, U-mode, Zaamo, Zalrsc, I-cache)
│   ├── IMEM 8 KB (stage2 bootloader)
│   ├── DMEM 8 KB
│   ├── UART0 (19200 bootloader / 115200 app)
│   ├── GPIO (4-bit LEDs)
│   └── CLINT (timer)
├── wb_sdram_ctrl → sdram_ctrl → 32 MB SDRAM
├── wb_tpu_accel → tpu_accel → systolic_array_4x4 → 16× pe
└── neorv32 SPI → SD card (boot-time only, read-only bulk storage)
```

### Memory Map

| Range | Device |
|-------|--------|
| `0x00000000` | 8 KB IMEM (M9K BRAM) |
| `0x40000000` | 32 MB SDRAM (kernel + data) |
| `0x80000000` | 8 KB DMEM (M9K BRAM) |
| `0xF0000000` | TPU accelerator (64 bytes) |
| `0xFFF40000` | CLINT timer |
| `0xFFF50000` | UART0 |

### TPU Register Map (0xF0000000)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x00 | CTRL | W | [0]=start compute, [4]=clear accumulators |
| 0x04 | STATUS | R | [0]=done |
| 0x08 | W_ADDR | W | [1:0]=col, [3:2]=row |
| 0x0C | W_DATA | W | [7:0]=int8 weight (triggers load) |
| 0x10 | X_IN | W | [31:0]={x3,x2,x1,x0} packed int8 |
| 0x14 | W_DATA4 | W | Bulk load 4 weights for one row |
| 0x20–0x2C | RES0–3 | R | int32 row accumulators |

## Linux /dev/npu Driver

The kernel driver (`kernel/neorv32_npu.c`) provides a misc char device with ioctl and mmap interface:

```c
#define NPU_LOAD_WEIGHTS  _IOW('N', 1, struct npu_weights)  /* 4×4 int8 matrix */
#define NPU_COMPUTE       _IOW('N', 2, struct npu_compute)  /* 4× int8 input   */
#define NPU_GET_RESULT    _IOR('N', 3, struct npu_result)   /* 4× int32 output */
#define NPU_CLEAR         _IO('N', 4)                       /* reset accumulators */
```

The driver also supports `mmap()` for direct userspace register access, bypassing syscall overhead. On nommu Linux, the TPU physical address (0xF0000000) is mapped directly into userspace via `NOMMU_MAP_DIRECT`. The MNIST inference uses the mmap fast path, eliminating ~13,784 ioctl syscalls per sample.

### Performance: mmap vs ioctl

| Mode | Per sample | Speedup |
|------|-----------|---------|
| mmap (direct register access) | 704 ms | **26.7×** |
| ioctl (syscall per operation) | 18,841 ms | 1× |

The `bench` shell command runs MNIST inference with both paths for direct comparison.

## CNN Inference

im2col CNN for MNIST running entirely on the NPU via mmap'd registers. 4-layer network: two conv+pool layers (im2col → tiled 4×4 matmul on NPU) followed by two FC layers (same tiled matmul as MLP).

```
Input: 28×28×1 int8 [0,127]
Conv(1→4, 5×5, valid) + ReLU + MaxPool(2×2) → 12×12×4   [576 patches × 7 tiles on NPU]
Conv(4→8, 3×3, valid) + ReLU + MaxPool(2×2) →  5×5×8    [100 patches × 9 tiles on NPU]
FC(200→64) + ReLU                                         [50 tile groups on NPU]
FC(64→10)                                                  [48 tile groups on NPU]
```

~6,698 NPU operations per sample (vs ~866 for MLP). Weights generated from `tpu_demo/cnn/` trained model by `sw/initramfs/gen_cnn_weights.py` (~22 KB embedded in initramfs).

```
npu# cnn

=== CNN Inference (im2col on NPU) ===
  Conv(1->4, 5x5)+Pool -> Conv(4->8, 3x3)+Pool -> FC(200->64) -> FC(64->10)
  mode: mmap

Sample 0 (label=7): predicted=7 [CORRECT] (983.7 ms)
Sample 1 (label=4): predicted=4 [CORRECT] (988.9 ms)
Sample 2 (label=5): predicted=5 [CORRECT] (979.0 ms)
Sample 3 (label=1): predicted=1 [CORRECT] (991.7 ms)
Sample 4 (label=8): predicted=8 [CORRECT] (980.8 ms)
Sample 5 (label=1): predicted=1 [CORRECT] (992.3 ms)
Sample 6 (label=5): predicted=5 [CORRECT] (995.4 ms)
Sample 7 (label=8): predicted=8 [CORRECT] (979.2 ms)
Sample 8 (label=7): predicted=7 [CORRECT] (983.4 ms)
Sample 9 (label=4): predicted=4 [CORRECT] (987.7 ms)

=== CNN: 10/10 correct, total 9862.5 ms (986.2 ms/sample) ===
```

### CNN vs MLP Performance

| Model | Accuracy | Per sample | NPU ops/sample |
|-------|----------|-----------|-----------------|
| MLP (784→128→64→10) | 10/10 | 452 ms | ~866 |
| CNN (2 conv + 2 FC) | 10/10 | 986 ms | ~6,698 |

CNN is 2.2× slower due to im2col conv layers requiring ~5,832 additional NPU operations per sample. However, CNN achieves higher accuracy on larger test sets (99.2% vs ~97% for MLP).

## MNIST Inference

3-layer INT8 quantized MLP (784 → 128 → 64 → 10) running on the NPU via mmap'd registers. The `mnist` shell command performs tiled 4×4 matrix multiplication across all layers with bias, ReLU, and INT8 requantization. Weight loading, compute, and result readback are done via direct register writes/reads (no syscall per operation).

Weights are generated from the `tpu_demo/` trained model by `sw/initramfs/gen_weights.py` and embedded in the initramfs binary.

```
npu# mnist

=== MNIST Inference (3-layer MLP on NPU) ===
  mode: mmap

Sample 0 (label=7): predicted=7 [CORRECT] (428.2 ms)
Sample 1 (label=2): predicted=2 [CORRECT] (479.2 ms)
Sample 2 (label=1): predicted=1 [CORRECT] (430.9 ms)
Sample 3 (label=0): predicted=0 [CORRECT] (428.2 ms)
Sample 4 (label=4): predicted=4 [CORRECT] (495.3 ms)
Sample 5 (label=1): predicted=1 [CORRECT] (428.1 ms)
Sample 6 (label=4): predicted=4 [CORRECT] (483.2 ms)
Sample 7 (label=9): predicted=9 [CORRECT] (428.3 ms)
Sample 8 (label=5): predicted=5 [CORRECT] (428.1 ms)
Sample 9 (label=9): predicted=9 [CORRECT] (495.3 ms)

=== MNIST: 10/10 correct, total 4525.3 ms (452.5 ms/sample) ===
```

## Build & Run

### Prerequisites

- Intel Quartus Prime Lite 21.1+
- xPack RISC-V GCC 14.2.0 (for kernel + stage2)
- Buildroot Linux GCC (for initramfs init)
- `openFPGALoader` with EP4CE6 support (built from source recommended)
- `linux-6.6.83.tar.xz` placed at repo root (download from https://cdn.kernel.org/pub/linux/kernel/v6.x/ — not committed)

### Quick Start (pre-built)

Two boot paths are supported. Both use the same FPGA + Linux image; the only difference is how the kernel reaches SDRAM.

**Path A — no SD card (UART xmodem, default, works on any AX301):**

```bash
python3 host/boot_linux.py --port /dev/ttyUSB0   # ~243 s to shell
```

**Path B — SD card fast boot (requires SD card wired to on-board slot):**

```bash
# One-time: pack Image + DTB + initramfs and stream-write to SD
python3 host/sd_pack.py --port /dev/ttyUSB0      # ~108 s @ 230400 baud

# Every boot: stage2 reads blob from SD into SDRAM, jumps to kernel
python3 host/boot_sd.py  --port /dev/ttyUSB0     # ~150 s to shell
```

At the `npu#` prompt:

```text
npu     # runs 4 NPU hardware tests
mnist   # runs MNIST MLP inference (10 samples)
cnn     # runs CNN inference with im2col (10 samples)
bench   # benchmarks mmap vs ioctl NPU access
```

## Fast Boot from SD Card (optional)

The stage2 loader can read a packed kernel blob directly from an SD card over NEORV32's hardware SPI peripheral, skipping the ~145 s UART xmodem transfer. **Linux still runs from SDRAM** — the SD card is only read-only bulk storage at boot, so no kernel driver is involved. Ported from [see_neorv32_run_linux](https://github.com/14sea/see_neorv32_run_linux), including the Phase 1–6 speed-ups (230400 baud UART, persistent stage2, `--update` one-shot, build-tag check, parametric dump).

**Wiring** (AX301 on-board SD slot): `PIN_J15=SD_CLK`, `PIN_K16=SD_DI (MOSI)`, `PIN_J16=SD_DO (MISO)`, `PIN_K15=SD_NCS`. Requires `IO_SPI_EN=true` in `rtl/ax301_top.vhd` (already set).

**Blob layout** (fixed LBA slots, see `host/sd_layout.py`):

```
LBA 0            header (NEOLNX magic + sizes + LBAs + layout_version)
LBA 1..4000      Image   (reserve 2 MB)
LBA 4001..4008   DTB     (reserve 4 KB)
LBA 4009..8008   initrd  (reserve 2 MB)
```

**Decoupled kernel / initramfs**: `board/linux_defconfig` uses `CONFIG_INITRAMFS_SOURCE=""` — the Image does **not** embed initramfs. Stage2 loads Image / DTB / initramfs as three independent sections from the SD blob, then patches the DTB's `chosen/linux,initrd-end` sentinel (`0xC0DEDEAD`) in RAM with the real end address before jumping to the kernel. This lets you iterate on `/init` (22 s cycle) without rebuilding or re-flashing the kernel.

**Iteration loop** (edit `sw/initramfs/init.c` → test):

```bash
make -C sw/initramfs LINUX_DIR=../../linux-6.6.83
cp sw/initramfs/neo_initramfs.cpio.gz output/
python3 host/boot_sd.py --update                 # ~17 s update + boot
```

**Host tools** (all under `host/`):

| Tool | Purpose |
|------|---------|
| `boot_linux.py` | UART xmodem boot (no SD needed) |
| `sd_pack.py`    | One-time full pack + write (header + Image + DTB + initrd) |
| `sd_update.py`  | Incremental update (typically 7 sectors / ~10 s for init-only) |
| `boot_sd.py`    | Boot from on-card blob, with build-tag check |
| `sd_dump.py`    | Parametric SD block reader for debugging |
| `sd_proto.py`   | Shared FPGA program + stage2 upload + baud switching |
| `sd_layout.py`  | Single source of truth for slot LBAs/sizes |

**Stage2 loader** fits in 8 KB IMEM with full SD support (~7832 / 8192 B). UART command modes:

| Cmd | Mode |
|-----|------|
| `l` | xmodem Linux boot (used by `boot_linux.py`) |
| `b` | Boot from SD blob (used by `boot_sd.py`) |
| `W` | Multi-segment SD write with per-block `K` ACK |
| `R` | Read on-card header for verify / build-tag check |
| `B` | Bump UART baud (115200 → 230400) |
| `d` | Parametric SD dump (cap 4096 sectors per call) |
| `s`/`w` | SD smoke / single-block write test |

### Full Build

```bash
# 1. Build initramfs
cd sw/initramfs && make && cd ../..

# 2. Inject drivers and build kernel
./build_linux.sh

# 3. Compile FPGA (if RTL changed)
cd quartus
quartus_sh --flow compile neorv32_tpu
quartus_cpf -c -o bitstream_compression=off output_files/neorv32_tpu.sof ../neorv32_tpu.rbf
```

### Simulation (TPU only)

```bash
cd sim
iverilog -o tb_tpu_accel tb_tpu_accel.v ../rtl/tpu_accel.v ../rtl/systolic_array_4x4.v ../rtl/pe.v
vvp tb_tpu_accel   # 16/16 tests pass
```

## Boot Sequence

1. **NEORV32 bootloader** (ROM, 19200 baud) → uploads stage2_loader
2. **Stage2 loader** (IMEM, 115200 baud) → dispatches on UART command:
   - `l` → xmodem kernel + DTB + initramfs to SDRAM, CRC-32 verify
   - `b` → read NEOLNX blob from SD into SDRAM, patch DTB initrd sentinel
3. **Linux kernel** boots from SDRAM (0x40000000), ~36 s (SD path) or ~132 s (xmodem path) to shell
4. **Mini shell** with `npu` / `mnist` / `cnn` / `bench` commands

## Directory Structure

```
neorv32_tpu/
├── rtl/                    — Verilog/VHDL RTL
│   ├── ax301_top.vhd      — Top-level SoC (NEORV32 + SDRAM + TPU)
│   ├── wb_tpu_accel.v     — Wishbone → TPU bridge
│   ├── tpu_accel.v        — Register-mapped systolic array wrapper
│   ├── systolic_array_4x4.v — 4×4 PE array
│   ├── pe.v               — Weight-stationary MAC PE
│   ├── wb_sdram_ctrl.v    — Wishbone → SDRAM bridge
│   └── sdram_ctrl.v       — SDRAM controller
├── kernel/                 — Linux kernel driver
│   ├── neorv32_npu.c      — /dev/npu misc device driver
│   ├── neorv32_npu.h      — ioctl header (kernel + userspace)
│   └── inject_npu_driver.sh — Injects driver into kernel tree
├── board/                  — Board support
│   ├── neorv32_tpu.dts    — Device tree (CPU + SDRAM + UART + NPU)
│   └── linux_defconfig    — Kernel config with CONFIG_NEORV32_NPU=y
├── sw/
│   ├── tpu_test/           — Phase 1 bare-metal test firmware
│   ├── stage2_loader/      — Bootloader stage2 (xmodem receiver)
│   ├── initramfs/          — Linux init with npu + mnist commands
│   │   ├── gen_weights.py  — Generates mnist_data.h from tpu_demo MLP weights
│   │   ├── gen_cnn_weights.py — Generates cnn_data.h from tpu_demo CNN weights
│   │   ├── mnist_data.h    — INT8 MLP weights/biases/test samples (generated)
│   │   └── cnn_data.h      — INT8 CNN weights/biases/test samples (generated)
│   └── npu_test/           — Standalone userspace NPU test (libc)
├── quartus/                — Quartus project
├── sim/                    — Verilog testbenches
├── host/                   — Host-side boot + SD tools (boot_linux.py, boot_sd.py, sd_*.py)
├── output/                 — Build outputs (Image, DTB, stage2, initramfs)
├── neorv32_tpu.rbf         — Pre-built FPGA bitstream
└── build_linux.sh          — Full build script
```

## Hardware Test Output

```
========================================
 NEORV32 + NPU — Linux mini shell
========================================
Linux (none) 6.6.83 #3 Sat Apr  4 14:56:06 BST 2026 riscv32
Uptime:    132 s
Total RAM: 31004 KB
Free RAM:  30240 KB
Processes: 15

Type 'help' for commands. 'npu' to test.

npu# npu

=== NPU Test (4x4 systolic array at 0xF0000000) ===

Test 1: Identity matrix
  PASS: I * [10,20,30,40]
Test 2: General 4x4 matmul
  PASS: W * [1,2,3,4] = [30,70,110,150]
Test 3: Signed values
  PASS: signed W * [1,-1,2,-2]
Test 4: Accumulation (two MACs)
  PASS: I*[10..40] + I*[5,5,5,5] = [15,25,35,45]

=== NPU Results: 4/4 ALL PASSED ===
```

## Known Pitfalls

1. **Stage2 loader must be built against this project's NEORV32 submodule** — `sw/stage2_loader/` links against `neorv32/sw/common/` HAL. Using a stage2 built against a different NEORV32 version may cause `ERROR_SIGNATURE` from the bootloader.
2. **Init must mount devtmpfs and proc** — `/dev/npu` is created by devtmpfs but only visible after mounting. `/proc/misc` is needed to find the dynamic minor number for mknod fallback.
3. **85% LE utilization** — D-cache disabled (~295 LEs saved) and 15 of 16 PE multipliers moved to DSP via `(* multstyle = "dsp" *)` (~272 LEs saved). Only `pe_2_3` remains on LUT due to EP4CE6 physical placement limit. CPU_FAST_MUL_EN cannot be used (would exceed DSP capacity). Zicntr must stay enabled — stage2_loader uses `neorv32_cpu_get_cycle()`.
4. **Timing slack -0.976 ns** — fails slow-corner STA but passes fast-corner (+0.640 ns). Same pattern as the Linux-only project (-0.583 ns) which works reliably on hardware.
5. **DSP placement** — 15 of 16 PEs use DSP blocks; only `pe_2_3` is forced to LUT due to EP4CE6 physical placement constraints. The `(* multstyle = "dsp" *)` Verilog attribute forces synthesis to use DSP; without it Quartus falls back to LUT for some PEs.
6. **nommu mmap requires 3 things** — (a) `mmap_capabilities` returning `NOMMU_MAP_DIRECT`, (b) `get_unmapped_area` returning the physical address, (c) the mmap callback must not fail the nommu `remap_pfn_range` check (`addr == pfn << PAGE_SHIFT`). The driver stores `phys_addr` at probe time rather than navigating `misc.this_device->parent` (which doesn't point to the platform device).
7. **mmap return value on RV32** — `0xF0000000` is negative as `signed long` on 32-bit. Userspace must use `unsigned long` and check for kernel error codes (`>= 0xFFFFF000`) instead of `< 0`.
