# NEORV32 + NPU — Linux on RISC-V with Hardware Neural Accelerator

NEORV32 RV32IMAC soft-core running **nommu Linux** with a **4×4 INT8 systolic array NPU** on the Heijin AX301 board (Altera Cyclone IV EP4CE6).

The NPU is memory-mapped on the Wishbone (XBUS) bus and accessible from Linux userspace via `/dev/npu` ioctl interface.

## Status

| Phase | Status |
|-------|--------|
| Phase 1: Bare-metal NEORV32 + TPU | ✅ 21/21 tests passed |
| Phase 2: Linux + /dev/npu driver  | ✅ 4/4 tests passed |
| Phase 2: MNIST inference on NPU   | ✅ 3/3 correct (labels 7, 2, 1) |

## Resource Usage (EP4CE6, 50 MHz)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| Logic Elements | 5,590 | 6,272 | 89% |
| Memory bits | 166,912 | 276,480 | 60% |
| DSP 9-bit elements | 12 | 30 | 40% |

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
└── wb_tpu_accel → tpu_accel → systolic_array_4x4 → 16× pe
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

The kernel driver (`kernel/neorv32_npu.c`) provides a misc char device with ioctl interface:

```c
#define NPU_LOAD_WEIGHTS  _IOW('N', 1, struct npu_weights)  /* 4×4 int8 matrix */
#define NPU_COMPUTE       _IOW('N', 2, struct npu_compute)  /* 4× int8 input   */
#define NPU_GET_RESULT    _IOR('N', 3, struct npu_result)   /* 4× int32 output */
#define NPU_CLEAR         _IO('N', 4)                       /* reset accumulators */
```

## MNIST Inference

3-layer INT8 quantized MLP (784 → 128 → 64 → 10) running on the NPU via `/dev/npu` ioctl. The `mnist` shell command performs tiled 4×4 matrix multiplication across all layers with bias, ReLU, and INT8 requantization.

Weights are generated from the `tpu_demo/` trained model by `sw/initramfs/gen_weights.py` and embedded in the initramfs binary.

```
npu# mnist

=== MNIST Inference (3-layer MLP on NPU) ===

Sample 0 (label=7): predicted=7 [CORRECT]
Sample 1 (label=2): predicted=2 [CORRECT]
Sample 2 (label=1): predicted=1 [CORRECT]

=== MNIST: 3/3 correct ===
```

## Build & Run

### Prerequisites

- Intel Quartus Prime Lite 21.1+
- xPack RISC-V GCC 14.2.0 (for kernel)
- Buildroot Linux GCC (for initramfs init)
- `see_neorv32_run_linux/` project (patched kernel source, stage2 loader)

### Quick Start (pre-built)

```bash
# Program FPGA
quartus_pgm -c usb-blaster -m jtag -o "p;quartus/neorv32_tpu.sof"

# Reset board (press KEY2), then boot Linux
cd ~/see_neorv32_run_linux
python3 host/boot_linux.py --port /dev/ttyUSB0 --skip-program

# At the npu# prompt:
npu     # runs 4 NPU hardware tests
mnist   # runs MNIST inference (3 samples)
```

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
2. **Stage2 loader** (IMEM, 115200 baud) → xmodem kernel+DTB+initramfs to SDRAM, CRC-32 verify
3. **Linux kernel** boots from SDRAM (0x40000000), ~132s to shell
4. **Mini shell** with `npu` (driver test) and `mnist` (inference) commands

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
│   │   ├── gen_weights.py  — Generates mnist_data.h from tpu_demo weights
│   │   └── mnist_data.h    — INT8 weights/biases/test samples (generated)
│   └── npu_test/           — Standalone userspace NPU test (libc)
├── quartus/                — Quartus project
├── sim/                    — Verilog testbenches
├── host/                   — Host-side boot script
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

1. **Stage2 loader must be built against this project's NEORV32** — the `neorv32/` subdir has a different commit than `see_neorv32_run_linux/neorv32/`. Using the wrong stage2 causes `ERROR_SIGNATURE` from the bootloader.
2. **Init must mount devtmpfs and proc** — `/dev/npu` is created by devtmpfs but only visible after mounting. `/proc/misc` is needed to find the dynamic minor number for mknod fallback.
3. **89% LE utilization** — D-cache disabled to save ~300 LEs. CPU_FAST_MUL_EN cannot be used due to EP4CE6 DSP placement limits (12 PE + 8 CPU MUL > physical capacity). Zicntr must stay enabled — stage2_loader uses `neorv32_cpu_get_cycle()` for timeouts.
4. **Timing slack -0.976 ns** — fails slow-corner STA but passes fast-corner (+0.640 ns). Same pattern as the Linux-only project (-0.583 ns) which works reliably on hardware.
5. **DSP placement** — 12 of 16 PEs use DSP blocks; 4 PEs in column 3 are forced to LUT multipliers due to EP4CE6 physical placement constraints.
