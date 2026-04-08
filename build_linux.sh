#!/bin/bash
# Build Linux + NPU kernel for NEORV32 TPU project (standalone)
set -e

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LINUX_SRC="$PROJ_DIR/linux-6.6.83"
TARBALL="$PROJ_DIR/linux-6.6.83.tar.xz"
OUTPUT_DIR="$PROJ_DIR/output"
XPACK_PATH="/home/test/xpack-riscv-none-elf-gcc-14.2.0-3/bin"

export PATH="$XPACK_PATH:$PATH"

mkdir -p "$OUTPUT_DIR"

echo "=== Step 0: Prepare kernel source ==="
if [ ! -d "$LINUX_SRC" ]; then
    if [ ! -f "$TARBALL" ]; then
        echo "ERROR: $TARBALL not found."
        echo "Download linux-6.6.83.tar.xz from https://cdn.kernel.org/pub/linux/kernel/v6.x/ into $PROJ_DIR/"
        exit 1
    fi
    echo "  Extracting $TARBALL ..."
    tar -C "$PROJ_DIR" -xf "$TARBALL"
    echo "  Applying nommu patches ..."
    ( cd "$LINUX_SRC" && patch -p1 < "$PROJ_DIR/kernel/neorv32_nommu.patch" )
else
    echo "  $LINUX_SRC already exists — skipping extract"
fi

echo ""
echo "=== Step 1: Build initramfs ==="
cd "$PROJ_DIR/sw/initramfs"
make LINUX_DIR="$LINUX_SRC" clean all
cp neo_initramfs.cpio.gz "$OUTPUT_DIR/"
echo "  → $OUTPUT_DIR/neo_initramfs.cpio.gz"

echo ""
echo "=== Step 2: Inject NPU driver into kernel ==="
"$PROJ_DIR/kernel/inject_npu_driver.sh" "$LINUX_SRC"

echo ""
echo "=== Step 3: Inject NEORV32 UART driver ==="
"$PROJ_DIR/board/inject_driver.sh" "$LINUX_SRC"

echo ""
echo "=== Step 4: Configure and build kernel ==="
cd "$LINUX_SRC"

sed "s|__INITRAMFS_PATH__|$OUTPUT_DIR/neo_initramfs.cpio.gz|" \
    "$PROJ_DIR/board/linux_defconfig" > arch/riscv/configs/neorv32_tpu_defconfig

make ARCH=riscv CROSS_COMPILE=riscv-none-elf- neorv32_tpu_defconfig

# Critical: force-disable FPU/ISA_V/ISA_FALLBACK (they auto-enable and break NEORV32)
scripts/config --disable CONFIG_FPU
scripts/config --disable CONFIG_RISCV_ISA_V
scripts/config --disable CONFIG_RISCV_ISA_FALLBACK

make ARCH=riscv CROSS_COMPILE=riscv-none-elf- -j$(nproc)
cp arch/riscv/boot/Image "$OUTPUT_DIR/"
echo "  → $OUTPUT_DIR/Image ($(wc -c < arch/riscv/boot/Image) bytes)"

echo ""
echo "=== Step 5: Compile device tree ==="
dtc -I dts -O dtb -o "$OUTPUT_DIR/neorv32_tpu.dtb" "$PROJ_DIR/board/neorv32_tpu.dts"
echo "  → $OUTPUT_DIR/neorv32_tpu.dtb"

echo ""
echo "=== Step 6: Build stage2 loader ==="
cd "$PROJ_DIR/sw/stage2_loader"
make NEORV32_HOME="$PROJ_DIR/neorv32" clean exe
cp neorv32_exe.bin "$OUTPUT_DIR/stage2_loader.bin"
echo "  → $OUTPUT_DIR/stage2_loader.bin"

echo ""
echo "=== Build complete ==="
echo "Output files in $OUTPUT_DIR/:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Program FPGA:  openFPGALoader -c usb-blaster $OUTPUT_DIR/neorv32_tpu.rbf"
echo "  2. Boot Linux:    python3 host/boot_linux.py --port /dev/ttyUSB0"
