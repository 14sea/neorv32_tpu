#!/bin/bash
# Inject NEORV32 NPU driver into kernel source tree
# Usage: ./inject_npu_driver.sh /path/to/linux-source
#
# Copies neorv32_npu.c into drivers/misc/ and patches Kconfig + Makefile.
# Idempotent — safe to run multiple times.

set -e

KERN_DIR="${1:?Usage: $0 /path/to/linux-source}"
MISC_DIR="$KERN_DIR/drivers/misc"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$MISC_DIR/Kconfig" ]; then
    echo "ERROR: $MISC_DIR/Kconfig not found"
    exit 1
fi

# 1. Copy driver source
echo "Copying neorv32_npu.c ..."
cp "$SCRIPT_DIR/neorv32_npu.c" "$MISC_DIR/neorv32_npu.c"

# 2. Copy header to include/uapi/linux/
UAPI_DIR="$KERN_DIR/include/uapi/linux"
echo "Copying neorv32_npu.h to $UAPI_DIR ..."
cp "$SCRIPT_DIR/neorv32_npu.h" "$UAPI_DIR/neorv32_npu.h"

# 3. Patch Kconfig
if ! grep -q "NEORV32_NPU" "$MISC_DIR/Kconfig"; then
    echo "Patching Kconfig ..."
    cat >> "$MISC_DIR/Kconfig" << 'KCONFIG'

config NEORV32_NPU
	bool "NEORV32 NPU (4x4 systolic array) support"
	depends on HAS_IOMEM && OF
	help
	  Driver for the NEORV32 4x4 INT8 systolic array NPU accelerator.
	  Provides /dev/npu with ioctl interface for weight loading,
	  matrix-vector compute, and result readback.
	  Say 'Y' here if your NEORV32 SoC has the TPU peripheral.
KCONFIG
    echo "  Kconfig patched."
else
    echo "  Kconfig already patched, skipping."
fi

# 4. Patch Makefile
if ! grep -q "NEORV32_NPU" "$MISC_DIR/Makefile"; then
    echo "Patching Makefile ..."
    echo 'obj-$(CONFIG_NEORV32_NPU) += neorv32_npu.o' >> "$MISC_DIR/Makefile"
    echo "  Makefile patched."
else
    echo "  Makefile already patched, skipping."
fi

echo "Done. NEORV32 NPU driver injected into $KERN_DIR"
