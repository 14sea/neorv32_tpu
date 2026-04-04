#!/usr/bin/env python3
"""
boot_linux.py — Boot nommu Linux on NEORV32 + NPU (AX301)

Wraps the see_neorv32_run_linux boot_linux.py with neorv32_tpu paths.
Run from the neorv32_tpu/ directory.

Usage:
    python3 host/boot_linux.py --port /dev/ttyUSB0
    python3 host/boot_linux.py --port /dev/ttyUSB0 --skip-program
"""

import os
import sys
import subprocess

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_BOOT = "/home/test/see_neorv32_run_linux/host/boot_linux.py"


def main():
    output_dir = os.path.join(PROJ_DIR, "output")

    # Check that output files exist
    for f in ["stage2_loader.bin", "Image", "neorv32_tpu.dtb"]:
        path = os.path.join(output_dir, f)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run build_linux.sh first.")
            sys.exit(1)

    initramfs = os.path.join(output_dir, "neo_initramfs.cpio.gz")
    if not os.path.exists(initramfs):
        initramfs = None  # embedded in kernel

    # Build command
    cmd = [sys.executable, LINUX_BOOT]

    # Pass through all arguments
    cmd.extend(sys.argv[1:])

    # Override paths via environment or command-line
    # The boot_linux.py looks for files relative to its own repo.
    # We need to override them. Let's check if it supports --image etc.
    # Actually, the simplest approach is to symlink our output files.

    # Create symlinks in see_neorv32_run_linux/output/ pointing to our files
    target_output = "/home/test/see_neorv32_run_linux/output"
    os.makedirs(target_output, exist_ok=True)

    links = {
        "neorv32_demo.rbf": os.path.join(PROJ_DIR, "neorv32_tpu.rbf"),
        "stage2_loader.bin": os.path.join(output_dir, "stage2_loader.bin"),
        "Image": os.path.join(output_dir, "Image"),
        "neorv32_ax301.dtb": os.path.join(output_dir, "neorv32_tpu.dtb"),
    }
    if initramfs:
        links["neo_initramfs.cpio.gz"] = initramfs

    for name, src in links.items():
        dst = os.path.join(target_output, name)
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.exists(dst):
            # Back up existing file
            os.rename(dst, dst + ".bak")
        os.symlink(src, dst)
        print(f"  {name} → {src}")

    print()
    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        os.chdir("/home/test/see_neorv32_run_linux")
        os.execv(sys.executable, cmd)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
