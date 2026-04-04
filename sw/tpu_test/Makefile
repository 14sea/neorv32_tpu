# tpu_test/Makefile — NEORV32 bare-metal TPU test for AX301

# RV32IMC + Zicsr/Zifencei
MARCH = rv32imc_zicsr_zifencei

# Toolchain
RISCV_PREFIX = /home/test/xpack-riscv-none-elf-gcc-14.2.0-3/bin/riscv-none-elf-

# Optimization
EFFORT = -Os

# IMEM/DMEM sizes must match RTL (ax301_top.vhd generics)
USER_FLAGS += -Wl,--defsym,__neorv32_rom_size=16k
USER_FLAGS += -Wl,--defsym,__neorv32_ram_size=8k

# UART port for 'make upload'
UART_TTY ?= /dev/ttyUSB0

# Path to NEORV32 root
NEORV32_HOME ?= ../../neorv32

include $(NEORV32_HOME)/sw/common/common.mk
