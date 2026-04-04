/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef _NEORV32_NPU_H
#define _NEORV32_NPU_H

#include <linux/types.h>
#include <linux/ioctl.h>

/* 4x4 weight matrix: row-major, int8 */
struct npu_weights {
	__s8 w[4][4];
};

/* Compute request: 4 packed int8 inputs */
struct npu_compute {
	__s8 x[4];
};

/* Result: 4 int32 accumulators */
struct npu_result {
	__s32 res[4];
};

#define NPU_IOC_MAGIC    'N'
#define NPU_LOAD_WEIGHTS  _IOW(NPU_IOC_MAGIC, 1, struct npu_weights)
#define NPU_COMPUTE       _IOW(NPU_IOC_MAGIC, 2, struct npu_compute)
#define NPU_GET_RESULT    _IOR(NPU_IOC_MAGIC, 3, struct npu_result)
#define NPU_CLEAR         _IO(NPU_IOC_MAGIC, 4)

#endif /* _NEORV32_NPU_H */
