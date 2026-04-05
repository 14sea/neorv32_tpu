// SPDX-License-Identifier: GPL-2.0
/*
 * neorv32_npu.c — /dev/npu driver for NEORV32 4x4 systolic array TPU
 *
 * Memory-mapped at 0xF0000000 (from device tree).
 * Provides ioctl interface for weight loading, compute, and result readback.
 *
 * Register map (from tpu_accel.v):
 *   0x00  CTRL    (W)  [0]=start, [4]=clear
 *   0x04  STATUS  (R)  [0]=done
 *   0x08  W_ADDR  (W)  [1:0]=col, [3:2]=row
 *   0x0C  W_DATA  (W)  [7:0]=weight (triggers load pulse)
 *   0x10  X_IN    (W)  [31:0]={x3,x2,x1,x0} packed int8
 *   0x14  W_DATA4 (W)  [31:0]={w3,w2,w1,w0} bulk load 1 row
 *   0x20  RES0    (R)  int32 row 0 accumulator
 *   0x24  RES1    (R)  int32 row 1 accumulator
 *   0x28  RES2    (R)  int32 row 2 accumulator
 *   0x2C  RES3    (R)  int32 row 3 accumulator
 */

#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/of.h>
#include <linux/uaccess.h>
#include <linux/ioctl.h>
#include <linux/mm.h>

/* Register offsets */
#define TPU_CTRL      0x00
#define TPU_STATUS    0x04
#define TPU_W_ADDR    0x08
#define TPU_W_DATA    0x0C
#define TPU_X_IN      0x10
#define TPU_W_DATA4   0x14
#define TPU_RES0      0x20
#define TPU_RES1      0x24
#define TPU_RES2      0x28
#define TPU_RES3      0x2C

/* CTRL bits */
#define CTRL_START    (1 << 0)
#define CTRL_CLEAR    (1 << 4)

/* ioctl commands */
#define NPU_IOC_MAGIC    'N'
#define NPU_LOAD_WEIGHTS  _IOW(NPU_IOC_MAGIC, 1, struct npu_weights)
#define NPU_COMPUTE       _IOW(NPU_IOC_MAGIC, 2, struct npu_compute)
#define NPU_GET_RESULT    _IOR(NPU_IOC_MAGIC, 3, struct npu_result)
#define NPU_CLEAR         _IO(NPU_IOC_MAGIC, 4)

/* 4x4 weight matrix: row-major, int8 */
struct npu_weights {
	int8_t w[4][4];
};

/* Compute request: 4 packed int8 inputs */
struct npu_compute {
	int8_t x[4];
};

/* Result: 4 int32 accumulators */
struct npu_result {
	int32_t res[4];
};

struct neorv32_npu {
	void __iomem *base;
	resource_size_t phys_addr;
	resource_size_t phys_size;
	struct miscdevice misc;
};

static struct neorv32_npu *npu_dev;

static long npu_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	void __iomem *base = npu_dev->base;

	switch (cmd) {
	case NPU_LOAD_WEIGHTS: {
		struct npu_weights wt;
		int row;

		if (copy_from_user(&wt, (void __user *)arg, sizeof(wt)))
			return -EFAULT;

		/* Use W_DATA4 bulk load: one write per row */
		for (row = 0; row < 4; row++) {
			uint32_t packed;

			/* Set W_ADDR row (col is ignored for W_DATA4) */
			writel(row << 2, base + TPU_W_ADDR);

			/* Pack 4 weights: {w3, w2, w1, w0} */
			packed = ((uint32_t)(uint8_t)wt.w[row][0])       |
				 ((uint32_t)(uint8_t)wt.w[row][1] << 8)  |
				 ((uint32_t)(uint8_t)wt.w[row][2] << 16) |
				 ((uint32_t)(uint8_t)wt.w[row][3] << 24);
			writel(packed, base + TPU_W_DATA4);
		}
		return 0;
	}

	case NPU_COMPUTE: {
		struct npu_compute req;
		uint32_t packed, status;
		int timeout = 1000;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		/* Pack x[0..3] into 32 bits */
		packed = ((uint32_t)(uint8_t)req.x[0])       |
			 ((uint32_t)(uint8_t)req.x[1] << 8)  |
			 ((uint32_t)(uint8_t)req.x[2] << 16) |
			 ((uint32_t)(uint8_t)req.x[3] << 24);
		writel(packed, base + TPU_X_IN);

		/* Start compute */
		writel(CTRL_START, base + TPU_CTRL);

		/* Poll for done (9 cycles at 50 MHz — effectively instant) */
		do {
			status = readl(base + TPU_STATUS);
		} while (!(status & 1) && --timeout);

		if (!timeout)
			return -ETIMEDOUT;

		return 0;
	}

	case NPU_GET_RESULT: {
		struct npu_result res;

		res.res[0] = readl(base + TPU_RES0);
		res.res[1] = readl(base + TPU_RES1);
		res.res[2] = readl(base + TPU_RES2);
		res.res[3] = readl(base + TPU_RES3);

		if (copy_to_user((void __user *)arg, &res, sizeof(res)))
			return -EFAULT;
		return 0;
	}

	case NPU_CLEAR:
		writel(CTRL_CLEAR, base + TPU_CTRL);
		return 0;

	default:
		return -ENOTTY;
	}
}

static int npu_mmap(struct file *file, struct vm_area_struct *vma)
{
	unsigned long size = vma->vm_end - vma->vm_start;

	if (vma->vm_pgoff != 0 || size > PAGE_SIZE)
		return -EINVAL;

	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);

	return remap_pfn_range(vma, vma->vm_start,
			       npu_dev->phys_addr >> PAGE_SHIFT,
			       size, vma->vm_page_prot);
}

#ifndef CONFIG_MMU
/* nommu: allow direct mmap of device registers */
static unsigned long npu_get_unmapped_area(struct file *file,
		unsigned long addr, unsigned long len,
		unsigned long pgoff, unsigned long flags)
{
	return npu_dev->phys_addr;
}

static unsigned npu_mmap_capabilities(struct file *file)
{
	return NOMMU_MAP_DIRECT | NOMMU_MAP_READ | NOMMU_MAP_WRITE;
}
#endif

static const struct file_operations npu_fops = {
	.owner          = THIS_MODULE,
	.unlocked_ioctl = npu_ioctl,
	.mmap           = npu_mmap,
#ifndef CONFIG_MMU
	.get_unmapped_area = npu_get_unmapped_area,
	.mmap_capabilities = npu_mmap_capabilities,
#endif
};

static int neorv32_npu_probe(struct platform_device *pdev)
{
	struct resource *res;

	npu_dev = devm_kzalloc(&pdev->dev, sizeof(*npu_dev), GFP_KERNEL);
	if (!npu_dev)
		return -ENOMEM;

	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	npu_dev->base = devm_ioremap_resource(&pdev->dev, res);
	if (IS_ERR(npu_dev->base))
		return PTR_ERR(npu_dev->base);

	npu_dev->phys_addr = res->start;
	npu_dev->phys_size = resource_size(res);

	npu_dev->misc.minor = MISC_DYNAMIC_MINOR;
	npu_dev->misc.name  = "npu";
	npu_dev->misc.fops  = &npu_fops;

	/* Clear accumulators on init */
	writel(CTRL_CLEAR, npu_dev->base + TPU_CTRL);

	dev_info(&pdev->dev, "NEORV32 NPU at %pR\n", res);

	return misc_register(&npu_dev->misc);
}

static int neorv32_npu_remove(struct platform_device *pdev)
{
	misc_deregister(&npu_dev->misc);
	return 0;
}

static const struct of_device_id neorv32_npu_of_match[] = {
	{ .compatible = "neorv32,npu" },
	{ }
};
MODULE_DEVICE_TABLE(of, neorv32_npu_of_match);

static struct platform_driver neorv32_npu_driver = {
	.probe  = neorv32_npu_probe,
	.remove = neorv32_npu_remove,
	.driver = {
		.name = "neorv32-npu",
		.of_match_table = neorv32_npu_of_match,
	},
};
module_platform_driver(neorv32_npu_driver);

MODULE_DESCRIPTION("NEORV32 4x4 Systolic Array NPU Driver");
MODULE_LICENSE("GPL");
