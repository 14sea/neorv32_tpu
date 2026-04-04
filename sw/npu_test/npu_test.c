/*
 * npu_test.c — Userspace test for /dev/npu (NEORV32 4x4 systolic array)
 *
 * Tests:
 *   1. Identity matrix: I × x = x
 *   2. General 4x4 matmul with known result
 *   3. Accumulation across multiple MACs (tiled matmul)
 *   4. Signed values (-128, 127)
 *
 * Build with Linux toolchain (static-PIE for nommu):
 *   riscv32-buildroot-linux-gnu-gcc -static-pie -o npu_test npu_test.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/ioctl.h>

/* ioctl definitions (must match kernel driver) */
struct npu_weights {
	int8_t w[4][4];
};

struct npu_compute {
	int8_t x[4];
};

struct npu_result {
	int32_t res[4];
};

#define NPU_IOC_MAGIC    'N'
#define NPU_LOAD_WEIGHTS  _IOW(NPU_IOC_MAGIC, 1, struct npu_weights)
#define NPU_COMPUTE       _IOW(NPU_IOC_MAGIC, 2, struct npu_compute)
#define NPU_GET_RESULT    _IOR(NPU_IOC_MAGIC, 3, struct npu_result)
#define NPU_CLEAR         _IO(NPU_IOC_MAGIC, 4)

static int fd;
static int pass_count, fail_count;

static void check_result(const char *name, const int32_t *expected, const int32_t *actual)
{
	int ok = 1;
	int i;

	for (i = 0; i < 4; i++) {
		if (expected[i] != actual[i]) {
			ok = 0;
			break;
		}
	}

	if (ok) {
		printf("  PASS: %s\n", name);
		pass_count++;
	} else {
		printf("  FAIL: %s\n", name);
		printf("    expected: [%d, %d, %d, %d]\n",
		       expected[0], expected[1], expected[2], expected[3]);
		printf("    actual:   [%d, %d, %d, %d]\n",
		       actual[0], actual[1], actual[2], actual[3]);
		fail_count++;
	}
}

/* Compute W × x using ioctl and return result */
static int npu_matmul(const int8_t w[4][4], const int8_t x[4], int32_t res[4])
{
	struct npu_weights wt;
	struct npu_compute comp;
	struct npu_result result;

	memcpy(wt.w, w, sizeof(wt.w));
	memcpy(comp.x, x, sizeof(comp.x));

	if (ioctl(fd, NPU_CLEAR, 0) < 0) return -1;
	if (ioctl(fd, NPU_LOAD_WEIGHTS, &wt) < 0) return -1;
	if (ioctl(fd, NPU_COMPUTE, &comp) < 0) return -1;
	if (ioctl(fd, NPU_GET_RESULT, &result) < 0) return -1;

	memcpy(res, result.res, sizeof(result.res));
	return 0;
}

static void test_identity(void)
{
	int8_t w[4][4] = {
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1}
	};
	int8_t x[4] = {10, 20, 30, 40};
	int32_t expected[4] = {10, 20, 30, 40};
	int32_t res[4];

	printf("Test 1: Identity matrix\n");
	if (npu_matmul(w, x, res) < 0) {
		printf("  FAIL: ioctl error\n");
		fail_count++;
		return;
	}
	check_result("I * [10,20,30,40]", expected, res);
}

static void test_general_matmul(void)
{
	/* W = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
	 * x = [1,2,3,4]
	 * res[r] = sum(W[r][c] * x[c]) for c=0..3
	 * res[0] = 1*1 + 2*2 + 3*3 + 4*4 = 30
	 * res[1] = 5*1 + 6*2 + 7*3 + 8*4 = 70
	 * res[2] = 9*1 + 10*2 + 11*3 + 12*4 = 110
	 * res[3] = 13*1 + 14*2 + 15*3 + 16*4 = 150
	 */
	int8_t w[4][4] = {
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
		{13, 14, 15, 16}
	};
	int8_t x[4] = {1, 2, 3, 4};
	int32_t expected[4] = {30, 70, 110, 150};
	int32_t res[4];

	printf("Test 2: General 4x4 matmul\n");
	if (npu_matmul(w, x, res) < 0) {
		printf("  FAIL: ioctl error\n");
		fail_count++;
		return;
	}
	check_result("W * [1,2,3,4]", expected, res);
}

static void test_signed_values(void)
{
	/* W = [[-1,2,-3,4],[-5,6,-7,8],[-128,127,0,1],[100,-100,50,-50]]
	 * x = [1,-1,2,-2]
	 * res[0] = -1*1 + 2*(-1) + (-3)*2 + 4*(-2) = -1-2-6-8 = -17
	 * res[1] = -5*1 + 6*(-1) + (-7)*2 + 8*(-2) = -5-6-14-16 = -41
	 * res[2] = -128*1 + 127*(-1) + 0*2 + 1*(-2) = -128-127+0-2 = -257
	 * res[3] = 100*1 + (-100)*(-1) + 50*2 + (-50)*(-2) = 100+100+100+100 = 400
	 */
	int8_t w[4][4] = {
		{-1, 2, -3, 4},
		{-5, 6, -7, 8},
		{-128, 127, 0, 1},
		{100, -100, 50, -50}
	};
	int8_t x[4] = {1, -1, 2, -2};
	int32_t expected[4] = {-17, -41, -257, 400};
	int32_t res[4];

	printf("Test 3: Signed values\n");
	if (npu_matmul(w, x, res) < 0) {
		printf("  FAIL: ioctl error\n");
		fail_count++;
		return;
	}
	check_result("signed W * [1,-1,2,-2]", expected, res);
}

static void test_accumulation(void)
{
	/* Test accumulation across two MACs (simulates tiled matmul).
	 * Load W1, compute with x1, then load W2, compute with x2 (without clearing).
	 * acc = W1*x1 + W2*x2
	 */
	struct npu_weights wt;
	struct npu_compute comp;
	struct npu_result result;

	/* W1 = identity, x1 = [10,20,30,40] → acc = [10,20,30,40] */
	int8_t w1[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
	int8_t x1[4] = {10, 20, 30, 40};

	/* W2 = identity, x2 = [5,5,5,5] → acc += [5,5,5,5] = [15,25,35,45] */
	int8_t w2[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
	int8_t x2[4] = {5, 5, 5, 5};

	int32_t expected[4] = {15, 25, 35, 45};

	printf("Test 4: Accumulation across two MACs\n");

	/* Clear, load W1, compute x1 */
	ioctl(fd, NPU_CLEAR, 0);
	memcpy(wt.w, w1, sizeof(wt.w));
	ioctl(fd, NPU_LOAD_WEIGHTS, &wt);
	memcpy(comp.x, x1, sizeof(comp.x));
	ioctl(fd, NPU_COMPUTE, &comp);

	/* Load W2, compute x2 (no clear — accumulates) */
	memcpy(wt.w, w2, sizeof(wt.w));
	ioctl(fd, NPU_LOAD_WEIGHTS, &wt);
	memcpy(comp.x, x2, sizeof(comp.x));
	ioctl(fd, NPU_COMPUTE, &comp);

	/* Read accumulated result */
	ioctl(fd, NPU_GET_RESULT, &result);

	check_result("I*[10..40] + I*[5,5,5,5]", expected, result.res);
}

int main(void)
{
	printf("\n=== NEORV32 NPU Test (/dev/npu) ===\n\n");

	fd = open("/dev/npu", O_RDWR);
	if (fd < 0) {
		perror("open /dev/npu");
		printf("\nFAILED: cannot open /dev/npu\n");
		return 1;
	}

	test_identity();
	test_general_matmul();
	test_signed_values();
	test_accumulation();

	close(fd);

	printf("\n=== Results: %d passed, %d failed ===\n\n",
	       pass_count, fail_count);

	return fail_count ? 1 : 0;
}
