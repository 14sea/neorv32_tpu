/* Minimal nommu init for NEORV32 + TPU — static PIE, no libc
 * Extends the base nommu shell with /dev/npu NPU test commands.
 *
 * Build: riscv32-buildroot-linux-gnu-gcc -nostdlib -nostartfiles -fpie \
 *        -mcmodel=medany -fno-plt -static -Wl,-pie -Wl,--no-dynamic-linker \
 *        -Os -fno-stack-protector -fno-builtin -o init init.c -e _start
 */

static inline __attribute__((always_inline)) long
my_syscall(long n, long a0, long a1, long a2) {
    register long _a7 __asm__("a7") = n;
    register long _a0 __asm__("a0") = a0;
    register long _a1 __asm__("a1") = a1;
    register long _a2 __asm__("a2") = a2;
    __asm__ volatile(
        "ecall"
        : "+r"(_a0)
        : "r"(_a1), "r"(_a2), "r"(_a7)
        : "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6"
    );
    return _a0;
}

/* 4-arg syscall for ioctl */
static inline __attribute__((always_inline)) long
my_syscall4(long n, long a0, long a1, long a2, long a3) {
    register long _a7 __asm__("a7") = n;
    register long _a0 __asm__("a0") = a0;
    register long _a1 __asm__("a1") = a1;
    register long _a2 __asm__("a2") = a2;
    register long _a3 __asm__("a3") = a3;
    __asm__ volatile(
        "ecall"
        : "+r"(_a0)
        : "r"(_a1), "r"(_a2), "r"(_a3), "r"(_a7)
        : "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6"
    );
    return _a0;
}

/* 6-arg syscall for mmap2 */
static inline __attribute__((always_inline)) long
my_syscall6(long n, long a0, long a1, long a2, long a3, long a4, long a5) {
    register long _a7 __asm__("a7") = n;
    register long _a0 __asm__("a0") = a0;
    register long _a1 __asm__("a1") = a1;
    register long _a2 __asm__("a2") = a2;
    register long _a3 __asm__("a3") = a3;
    register long _a4 __asm__("a4") = a4;
    register long _a5 __asm__("a5") = a5;
    __asm__ volatile(
        "ecall"
        : "+r"(_a0)
        : "r"(_a1), "r"(_a2), "r"(_a3), "r"(_a4), "r"(_a5), "r"(_a7)
        : "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6"
    );
    return _a0;
}

/* syscall numbers (RISC-V 32) */
#define __NR_openat     56
#define __NR_close      57
#define __NR_read       63
#define __NR_write      64
#define __NR_ioctl      29
#define __NR_mknodat    33
#define __NR_mount      40
#define __NR_exit       93
#define __NR_mmap2      222
#define __NR_munmap     215
#define __NR_uname      160
#define __NR_sysinfo    179
#define __NR_clock_gettime64 403

#define CLOCK_MONOTONIC 1

struct timespec64 {
    long long tv_sec;
    long long tv_nsec;
};

static unsigned long get_time_us(void) {
    struct timespec64 ts;
    unsigned long nsec;
    ts.tv_sec = 0;
    ts.tv_nsec = 0;
    my_syscall(__NR_clock_gettime64, CLOCK_MONOTONIC, (long)&ts, 0);
    /* Avoid 64-bit division: tv_nsec < 1e9 fits in 32 bits */
    nsec = (unsigned long)ts.tv_nsec;
    return (unsigned long)ts.tv_sec * 1000000UL + nsec / 1000;
}

#define AT_FDCWD        -100
#define O_RDWR          2
#define S_IFCHR         0020000

/* ioctl encoding macros (must match kernel <asm-generic/ioctl.h>) */
#define _IOC_NRBITS     8
#define _IOC_TYPEBITS   8
#define _IOC_SIZEBITS   14
#define _IOC_DIRBITS    2
#define _IOC_NRSHIFT    0
#define _IOC_TYPESHIFT  8
#define _IOC_SIZESHIFT  16
#define _IOC_DIRSHIFT   30
#define _IOC_NONE       0
#define _IOC_WRITE      1
#define _IOC_READ       2
#define _IOC(dir,type,nr,size) \
    (((dir) << _IOC_DIRSHIFT) | ((type) << _IOC_TYPESHIFT) | \
     ((nr) << _IOC_NRSHIFT) | ((size) << _IOC_SIZESHIFT))
#define _IO(type,nr)       _IOC(_IOC_NONE,(type),(nr),0)
#define _IOW(type,nr,sz)   _IOC(_IOC_WRITE,(type),(nr),sizeof(sz))
#define _IOR(type,nr,sz)   _IOC(_IOC_READ,(type),(nr),sizeof(sz))

/* NPU ioctl structures (must match driver) */
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef int int32_t;
typedef unsigned int uint32_t;

struct npu_weights { int8_t w[4][4]; };
struct npu_compute { int8_t x[4]; };
struct npu_result  { int32_t res[4]; };

#define NPU_IOC_MAGIC    'N'
#define NPU_LOAD_WEIGHTS  _IOW(NPU_IOC_MAGIC, 1, struct npu_weights)
#define NPU_COMPUTE       _IOW(NPU_IOC_MAGIC, 2, struct npu_compute)
#define NPU_GET_RESULT    _IOR(NPU_IOC_MAGIC, 3, struct npu_result)
#define NPU_CLEAR         _IO(NPU_IOC_MAGIC, 4)

/* mmap constants */
#define PROT_READ       1
#define PROT_WRITE      2
#define MAP_SHARED      1
#define MAP_FAILED      ((void *)-1)

/* TPU register offsets (for mmap direct access) */
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

#define CTRL_START    (1 << 0)
#define CTRL_CLEAR    (1 << 4)

static volatile uint32_t *npu_regs;  /* mmap'd TPU registers */

static inline void npu_reg_write(int offset, uint32_t val) {
    npu_regs[offset / 4] = val;
}
static inline uint32_t npu_reg_read(int offset) {
    return npu_regs[offset / 4];
}

struct utsname {
    char sysname[65];
    char nodename[65];
    char release[65];
    char version[65];
    char machine[65];
    char domainname[65];
};

struct sysinfo {
    long uptime;
    unsigned long loads[3];
    unsigned long totalram;
    unsigned long freeram;
    unsigned long sharedram;
    unsigned long bufferram;
    unsigned long totalswap;
    unsigned long freeswap;
    unsigned short procs;
    unsigned short pad;
    unsigned long totalhigh;
    unsigned long freehigh;
    unsigned int mem_unit;
    char _f[8];
};

static int my_strlen(const char *s) { int n=0; while(s[n])n++; return n; }
static void my_puts(const char *s) { my_syscall(__NR_write, 1, (long)s, my_strlen(s)); }

static void my_putnum(unsigned long v) {
    char buf[12];
    int i = 11;
    buf[i] = 0;
    if (v == 0) { my_puts("0"); return; }
    do { buf[--i] = '0' + (v % 10); v /= 10; } while (v);
    my_puts(buf + i);
}

static void my_putint(long v) {
    if (v < 0) { my_puts("-"); my_putnum((unsigned long)(-v)); }
    else my_putnum((unsigned long)v);
}

static void my_puthex(unsigned long v) {
    char buf[12];
    int i;
    buf[0] = '0'; buf[1] = 'x';
    for (i = 0; i < 8; i++)
        buf[2+i] = "0123456789abcdef"[(v >> (28 - i*4)) & 0xf];
    buf[10] = 0;
    my_puts(buf);
}

static int my_strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return *a - *b;
}

static void chomp(char *s) {
    int n = my_strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r')) s[--n] = 0;
}

static void *my_memcpy(void *dst, const void *src, unsigned long n) {
    char *d = dst; const char *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

/* GCC emits calls to memcpy/memset for struct init — provide them */
void *memcpy(void *dst, const void *src, unsigned long n) __attribute__((alias("my_memcpy")));

static void *my_memset(void *dst, int c, unsigned long n) {
    char *d = dst;
    while (n--) *d++ = (char)c;
    return dst;
}
void *memset(void *dst, int c, unsigned long n) __attribute__((alias("my_memset")));

static int my_open(const char *path, int flags) {
    return my_syscall(__NR_openat, AT_FDCWD, (long)path, flags);
}

static int my_close(int fd) {
    return my_syscall(__NR_close, fd, 0, 0);
}

static long my_ioctl(int fd, unsigned long cmd, void *arg) {
    return my_syscall(__NR_ioctl, fd, cmd, (long)arg);
}

static int my_mknod(const char *path, unsigned int mode, unsigned int dev) {
    return my_syscall4(__NR_mknodat, AT_FDCWD, (long)path, mode, dev);
}

/* 5-arg syscall for mount */
static inline __attribute__((always_inline)) long
my_syscall5(long n, long a0, long a1, long a2, long a3, long a4) {
    register long _a7 __asm__("a7") = n;
    register long _a0 __asm__("a0") = a0;
    register long _a1 __asm__("a1") = a1;
    register long _a2 __asm__("a2") = a2;
    register long _a3 __asm__("a3") = a3;
    register long _a4 __asm__("a4") = a4;
    __asm__ volatile(
        "ecall"
        : "+r"(_a0)
        : "r"(_a1), "r"(_a2), "r"(_a3), "r"(_a4), "r"(_a7)
        : "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6"
    );
    return _a0;
}

static int my_mount(const char *src, const char *tgt, const char *fstype, unsigned long flags) {
    return my_syscall5(__NR_mount, (long)src, (long)tgt, (long)fstype, flags, 0);
}

/* makedev(major, minor) */
static unsigned int my_makedev(int major, int minor) {
    return ((major & 0xfffff000) << 12) | ((major & 0xfff) << 8) |
           ((minor & 0xffffff00) << 12) | (minor & 0xff);
}

/* ── Shell commands ────────────────────────────────────────────────── */

static void cmd_uname(void) {
    struct utsname u;
    if (my_syscall(__NR_uname, (long)&u, 0, 0) == 0) {
        my_puts(u.sysname); my_puts(" ");
        my_puts(u.nodename); my_puts(" ");
        my_puts(u.release); my_puts(" ");
        my_puts(u.version); my_puts(" ");
        my_puts(u.machine); my_puts("\n");
    }
}

static void cmd_info(void) {
    struct sysinfo si;
    if (my_syscall(__NR_sysinfo, (long)&si, 0, 0) == 0) {
        unsigned long unit = si.mem_unit ? si.mem_unit : 1;
        my_puts("Uptime:    "); my_putnum(si.uptime); my_puts(" s\n");
        my_puts("Total RAM: "); my_putnum((si.totalram * unit) >> 10); my_puts(" KB\n");
        my_puts("Free RAM:  "); my_putnum((si.freeram * unit) >> 10); my_puts(" KB\n");
        my_puts("Processes: "); my_putnum(si.procs); my_puts("\n");
    }
}

/* ── NPU test ──────────────────────────────────────────────────────── */

static int npu_fd = -1;
static int npu_pass, npu_fail;

static int npu_open(void) {
    if (npu_fd >= 0) return 0;

    npu_fd = my_open("/dev/npu", O_RDWR);
    if (npu_fd < 0) {
        /* Try creating the device node first (misc device, dynamic minor) */
        /* We'll try minor 0..255 but the kernel logs the actual minor */
        my_puts("  /dev/npu not found, trying mknod...\n");

        /* Read /sys/class/misc/npu/dev to find the actual minor */
        /* For nommu without /sys, we'll just try the ioctl on the fd
         * after creating with a known minor. The misc driver usually
         * gets minor 59-63 range. Let's try common minors. */

        /* Actually, for simplicity, read /proc/misc to find it */
        int proc_fd = my_open("/proc/misc", 0);
        if (proc_fd >= 0) {
            char pbuf[256];
            int pn = my_syscall(__NR_read, proc_fd, (long)pbuf, 255);
            my_close(proc_fd);
            if (pn > 0) {
                pbuf[pn] = 0;
                /* Find "npu" line, extract minor number */
                char *p = pbuf;
                while (*p) {
                    /* Skip whitespace */
                    while (*p == ' ' || *p == '\t') p++;
                    /* Parse number */
                    int minor = 0;
                    while (*p >= '0' && *p <= '9') { minor = minor * 10 + (*p - '0'); p++; }
                    /* Skip whitespace */
                    while (*p == ' ' || *p == '\t') p++;
                    /* Check name */
                    if (p[0]=='n' && p[1]=='p' && p[2]=='u' && (p[3]=='\n' || p[3]==0)) {
                        my_puts("  Found npu minor="); my_putnum(minor); my_puts("\n");
                        my_mknod("/dev/npu", S_IFCHR | 0666, my_makedev(10, minor));
                        npu_fd = my_open("/dev/npu", O_RDWR);
                        if (npu_fd >= 0) return 0;
                    }
                    /* Skip to next line */
                    while (*p && *p != '\n') p++;
                    if (*p) p++;
                }
            }
        }

        if (npu_fd < 0) {
            my_puts("  ERROR: cannot open /dev/npu\n");
            return -1;
        }
    }
    /* mmap TPU registers for direct userspace access */
    if (!npu_regs) {
        unsigned long ret = (unsigned long)my_syscall6(__NR_mmap2, 0, 4096,
                               PROT_READ | PROT_WRITE, MAP_SHARED,
                               npu_fd, 0);
        /* Kernel error codes are 0xFFFFF000..0xFFFFFFFF (-4096..-1) */
        if (ret >= 0xFFFFF000UL) {
            my_puts("  WARNING: mmap failed (err=");
            my_putint((int)(long)ret);
            my_puts("), using ioctl fallback\n");
        } else {
            npu_regs = (volatile uint32_t *)ret;
        }
    }
    return 0;
}

static void npu_check(const char *name, const int32_t *expected, const int32_t *actual) {
    int ok = 1, i;
    for (i = 0; i < 4; i++) if (expected[i] != actual[i]) { ok = 0; break; }

    if (ok) {
        my_puts("  PASS: "); my_puts(name); my_puts("\n");
        npu_pass++;
    } else {
        my_puts("  FAIL: "); my_puts(name); my_puts("\n");
        my_puts("    expected: [");
        for (i = 0; i < 4; i++) { if (i) my_puts(", "); my_putint(expected[i]); }
        my_puts("]\n    actual:   [");
        for (i = 0; i < 4; i++) { if (i) my_puts(", "); my_putint(actual[i]); }
        my_puts("]\n");
        npu_fail++;
    }
}

static int npu_matmul(const int8_t w[4][4], const int8_t x[4], int32_t res[4]) {
    struct npu_weights wt;
    struct npu_compute comp;
    struct npu_result result;

    my_memcpy(wt.w, w, 16);
    my_memcpy(comp.x, x, 4);

    if (my_ioctl(npu_fd, NPU_CLEAR, 0) < 0) return -1;
    if (my_ioctl(npu_fd, NPU_LOAD_WEIGHTS, &wt) < 0) return -1;
    if (my_ioctl(npu_fd, NPU_COMPUTE, &comp) < 0) return -1;
    if (my_ioctl(npu_fd, NPU_GET_RESULT, &result) < 0) return -1;

    my_memcpy(res, result.res, 16);
    return 0;
}

static void cmd_npu(void) {
    npu_pass = 0;
    npu_fail = 0;

    my_puts("\n=== NPU Test (4x4 systolic array at 0xF0000000) ===\n\n");

    if (npu_open() < 0) return;

    /* Test 1: Identity matrix */
    {
        int8_t w[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        int8_t x[4] = {10, 20, 30, 40};
        int32_t exp[4] = {10, 20, 30, 40};
        int32_t res[4];
        my_puts("Test 1: Identity matrix\n");
        if (npu_matmul(w, x, res) == 0)
            npu_check("I * [10,20,30,40]", exp, res);
        else { my_puts("  FAIL: ioctl error\n"); npu_fail++; }
    }

    /* Test 2: General matmul */
    {
        int8_t w[4][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
        int8_t x[4] = {1, 2, 3, 4};
        int32_t exp[4] = {30, 70, 110, 150};
        int32_t res[4];
        my_puts("Test 2: General 4x4 matmul\n");
        if (npu_matmul(w, x, res) == 0)
            npu_check("W * [1,2,3,4] = [30,70,110,150]", exp, res);
        else { my_puts("  FAIL: ioctl error\n"); npu_fail++; }
    }

    /* Test 3: Signed values */
    {
        int8_t w[4][4] = {{-1,2,-3,4},{-5,6,-7,8},{-128,127,0,1},{100,-100,50,-50}};
        int8_t x[4] = {1, -1, 2, -2};
        int32_t exp[4] = {-17, -41, -257, 400};
        int32_t res[4];
        my_puts("Test 3: Signed values\n");
        if (npu_matmul(w, x, res) == 0)
            npu_check("signed W * [1,-1,2,-2]", exp, res);
        else { my_puts("  FAIL: ioctl error\n"); npu_fail++; }
    }

    /* Test 4: Accumulation across two MACs */
    {
        struct npu_weights wt;
        struct npu_compute comp;
        struct npu_result result;
        int8_t w1[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        int8_t x1[4] = {10, 20, 30, 40};
        int8_t w2[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        int8_t x2[4] = {5, 5, 5, 5};
        int32_t exp[4] = {15, 25, 35, 45};

        my_puts("Test 4: Accumulation (two MACs)\n");

        my_ioctl(npu_fd, NPU_CLEAR, 0);
        my_memcpy(wt.w, w1, 16);
        my_ioctl(npu_fd, NPU_LOAD_WEIGHTS, &wt);
        my_memcpy(comp.x, x1, 4);
        my_ioctl(npu_fd, NPU_COMPUTE, &comp);

        my_memcpy(wt.w, w2, 16);
        my_ioctl(npu_fd, NPU_LOAD_WEIGHTS, &wt);
        my_memcpy(comp.x, x2, 4);
        my_ioctl(npu_fd, NPU_COMPUTE, &comp);

        my_ioctl(npu_fd, NPU_GET_RESULT, &result);
        npu_check("I*[10..40] + I*[5,5,5,5] = [15,25,35,45]", exp, result.res);
    }

    my_puts("\n=== NPU Results: ");
    my_putnum(npu_pass); my_puts("/"); my_putnum(npu_pass + npu_fail);
    if (npu_fail == 0) my_puts(" ALL PASSED");
    else my_puts(" SOME FAILED");
    my_puts(" ===\n\n");
}

/* ── MNIST Inference ───────────────────────────────────────────────── */

#include "mnist_data.h"
#include "cnn_data.h"

/* Temporary buffers for inference (static to avoid stack overflow) */
static int8_t  infer_input[784];
static int8_t  infer_output[128];  /* max dim */
static int32_t infer_acc[128];     /* max dim, int32 for last layer */

static void cmd_mnist(void) {
    int s, correct = 0;
    unsigned long total_us = 0;

    my_puts("\n=== MNIST Inference (3-layer MLP on NPU) ===\n");

    if (npu_open() < 0) return;

    my_puts("  mode: "); my_puts(npu_regs ? "mmap" : "ioctl"); my_puts("\n\n");

    for (s = 0; s < NUM_SAMPLES; s++) {
        int si = test_sample_indices[s];
        int label = test_labels[si];
        const uint8_t *pixels = get_test_sample(s);
        int8_t *cur_in;
        int cur_in_dim;
        int layer, g, t, i;
        unsigned long t0, t1;

        my_puts("Sample "); my_putnum(s);
        my_puts(" (label="); my_putnum(label); my_puts("): ");

        t0 = get_time_us();

        /* Quantize input: uint8 [0,255] → int8 [0,127] */
        for (i = 0; i < 784; i++)
            infer_input[i] = (int8_t)(pixels[i] >> 1);

        cur_in = infer_input;
        cur_in_dim = 784;

        /* Run 3 layers */
        for (layer = 0; layer < NUM_LAYERS; layer++) {
            const layer_dims_t *ld = &layer_dims[layer];
            const int8_t *weights = get_layer_weights(layer);
            const int32_t *bias = get_layer_bias(layer);
            int n_out_groups = ld->out_dim_pad / 4;
            int n_in_tiles = ld->in_dim_pad / 4;
            int is_last = (layer == NUM_LAYERS - 1);

            for (g = 0; g < n_out_groups; g++) {
                int32_t acc[4];

                if (npu_regs) {
                    /* ── Fast path: direct register access via mmap ── */
                    npu_reg_write(TPU_CTRL, CTRL_CLEAR);

                    for (t = 0; t < n_in_tiles; t++) {
                        int row;
                        /* Load 4×4 weight tile via W_DATA4 (1 write per row) */
                        for (row = 0; row < 4; row++) {
                            const int8_t *wp = &weights[(g*4 + row) * ld->in_dim_pad + t*4];
                            npu_reg_write(TPU_W_ADDR, row << 2);
                            npu_reg_write(TPU_W_DATA4,
                                ((uint32_t)(uint8_t)wp[0])       |
                                ((uint32_t)(uint8_t)wp[1] << 8)  |
                                ((uint32_t)(uint8_t)wp[2] << 16) |
                                ((uint32_t)(uint8_t)wp[3] << 24));
                        }
                        /* Pack and write input */
                        {
                            int idx = t * 4;
                            uint8_t x0 = (idx   < cur_in_dim) ? (uint8_t)cur_in[idx]   : 0;
                            uint8_t x1 = (idx+1 < cur_in_dim) ? (uint8_t)cur_in[idx+1] : 0;
                            uint8_t x2 = (idx+2 < cur_in_dim) ? (uint8_t)cur_in[idx+2] : 0;
                            uint8_t x3 = (idx+3 < cur_in_dim) ? (uint8_t)cur_in[idx+3] : 0;
                            npu_reg_write(TPU_X_IN, x0 | (x1<<8) | (x2<<16) | (x3<<24));
                        }
                        /* Start compute and poll done */
                        npu_reg_write(TPU_CTRL, CTRL_START);
                        while (!(npu_reg_read(TPU_STATUS) & 1)) ;
                    }

                    acc[0] = (int32_t)npu_reg_read(TPU_RES0) + bias[g*4];
                    acc[1] = (int32_t)npu_reg_read(TPU_RES1) + bias[g*4+1];
                    acc[2] = (int32_t)npu_reg_read(TPU_RES2) + bias[g*4+2];
                    acc[3] = (int32_t)npu_reg_read(TPU_RES3) + bias[g*4+3];
                } else {
                    /* ── Slow path: ioctl fallback ── */
                    struct npu_weights wt;
                    struct npu_compute comp;
                    struct npu_result result;

                    my_ioctl(npu_fd, NPU_CLEAR, 0);
                    for (t = 0; t < n_in_tiles; t++) {
                        int row, col;
                        for (row = 0; row < 4; row++)
                            for (col = 0; col < 4; col++)
                                wt.w[row][col] = weights[(g*4 + row) * ld->in_dim_pad + t*4 + col];
                        my_ioctl(npu_fd, NPU_LOAD_WEIGHTS, &wt);
                        for (i = 0; i < 4; i++) {
                            int idx = t * 4 + i;
                            comp.x[i] = (idx < cur_in_dim) ? cur_in[idx] : 0;
                        }
                        my_ioctl(npu_fd, NPU_COMPUTE, &comp);
                    }
                    my_ioctl(npu_fd, NPU_GET_RESULT, &result);
                    for (i = 0; i < 4; i++)
                        acc[i] = result.res[i] + bias[g * 4 + i];
                }

                if (is_last) {
                    /* Last layer: keep int32 scores */
                    for (i = 0; i < 4; i++) {
                        int idx = g * 4 + i;
                        if (idx < ld->out_dim)
                            infer_acc[idx] = acc[i];
                    }
                } else {
                    /* Requantize + ReLU → int8 */
                    for (i = 0; i < 4; i++) {
                        int idx = g * 4 + i;
                        int32_t val = (acc[i] * ld->requant_mult) >> 16;
                        if (ld->has_relu && val < 0) val = 0;
                        if (val > 127) val = 127;
                        if (val < -128) val = -128;
                        if (idx < ld->out_dim)
                            infer_output[idx] = (int8_t)val;
                    }
                }
            }

            if (!is_last) {
                /* Copy output to input for next layer */
                for (i = 0; i < ld->out_dim; i++)
                    infer_input[i] = infer_output[i];
                cur_in = infer_input;
                cur_in_dim = ld->out_dim;
            }
        }

        /* Find argmax of int32 scores */
        {
            int pred = 0;
            int32_t max_val = infer_acc[0];
            for (i = 1; i < layer_dims[NUM_LAYERS-1].out_dim; i++) {
                if (infer_acc[i] > max_val) {
                    max_val = infer_acc[i];
                    pred = i;
                }
            }

            t1 = get_time_us();
            {
                unsigned long elapsed = t1 - t0;
                total_us += elapsed;
                my_puts("predicted="); my_putnum(pred);
                if (pred == label) {
                    my_puts(" [CORRECT]");
                    correct++;
                } else {
                    my_puts(" [WRONG, expected="); my_putnum(label); my_puts("]");
                }
                my_puts(" ("); my_putnum(elapsed / 1000); my_puts(".");
                my_putnum((elapsed % 1000) / 100); my_puts(" ms)\n");
            }
        }
    }

    my_puts("\n=== MNIST: "); my_putnum(correct);
    my_puts("/"); my_putnum(NUM_SAMPLES);
    my_puts(" correct, total "); my_putnum(total_us / 1000);
    my_puts("."); my_putnum((total_us % 1000) / 100);
    my_puts(" ms ("); my_putnum(total_us / NUM_SAMPLES / 1000);
    my_puts("."); my_putnum((total_us / NUM_SAMPLES % 1000) / 100);
    my_puts(" ms/sample) ===\n\n");
}

static void cmd_bench(void) {
    volatile uint32_t *saved_regs = npu_regs;

    my_puts("\n=== NPU Benchmark: mmap vs ioctl ===\n");

    if (npu_open() < 0) return;

    if (!npu_regs) {
        my_puts("  mmap not available, cannot compare\n\n");
        cmd_mnist();
        return;
    }

    /* Run with mmap */
    my_puts("\n--- mmap (direct register access) ---\n");
    cmd_mnist();

    /* Force ioctl path */
    npu_regs = (volatile uint32_t *)0;
    my_puts("--- ioctl (syscall per operation) ---\n");
    cmd_mnist();

    /* Restore */
    npu_regs = saved_regs;
}

/* ── CNN Inference ─────────────────────────────────────────────────── */

/* Static buffers for CNN (avoid stack overflow) */
static int8_t  cnn_feat_in[784];       /* max: 1×28×28 = 784 */
static int8_t  cnn_feat_out[2304];     /* max: 4×24×24 = 2304 (conv0 pre-pool) */
static int8_t  cnn_patches[576 * 28];  /* max: 576 patches × 28 (conv0 K_pad) */
static int8_t  cnn_fc_buf[200];        /* FC input/output */
static int32_t cnn_scores[12];         /* FC1 output (pad4(10)) */

/* im2col: extract patches from (C, H, W) feature map
 * Output: patches[P][K_pad] where P = out_h * out_w, K = C*kH*kW
 * patches must be pre-zeroed for K_pad > K padding */
static void im2col(const int8_t *feat, int C, int H, int W,
                   int kH, int kW, int K_pad,
                   int8_t *patches, int out_h, int out_w)
{
    int oh, ow, c, kh, kw, k;
    int P = out_h * out_w;

    /* Zero the patch buffer for padding */
    for (k = 0; k < P * K_pad; k++)
        patches[k] = 0;

    for (oh = 0; oh < out_h; oh++) {
        for (ow = 0; ow < out_w; ow++) {
            int p = oh * out_w + ow;
            k = 0;
            for (c = 0; c < C; c++) {
                for (kh = 0; kh < kH; kh++) {
                    for (kw = 0; kw < kW; kw++) {
                        patches[p * K_pad + k] =
                            feat[c * H * W + (oh + kh) * W + (ow + kw)];
                        k++;
                    }
                }
            }
        }
    }
}

/* Run one conv layer on NPU: im2col → tiled 4×4 matmul → requant+ReLU
 * feat_in: (in_ch, in_h, in_w) int8
 * feat_out: (out_ch, out_h, out_w) int8 */
static void cnn_conv_npu(const int8_t *feat_in, int8_t *feat_out,
                         int conv_idx)
{
    const cnn_conv_t *cd = &cnn_conv_dims[conv_idx];
    const int8_t *weights = cnn_get_conv_w(conv_idx);
    const int32_t *bias = cnn_get_conv_b(conv_idx);
    int P = cd->out_h * cd->out_w;
    int n_out_groups = cd->out_ch_pad / 4;
    int n_in_tiles = cd->patch_size_pad / 4;
    int p, g, t, row, i;

    /* im2col */
    im2col(feat_in, cd->in_ch, cd->in_h, cd->in_w,
           cd->kH, cd->kW, cd->patch_size_pad,
           cnn_patches, cd->out_h, cd->out_w);

    /* For each patch, run tiled matmul */
    for (p = 0; p < P; p++) {
        const int8_t *patch = &cnn_patches[p * cd->patch_size_pad];

        for (g = 0; g < n_out_groups; g++) {
            int32_t acc[4];

            npu_reg_write(TPU_CTRL, CTRL_CLEAR);

            for (t = 0; t < n_in_tiles; t++) {
                /* Load 4×4 weight tile */
                for (row = 0; row < 4; row++) {
                    const int8_t *wp = &weights[(g*4 + row) * cd->patch_size_pad + t*4];
                    npu_reg_write(TPU_W_ADDR, row << 2);
                    npu_reg_write(TPU_W_DATA4,
                        ((uint32_t)(uint8_t)wp[0])       |
                        ((uint32_t)(uint8_t)wp[1] << 8)  |
                        ((uint32_t)(uint8_t)wp[2] << 16) |
                        ((uint32_t)(uint8_t)wp[3] << 24));
                }
                /* Pack and write input */
                {
                    const int8_t *xp = &patch[t * 4];
                    npu_reg_write(TPU_X_IN,
                        ((uint32_t)(uint8_t)xp[0])       |
                        ((uint32_t)(uint8_t)xp[1] << 8)  |
                        ((uint32_t)(uint8_t)xp[2] << 16) |
                        ((uint32_t)(uint8_t)xp[3] << 24));
                }
                npu_reg_write(TPU_CTRL, CTRL_START);
                while (!(npu_reg_read(TPU_STATUS) & 1)) ;
            }

            acc[0] = (int32_t)npu_reg_read(TPU_RES0) + bias[g*4];
            acc[1] = (int32_t)npu_reg_read(TPU_RES1) + bias[g*4+1];
            acc[2] = (int32_t)npu_reg_read(TPU_RES2) + bias[g*4+2];
            acc[3] = (int32_t)npu_reg_read(TPU_RES3) + bias[g*4+3];

            /* Requant + ReLU → int8 */
            for (i = 0; i < 4; i++) {
                int idx = g * 4 + i;
                if (idx < cd->out_ch) {
                    int32_t val = (acc[i] * cd->requant_mult) >> 16;
                    if (val < 0) val = 0;
                    if (val > 127) val = 127;
                    /* Store in (F, H_out, W_out) layout: feat_out[f * P + p] */
                    feat_out[idx * P + p] = (int8_t)val;
                }
            }
        }
    }
}

/* 2×2 max pooling: (C, H, W) → (C, H/2, W/2) */
static void cnn_maxpool(const int8_t *in, int8_t *out,
                        int C, int H, int W)
{
    int c, h, w;
    int H_out = H / 2;
    int W_out = W / 2;

    for (c = 0; c < C; c++) {
        for (h = 0; h < H_out; h++) {
            for (w = 0; w < W_out; w++) {
                int8_t v00 = in[c*H*W + (h*2)*W + w*2];
                int8_t v01 = in[c*H*W + (h*2)*W + w*2+1];
                int8_t v10 = in[c*H*W + (h*2+1)*W + w*2];
                int8_t v11 = in[c*H*W + (h*2+1)*W + w*2+1];
                int8_t m = v00;
                if (v01 > m) m = v01;
                if (v10 > m) m = v10;
                if (v11 > m) m = v11;
                out[c*H_out*W_out + h*W_out + w] = m;
            }
        }
    }
}

/* Run one FC layer on NPU (same as MLP tiled matmul) */
static void cnn_fc_npu(const int8_t *x_in, int x_dim,
                       void *out, int fc_idx, int is_last)
{
    const cnn_fc_t *fd = &cnn_fc_dims[fc_idx];
    const int8_t *weights = cnn_get_fc_w(fc_idx);
    const int32_t *bias = cnn_get_fc_b(fc_idx);
    int n_out_groups = fd->out_pad / 4;
    int n_in_tiles = fd->in_pad / 4;
    int g, t, row, i;

    for (g = 0; g < n_out_groups; g++) {
        int32_t acc[4];

        npu_reg_write(TPU_CTRL, CTRL_CLEAR);

        for (t = 0; t < n_in_tiles; t++) {
            for (row = 0; row < 4; row++) {
                const int8_t *wp = &weights[(g*4 + row) * fd->in_pad + t*4];
                npu_reg_write(TPU_W_ADDR, row << 2);
                npu_reg_write(TPU_W_DATA4,
                    ((uint32_t)(uint8_t)wp[0])       |
                    ((uint32_t)(uint8_t)wp[1] << 8)  |
                    ((uint32_t)(uint8_t)wp[2] << 16) |
                    ((uint32_t)(uint8_t)wp[3] << 24));
            }
            {
                int idx = t * 4;
                uint8_t x0 = (idx   < x_dim) ? (uint8_t)x_in[idx]   : 0;
                uint8_t x1 = (idx+1 < x_dim) ? (uint8_t)x_in[idx+1] : 0;
                uint8_t x2 = (idx+2 < x_dim) ? (uint8_t)x_in[idx+2] : 0;
                uint8_t x3 = (idx+3 < x_dim) ? (uint8_t)x_in[idx+3] : 0;
                npu_reg_write(TPU_X_IN, x0 | (x1<<8) | (x2<<16) | (x3<<24));
            }
            npu_reg_write(TPU_CTRL, CTRL_START);
            while (!(npu_reg_read(TPU_STATUS) & 1)) ;
        }

        acc[0] = (int32_t)npu_reg_read(TPU_RES0) + bias[g*4];
        acc[1] = (int32_t)npu_reg_read(TPU_RES1) + bias[g*4+1];
        acc[2] = (int32_t)npu_reg_read(TPU_RES2) + bias[g*4+2];
        acc[3] = (int32_t)npu_reg_read(TPU_RES3) + bias[g*4+3];

        if (is_last) {
            for (i = 0; i < 4; i++) {
                int idx = g * 4 + i;
                if (idx < fd->out_dim)
                    ((int32_t *)out)[idx] = acc[i];
            }
        } else {
            for (i = 0; i < 4; i++) {
                int idx = g * 4 + i;
                int32_t val = (acc[i] * fd->requant_mult) >> 16;
                if (fd->has_relu && val < 0) val = 0;
                if (val > 127) val = 127;
                if (val < -128) val = -128;
                if (idx < fd->out_dim)
                    ((int8_t *)out)[idx] = (int8_t)val;
            }
        }
    }
}

static void cmd_cnn(void) {
    int s, correct = 0;
    unsigned long total_us = 0;

    my_puts("\n=== CNN Inference (im2col on NPU) ===\n");
    my_puts("  Conv(1->4, 5x5)+Pool -> Conv(4->8, 3x3)+Pool -> FC(200->64) -> FC(64->10)\n");

    if (npu_open() < 0) return;

    if (!npu_regs) {
        my_puts("  ERROR: CNN requires mmap (too many NPU ops for ioctl)\n\n");
        return;
    }

    my_puts("  mode: mmap\n\n");

    for (s = 0; s < CNN_NUM_SAMPLES; s++) {
        int label = cnn_test_labels[s];
        const int8_t *pixels = cnn_get_test_sample(s);
        int i;
        unsigned long t0, t1, elapsed;

        my_puts("Sample "); my_putnum(s);
        my_puts(" (label="); my_putnum(label); my_puts("): ");

        t0 = get_time_us();

        /* Copy input to feat_in as (1, 28, 28) */
        for (i = 0; i < 784; i++)
            cnn_feat_in[i] = pixels[i];

        /* Conv0: (1,28,28) → (4,24,24) → pool → (4,12,12) */
        cnn_conv_npu(cnn_feat_in, cnn_feat_out, 0);
        cnn_maxpool(cnn_feat_out, cnn_feat_in, 4, 24, 24);
        /* cnn_feat_in now has (4,12,12) = 576 bytes */

        /* Conv1: (4,12,12) → (8,10,10) → pool → (8,5,5) */
        cnn_conv_npu(cnn_feat_in, cnn_feat_out, 1);
        cnn_maxpool(cnn_feat_out, cnn_feat_in, 8, 10, 10);
        /* cnn_feat_in now has (8,5,5) = 200 bytes */

        /* Flatten → FC0: 200 → 64 */
        cnn_fc_npu(cnn_feat_in, 200, cnn_fc_buf, 0, 0);

        /* FC1: 64 → 10 (int32 scores) */
        cnn_fc_npu(cnn_fc_buf, 64, cnn_scores, 1, 1);

        /* Find argmax */
        {
            int pred = 0;
            int32_t max_val = cnn_scores[0];
            for (i = 1; i < 10; i++) {
                if (cnn_scores[i] > max_val) {
                    max_val = cnn_scores[i];
                    pred = i;
                }
            }

            t1 = get_time_us();
            elapsed = t1 - t0;
            total_us += elapsed;

            my_puts("predicted="); my_putnum(pred);
            if (pred == label) {
                my_puts(" [CORRECT]");
                correct++;
            } else {
                my_puts(" [WRONG, expected="); my_putnum(label); my_puts("]");
            }
            my_puts(" ("); my_putnum(elapsed / 1000); my_puts(".");
            my_putnum((elapsed % 1000) / 100); my_puts(" ms)\n");
        }
    }

    my_puts("\n=== CNN: "); my_putnum(correct);
    my_puts("/"); my_putnum(CNN_NUM_SAMPLES);
    my_puts(" correct, total "); my_putnum(total_us / 1000);
    my_puts("."); my_putnum((total_us % 1000) / 100);
    my_puts(" ms ("); my_putnum(total_us / CNN_NUM_SAMPLES / 1000);
    my_puts("."); my_putnum((total_us / CNN_NUM_SAMPLES % 1000) / 100);
    my_puts(" ms/sample) ===\n\n");
}

static void cmd_help(void) {
    my_puts("Commands:\n");
    my_puts("  uname  - kernel info\n");
    my_puts("  info   - memory & uptime\n");
    my_puts("  npu    - test NPU (4x4 systolic array)\n");
    my_puts("  mnist  - run MNIST inference on NPU (3-layer MLP)\n");
    my_puts("  cnn    - run CNN inference on NPU (im2col)\n");
    my_puts("  bench  - benchmark mmap vs ioctl\n");
    my_puts("  help   - this message\n");
    my_puts("  exit   - halt system\n");
}

void _start(void) __attribute__((section(".text.init")));
void _start(void) {
    char buf[128];
    int n;

    /* Mount essential filesystems */
    my_mount("devtmpfs", "/dev", "devtmpfs", 0);
    my_mount("proc", "/proc", "proc", 0);

    my_puts("\n");
    my_puts("========================================\n");
    my_puts(" NEORV32 + NPU — Linux mini shell       \n");
    my_puts("========================================\n");
    cmd_uname();
    cmd_info();
    my_puts("\nType 'npu' to test, 'mnist' or 'cnn' for inference.\n\n");

    for (;;) {
        my_puts("npu# ");
        n = my_syscall(__NR_read, 0, (long)buf, 127);
        if (n <= 0) break;
        buf[n] = 0;
        chomp(buf);
        if (buf[0] == 0) continue;

        if (my_strcmp(buf, "uname") == 0) cmd_uname();
        else if (my_strcmp(buf, "info") == 0) cmd_info();
        else if (my_strcmp(buf, "npu") == 0) cmd_npu();
        else if (my_strcmp(buf, "mnist") == 0) cmd_mnist();
        else if (my_strcmp(buf, "cnn") == 0) cmd_cnn();
        else if (my_strcmp(buf, "bench") == 0) cmd_bench();
        else if (my_strcmp(buf, "help") == 0) cmd_help();
        else if (my_strcmp(buf, "exit") == 0) break;
        else { my_puts("unknown: "); my_puts(buf); my_puts("\n"); }
    }

    my_puts("Halting.\n");
    my_syscall(__NR_exit, 0, 0, 0);
}
