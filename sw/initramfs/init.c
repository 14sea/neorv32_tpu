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

/* syscall numbers (RISC-V 32) */
#define __NR_openat     56
#define __NR_close      57
#define __NR_read       63
#define __NR_write      64
#define __NR_ioctl      29
#define __NR_mknodat    33
#define __NR_mount      40
#define __NR_exit       93
#define __NR_uname      160
#define __NR_sysinfo    179

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

static void cmd_help(void) {
    my_puts("Commands:\n");
    my_puts("  uname  - kernel info\n");
    my_puts("  info   - memory & uptime\n");
    my_puts("  npu    - test NPU (4x4 systolic array)\n");
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
    my_puts("\nType 'help' for commands. 'npu' to test.\n\n");

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
        else if (my_strcmp(buf, "help") == 0) cmd_help();
        else if (my_strcmp(buf, "exit") == 0) break;
        else { my_puts("unknown: "); my_puts(buf); my_puts("\n"); }
    }

    my_puts("Halting.\n");
    my_syscall(__NR_exit, 0, 0, 0);
}
