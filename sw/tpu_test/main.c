// main.c — NEORV32 bare-metal TPU accelerator test
//
// Tests the 4x4 INT8 systolic array via Wishbone XBUS at 0xF0000000.
// Performs register read/write tests, then a 4x4 matrix multiply
// and compares against known reference values.

#include <neorv32.h>

#define BAUD_RATE 115200

// TPU register base address (XBUS)
#define TPU_BASE    0xF0000000U

// TPU register offsets
#define TPU_CTRL    (*(volatile uint32_t *)(TPU_BASE + 0x00))
#define TPU_STATUS  (*(volatile uint32_t *)(TPU_BASE + 0x04))
#define TPU_W_ADDR  (*(volatile uint32_t *)(TPU_BASE + 0x08))
#define TPU_W_DATA  (*(volatile uint32_t *)(TPU_BASE + 0x0C))
#define TPU_X_IN    (*(volatile uint32_t *)(TPU_BASE + 0x10))
#define TPU_W_DATA4 (*(volatile uint32_t *)(TPU_BASE + 0x14))
#define TPU_RES(r)  (*(volatile uint32_t *)(TPU_BASE + 0x20 + (r)*4))

static int test_count = 0;
static int pass_count = 0;

static void check(const char *name, int32_t expected, int32_t actual) {
    test_count++;
    if (expected == actual) {
        pass_count++;
        neorv32_uart0_printf("  [PASS] %s = %d\n", name, actual);
    } else {
        neorv32_uart0_printf("  [FAIL] %s: expected %d, got %d\n", name, expected, actual);
    }
}

// Load one weight via single W_DATA register
static void load_weight(int row, int col, int8_t w) {
    TPU_W_ADDR = (uint32_t)((row << 2) | col);
    TPU_W_DATA = (uint32_t)(uint8_t)w;
}

// Load 4 weights for one row via W_DATA4 bulk register
static void load_weight_row(int row, int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    TPU_W_ADDR = (uint32_t)(row << 2);  // set row, col ignored for W_DATA4
    TPU_W_DATA4 = ((uint32_t)(uint8_t)w3 << 24) |
                  ((uint32_t)(uint8_t)w2 << 16) |
                  ((uint32_t)(uint8_t)w1 <<  8) |
                  ((uint32_t)(uint8_t)w0);
}

// Run one MAC: write x_in, start compute, wait for done
static void run_mac(int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
    TPU_X_IN = ((uint32_t)(uint8_t)x3 << 24) |
               ((uint32_t)(uint8_t)x2 << 16) |
               ((uint32_t)(uint8_t)x1 <<  8) |
               ((uint32_t)(uint8_t)x0);
    TPU_CTRL = 0x01;  // start
    while (!(TPU_STATUS & 1)) { }  // wait for done
}

// Clear accumulators
static void clear_acc(void) {
    TPU_CTRL = 0x10;  // clear bit
}

int main(void) {
    neorv32_rte_setup();
    neorv32_uart0_setup(BAUD_RATE, 0);

    neorv32_uart0_printf("\n\n");
    neorv32_uart0_printf("=====================================\n");
    neorv32_uart0_printf("  NEORV32 + TPU Accelerator Test    \n");
    neorv32_uart0_printf("=====================================\n");
    neorv32_uart0_printf("  CPU: RV32IMC @ 50 MHz             \n");
    neorv32_uart0_printf("  TPU: 4x4 INT8 systolic array      \n");
    neorv32_uart0_printf("  TPU base: 0x%x\n", TPU_BASE);
    neorv32_uart0_printf("=====================================\n\n");

    // ── Test 1: Register access ──────────────────────────────────────────
    neorv32_uart0_printf("Test 1: Register access\n");

    // Clear and check STATUS
    clear_acc();
    uint32_t status = TPU_STATUS;
    neorv32_uart0_printf("  STATUS after clear = 0x%x\n", status);

    // Read RES0-3 after clear (should be 0)
    check("RES0 after clear", 0, (int32_t)TPU_RES(0));
    check("RES1 after clear", 0, (int32_t)TPU_RES(1));
    check("RES2 after clear", 0, (int32_t)TPU_RES(2));
    check("RES3 after clear", 0, (int32_t)TPU_RES(3));

    // ── Test 2: Identity matrix multiply ─────────────────────────────────
    neorv32_uart0_printf("\nTest 2: Identity weight matrix\n");

    // W = I (identity 4x4)
    //   row0: [1, 0, 0, 0]
    //   row1: [0, 1, 0, 0]
    //   row2: [0, 0, 1, 0]
    //   row3: [0, 0, 0, 1]
    load_weight_row(0, 1, 0, 0, 0);
    load_weight_row(1, 0, 1, 0, 0);
    load_weight_row(2, 0, 0, 1, 0);
    load_weight_row(3, 0, 0, 0, 1);

    clear_acc();
    run_mac(10, 20, 30, 40);

    // Result should be: row[i] = x[i] (identity)
    check("I*[10,20,30,40] row0", 10, (int32_t)TPU_RES(0));
    check("I*[10,20,30,40] row1", 20, (int32_t)TPU_RES(1));
    check("I*[10,20,30,40] row2", 30, (int32_t)TPU_RES(2));
    check("I*[10,20,30,40] row3", 40, (int32_t)TPU_RES(3));

    // ── Test 3: General 4x4 × 4x1 multiply ──────────────────────────────
    neorv32_uart0_printf("\nTest 3: General matrix multiply\n");

    // W = [[1, 2, 3, 4],
    //      [5, 6, 7, 8],
    //      [-1, -2, -3, -4],
    //      [0, 1, 0, -1]]
    load_weight_row(0, 1, 2, 3, 4);
    load_weight_row(1, 5, 6, 7, 8);
    load_weight_row(2, -1, -2, -3, -4);
    load_weight_row(3, 0, 1, 0, -1);

    // x = [1, 2, 3, 4]
    clear_acc();
    run_mac(1, 2, 3, 4);

    // row0 = 1*1 + 2*2 + 3*3 + 4*4 = 1+4+9+16 = 30
    // row1 = 5*1 + 6*2 + 7*3 + 8*4 = 5+12+21+32 = 70
    // row2 = -1*1 + -2*2 + -3*3 + -4*4 = -1-4-9-16 = -30
    // row3 = 0*1 + 1*2 + 0*3 + -1*4 = 0+2+0-4 = -2
    check("W*[1,2,3,4] row0", 30, (int32_t)TPU_RES(0));
    check("W*[1,2,3,4] row1", 70, (int32_t)TPU_RES(1));
    check("W*[1,2,3,4] row2", -30, (int32_t)TPU_RES(2));
    check("W*[1,2,3,4] row3", -2, (int32_t)TPU_RES(3));

    // ── Test 4: Accumulation (two MACs without clear) ────────────────────
    neorv32_uart0_printf("\nTest 4: Accumulation across MACs\n");

    // Same weights as test 3
    clear_acc();

    // First MAC: x = [1, 0, 0, 0]
    run_mac(1, 0, 0, 0);
    // row0 = 1, row1 = 5, row2 = -1, row3 = 0

    // Second MAC: x = [0, 1, 0, 0] (accumulates)
    run_mac(0, 1, 0, 0);
    // row0 += 2, row1 += 6, row2 += -2, row3 += 1
    // row0 = 3, row1 = 11, row2 = -3, row3 = 1

    check("acc row0 (1+2)", 3, (int32_t)TPU_RES(0));
    check("acc row1 (5+6)", 11, (int32_t)TPU_RES(1));
    check("acc row2 (-1+-2)", -3, (int32_t)TPU_RES(2));
    check("acc row3 (0+1)", 1, (int32_t)TPU_RES(3));

    // ── Test 5: Single weight load (non-bulk) ────────────────────────────
    neorv32_uart0_printf("\nTest 5: Single weight load\n");

    // Load all zeros first
    load_weight_row(0, 0, 0, 0, 0);
    load_weight_row(1, 0, 0, 0, 0);
    load_weight_row(2, 0, 0, 0, 0);
    load_weight_row(3, 0, 0, 0, 0);

    // Load just w[1][2] = 127
    load_weight(1, 2, 127);

    clear_acc();
    run_mac(0, 0, 3, 0);  // only x2=3 nonzero

    // Only row1 should have 127*3 = 381
    check("single w[1][2]=127, x2=3, row0", 0, (int32_t)TPU_RES(0));
    check("single w[1][2]=127, x2=3, row1", 381, (int32_t)TPU_RES(1));
    check("single w[1][2]=127, x2=3, row2", 0, (int32_t)TPU_RES(2));
    check("single w[1][2]=127, x2=3, row3", 0, (int32_t)TPU_RES(3));

    // ── Test 6: Negative weight boundary ─────────────────────────────────
    neorv32_uart0_printf("\nTest 6: Signed boundary values\n");

    // W row0 = [-128, 127, -1, 1]
    load_weight_row(0, -128, 127, -1, 1);
    load_weight_row(1, 0, 0, 0, 0);
    load_weight_row(2, 0, 0, 0, 0);
    load_weight_row(3, 0, 0, 0, 0);

    clear_acc();
    run_mac(1, 1, 1, 1);

    // row0 = -128 + 127 + -1 + 1 = -1
    check("boundary row0 (-128+127-1+1)", -1, (int32_t)TPU_RES(0));

    // ── Summary ──────────────────────────────────────────────────────────
    neorv32_uart0_printf("\n=====================================\n");
    neorv32_uart0_printf("  Results: %d/%d passed\n", pass_count, test_count);
    neorv32_uart0_printf("=====================================\n");

    if (pass_count == test_count) {
        neorv32_uart0_printf("  ALL TESTS PASSED!\n");
    } else {
        neorv32_uart0_printf("  SOME TESTS FAILED!\n");
    }

    // Blink LEDs to indicate completion
    uint32_t led = 0xF;
    while (1) {
        neorv32_gpio_port_set(led);
        for (volatile uint32_t d = 0; d < 2500000; d++) { }
        led ^= 0xF;
    }

    return 0;
}
