-- ax301_top.vhd — NEORV32 + TPU SoC for 黑金 AX301 (EP4CE6F17C8)
-- Phase 2: Linux-capable NEORV32 + 4×4 systolic array TPU
--
-- Memory map:
--   0x00000000 - 0x00001FFF : IMEM  (8 KB, M9K BRAM, stage2 loader)
--   0x80000000 - 0x80001FFF : DMEM  (8 KB, M9K BRAM)
--   0x40000000 - 0x41FFFFFF : SDRAM (32 MB, Linux kernel + data)
--   0xF0000000 - 0xF000003F : TPU   (4×4 systolic array, via Wishbone → tpu_accel)
--   0xFFE00000              : Boot ROM (NEORV32 internal bootloader)
--   0xFFF40000              : CLINT (timer)
--   0xFFF50000              : UART0 (19200 baud bootloader / 115200 baud app)
--   0xFFFC0000              : GPIO  (gpio_o[3:0] → LED active-low)
--
-- XBUS address decode:
--   SDRAM selected when adr[31:25] = "0100000"
--   TPU   selected when adr[31:6]  = x"F00000" & "00"
--   Others → immediate ACK with zero (safe default, handled by wb_sdram_ctrl)

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library neorv32;
use neorv32.neorv32_package.all;

entity ax301_top is
  port (
    CLOCK  : in    std_ulogic;                     -- 50 MHz
    KEY2   : in    std_ulogic;                     -- active-low reset button
    -- UART (PL2303 → /dev/ttyUSB0)
    RXD    : in    std_ulogic;
    TXD    : out   std_ulogic;
    -- LEDs (active-low, directly mapped)
    LED    : out   std_ulogic_vector(3 downto 0);
    -- SDRAM (HY57V2562GTR, 32 MB)
    S_CLK  : out   std_ulogic;
    S_CKE  : out   std_ulogic;
    S_NCS  : out   std_ulogic;
    S_NRAS : out   std_ulogic;
    S_NCAS : out   std_ulogic;
    S_NWE  : out   std_ulogic;
    S_BA   : out   std_ulogic_vector(1 downto 0);
    S_A    : out   std_ulogic_vector(12 downto 0);
    S_DQM  : out   std_ulogic_vector(1 downto 0);
    S_DB   : inout std_ulogic_vector(15 downto 0);
    -- SD card (SPI mode, on-board slot)
    SD_CLK : out   std_ulogic;
    SD_DI  : out   std_ulogic;  -- MOSI (controller → card)
    SD_DO  : in    std_ulogic;  -- MISO (card → controller)
    SD_NCS : out   std_ulogic
  );
end entity ax301_top;

architecture rtl of ax301_top is

  -- ── Internal signals ──────────────────────────────────────────────────────
  signal rstn_int    : std_ulogic;
  signal por_cnt     : std_ulogic_vector(3 downto 0) := (others => '0');
  signal gpio_out    : std_ulogic_vector(31 downto 0);

  -- XBUS (Wishbone) signals from NEORV32
  signal xbus_adr    : std_ulogic_vector(31 downto 0);
  signal xbus_dat_o  : std_ulogic_vector(31 downto 0);
  signal xbus_dat_i  : std_ulogic_vector(31 downto 0);
  signal xbus_we     : std_ulogic;
  signal xbus_sel    : std_ulogic_vector(3 downto 0);
  signal xbus_stb    : std_ulogic;
  signal xbus_cyc    : std_ulogic;
  signal xbus_ack    : std_ulogic;
  signal xbus_err    : std_ulogic;

  -- Address decode signals
  signal sdram_selected : std_ulogic;
  signal tpu_selected   : std_ulogic;

  -- SDRAM bridge signals
  signal sdram_dat_r : std_ulogic_vector(31 downto 0);
  signal sdram_ack   : std_ulogic;
  signal sdram_err   : std_ulogic;
  signal sdram_stb   : std_ulogic;

  -- TPU bridge signals
  signal tpu_dat_r   : std_ulogic_vector(31 downto 0);
  signal tpu_ack     : std_ulogic;
  signal tpu_err     : std_ulogic;
  signal tpu_stb     : std_ulogic;

  -- Debug
  signal sdram_dbg   : std_ulogic_vector(3 downto 0);
  signal tpu_dbg     : std_ulogic_vector(3 downto 0);

  -- ── Verilog component declarations ────────────────────────────────────────

  component wb_sdram_ctrl is
    port (
      clk       : in  std_ulogic;
      rst_n     : in  std_ulogic;
      xbus_adr  : in  std_ulogic_vector(31 downto 0);
      xbus_dat_w: in  std_ulogic_vector(31 downto 0);
      xbus_sel  : in  std_ulogic_vector(3 downto 0);
      xbus_we   : in  std_ulogic;
      xbus_stb  : in  std_ulogic;
      xbus_cyc  : in  std_ulogic;
      xbus_dat_r: out std_ulogic_vector(31 downto 0);
      xbus_ack  : out std_ulogic;
      xbus_err  : out std_ulogic;
      S_CLK     : out std_ulogic;
      S_CKE     : out std_ulogic;
      S_NCS     : out std_ulogic;
      S_NRAS    : out std_ulogic;
      S_NCAS    : out std_ulogic;
      S_NWE     : out std_ulogic;
      S_BA      : out std_ulogic_vector(1 downto 0);
      S_A       : out std_ulogic_vector(12 downto 0);
      S_DQM     : out std_ulogic_vector(1 downto 0);
      S_DB      : inout std_ulogic_vector(15 downto 0);
      dbg_leds  : out std_ulogic_vector(3 downto 0)
    );
  end component;

  component wb_tpu_accel is
    port (
      clk       : in  std_ulogic;
      rst_n     : in  std_ulogic;
      xbus_adr  : in  std_ulogic_vector(31 downto 0);
      xbus_dat_w: in  std_ulogic_vector(31 downto 0);
      xbus_sel  : in  std_ulogic_vector(3 downto 0);
      xbus_we   : in  std_ulogic;
      xbus_stb  : in  std_ulogic;
      xbus_cyc  : in  std_ulogic;
      xbus_dat_r: out std_ulogic_vector(31 downto 0);
      xbus_ack  : out std_ulogic;
      xbus_err  : out std_ulogic;
      dbg_leds  : out std_ulogic_vector(3 downto 0)
    );
  end component;

begin

  -- ── Power-on reset ────────────────────────────────────────────────────────
  por: process(CLOCK)
  begin
    if rising_edge(CLOCK) then
      if KEY2 = '0' then
        por_cnt <= (others => '0');
      elsif por_cnt(3) = '0' then
        por_cnt <= std_ulogic_vector(unsigned(por_cnt) + 1);
      end if;
    end if;
  end process;
  rstn_int <= KEY2 and por_cnt(3);

  -- ── Address decode ────────────────────────────────────────────────────────
  -- SDRAM: 0x40000000–0x41FFFFFF (adr[31:25] = "0100000")
  sdram_selected <= '1' when xbus_adr(31 downto 25) = "0100000" else '0';
  -- TPU:   0xF0000000–0xF000003F (adr[31:6] = 0xF0000000 >> 6)
  tpu_selected   <= '1' when xbus_adr(31 downto 6) = "11110000000000000000000000" else '0';

  -- Gate STB per device
  sdram_stb <= xbus_stb when sdram_selected = '1' else '0';
  tpu_stb   <= xbus_stb when tpu_selected = '1' else '0';

  -- Mux read data and ACK back to NEORV32
  xbus_dat_i <= tpu_dat_r when tpu_ack = '1' else sdram_dat_r;
  xbus_ack   <= sdram_ack or tpu_ack;
  xbus_err   <= sdram_err or tpu_err;

  -- ── LEDs (active-low): show TPU debug ─────────────────────────────────────
  LED(0) <= not tpu_dbg(0);
  LED(1) <= not tpu_dbg(1);
  LED(2) <= not tpu_dbg(2);
  LED(3) <= not tpu_dbg(3);

  -- ── NEORV32 processor ─────────────────────────────────────────────────────
  neorv32_top_inst: neorv32_top
  generic map (
    CLOCK_FREQUENCY  => 50_000_000,
    -- Boot: internal UART bootloader
    BOOT_MODE_SELECT => 0,
    -- ISA: RV32IMAC (Linux needs U-mode, atomics)
    RISCV_ISA_C      => true,
    RISCV_ISA_M      => true,
    RISCV_ISA_U      => true,    -- U-mode for Linux userspace
    RISCV_ISA_Zaamo  => true,    -- atomic AMO instructions
    RISCV_ISA_Zalrsc => true,    -- LR/SC instructions
    RISCV_ISA_Zicntr => true,   -- needed by stage2_loader (neorv32_cpu_get_cycle)
    -- Internal memories (8 KB IMEM for stage2 loader)
    IMEM_EN          => true,
    IMEM_SIZE        => 8*1024,
    DMEM_EN          => true,
    DMEM_SIZE        => 8*1024,
    -- External bus (Wishbone → SDRAM + TPU)
    XBUS_EN          => true,
    XBUS_TIMEOUT     => 4096,
    XBUS_REGSTAGE_EN => false,
    -- Caches (needed for SDRAM execution performance)
    ICACHE_EN        => true,
    CACHE_BLOCK_SIZE => 64,     -- 16 words per cache line
    CACHE_BURSTS_EN  => false,   -- sdram_ctrl does individual word reads
    DCACHE_EN        => false,   -- disabled to save ~300 LEs (data goes direct to SDRAM)
    -- Peripherals
    IO_GPIO_NUM      => 4,
    IO_CLINT_EN      => true,
    IO_UART0_EN      => true,
    IO_UART0_RX_FIFO => 4,      -- 2^4 = 16-entry FIFO for Linux console
    IO_UART0_TX_FIFO => 4,      -- 2^4 = 16-entry FIFO for Linux console
    -- Everything else off
    IO_SPI_EN        => true,
    IO_SDI_EN        => false,
    IO_TWI_EN        => false,
    IO_TWD_EN        => false,
    IO_PWM_NUM       => 0,
    IO_WDT_EN        => false,
    IO_TRNG_EN       => false,
    IO_CFS_EN        => false,
    IO_NEOLED_EN     => false,
    IO_GPTMR_NUM     => 0,
    IO_ONEWIRE_EN    => false,
    IO_DMA_EN        => false,
    IO_SLINK_EN      => false,
    OCD_EN           => false,
    DUAL_CORE_EN     => false
  )
  port map (
    clk_i        => CLOCK,
    rstn_i       => rstn_int,
    -- XBUS
    xbus_adr_o   => xbus_adr,
    xbus_dat_o   => xbus_dat_o,
    xbus_cti_o   => open,
    xbus_tag_o   => open,
    xbus_dat_i   => xbus_dat_i,
    xbus_we_o    => xbus_we,
    xbus_sel_o   => xbus_sel,
    xbus_stb_o   => xbus_stb,
    xbus_cyc_o   => xbus_cyc,
    xbus_ack_i   => xbus_ack,
    xbus_err_i   => xbus_err,
    -- GPIO
    gpio_o       => gpio_out,
    -- UART0
    uart0_txd_o  => TXD,
    uart0_rxd_i  => RXD,
    -- SPI (SD card)
    spi_clk_o    => SD_CLK,
    spi_dat_o    => SD_DI,
    spi_dat_i    => SD_DO,
    spi_csn_o(0) => SD_NCS,
    spi_csn_o(1) => open,
    spi_csn_o(2) => open,
    spi_csn_o(3) => open,
    spi_csn_o(4) => open,
    spi_csn_o(5) => open,
    spi_csn_o(6) => open,
    spi_csn_o(7) => open
  );

  -- ── Wishbone → SDRAM bridge ───────────────────────────────────────────────
  wb_sdram_inst: wb_sdram_ctrl
  port map (
    clk        => CLOCK,
    rst_n      => rstn_int,
    xbus_adr   => xbus_adr,
    xbus_dat_w => xbus_dat_o,
    xbus_sel   => xbus_sel,
    xbus_we    => xbus_we,
    xbus_stb   => sdram_stb,
    xbus_cyc   => xbus_cyc,
    xbus_dat_r => sdram_dat_r,
    xbus_ack   => sdram_ack,
    xbus_err   => sdram_err,
    S_CLK      => S_CLK,
    S_CKE      => S_CKE,
    S_NCS      => S_NCS,
    S_NRAS     => S_NRAS,
    S_NCAS     => S_NCAS,
    S_NWE      => S_NWE,
    S_BA       => S_BA,
    S_A        => S_A,
    S_DQM      => S_DQM,
    S_DB       => S_DB,
    dbg_leds   => sdram_dbg
  );

  -- ── Wishbone → TPU bridge ─────────────────────────────────────────────────
  wb_tpu_inst: wb_tpu_accel
  port map (
    clk        => CLOCK,
    rst_n      => rstn_int,
    xbus_adr   => xbus_adr,
    xbus_dat_w => xbus_dat_o,
    xbus_sel   => xbus_sel,
    xbus_we    => xbus_we,
    xbus_stb   => tpu_stb,
    xbus_cyc   => xbus_cyc,
    xbus_dat_r => tpu_dat_r,
    xbus_ack   => tpu_ack,
    xbus_err   => tpu_err,
    dbg_leds   => tpu_dbg
  );

end architecture rtl;
