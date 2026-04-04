// 4x4 Weight-Stationary Systolic Array
//
// Topology:
//            x_in[0]  x_in[1]  x_in[2]  x_in[3]
//              |        |        |        |
//   0 -> PE[0,0] -> PE[0,1] -> PE[0,2] -> PE[0,3] -> result[0]
//              |        |        |        |
//   0 -> PE[1,0] -> PE[1,1] -> PE[1,2] -> PE[1,3] -> result[1]
//              |        |        |        |
//   0 -> PE[2,0] -> PE[2,1] -> PE[2,2] -> PE[2,3] -> result[2]
//              |        |        |        |
//   0 -> PE[3,0] -> PE[3,1] -> PE[3,2] -> PE[3,3] -> result[3]
//
// x flows top-to-bottom (through columns), psum flows left-to-right (through rows).
// Inputs must be skewed: x_in[j] is fed j cycles after x_in[0].
// After feeding 4 skewed inputs, results appear staggered at the right edge:
//   result[0] valid after 4 cycles, result[3] valid after 7 cycles.
//
// Weight loading: serial, addressed by w_row_sel/w_col_sel, 16 cycles total.
//
// Port convention: packed buses instead of unpacked arrays for Verilog-2001 compatibility.
//   x_in:   [31:0]  = {x3[7:0], x2[7:0], x1[7:0], x0[7:0]}
//   result: [127:0] = {r3[31:0], r2[31:0], r1[31:0], r0[31:0]}

module systolic_array_4x4 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        load_weight,
    input  wire [1:0]  w_row_sel,
    input  wire [1:0]  w_col_sel,
    input  wire signed [7:0] w_data,
    input  wire [31:0] x_in,       // packed: {x3, x2, x1, x0} each signed [7:0]
    output wire [127:0] result     // packed: {r3, r2, r1, r0} each signed [31:0]
);

    // Unpack x_in
    wire signed [7:0] x_top [0:3];
    assign x_top[0] = x_in[ 7: 0];
    assign x_top[1] = x_in[15: 8];
    assign x_top[2] = x_in[23:16];
    assign x_top[3] = x_in[31:24];

    // Inter-PE wires: x flows down, psum flows right
    // x_wire[row][col] — 5 rows (0=top input, 1-4=PE outputs), 4 cols
    // psum_wire[row][col] — 4 rows, 5 cols (0=left zero, 1-4=PE outputs)
    wire signed [7:0]  x_w_0_0, x_w_0_1, x_w_0_2, x_w_0_3;
    wire signed [7:0]  x_w_1_0, x_w_1_1, x_w_1_2, x_w_1_3;
    wire signed [7:0]  x_w_2_0, x_w_2_1, x_w_2_2, x_w_2_3;
    wire signed [7:0]  x_w_3_0, x_w_3_1, x_w_3_2, x_w_3_3;
    wire signed [7:0]  x_w_4_0, x_w_4_1, x_w_4_2, x_w_4_3; // bottom (unused)

    wire signed [31:0] p_0_1, p_0_2, p_0_3, p_0_4;
    wire signed [31:0] p_1_1, p_1_2, p_1_3, p_1_4;
    wire signed [31:0] p_2_1, p_2_2, p_2_3, p_2_4;
    wire signed [31:0] p_3_1, p_3_2, p_3_3, p_3_4;

    // Top inputs
    assign x_w_0_0 = x_top[0];
    assign x_w_0_1 = x_top[1];
    assign x_w_0_2 = x_top[2];
    assign x_w_0_3 = x_top[3];

    // Row 0
    pe pe_0_0 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd0 && w_col_sel==2'd0),
        .w_in(w_data), .x_in(x_w_0_0), .psum_in(32'sd0),
        .x_out(x_w_1_0), .psum_out(p_0_1));
    pe pe_0_1 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd0 && w_col_sel==2'd1),
        .w_in(w_data), .x_in(x_w_0_1), .psum_in(p_0_1),
        .x_out(x_w_1_1), .psum_out(p_0_2));
    pe pe_0_2 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd0 && w_col_sel==2'd2),
        .w_in(w_data), .x_in(x_w_0_2), .psum_in(p_0_2),
        .x_out(x_w_1_2), .psum_out(p_0_3));
    pe pe_0_3 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd0 && w_col_sel==2'd3),
        .w_in(w_data), .x_in(x_w_0_3), .psum_in(p_0_3),
        .x_out(x_w_1_3), .psum_out(p_0_4));

    // Row 1
    pe pe_1_0 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd1 && w_col_sel==2'd0),
        .w_in(w_data), .x_in(x_w_1_0), .psum_in(32'sd0),
        .x_out(x_w_2_0), .psum_out(p_1_1));
    pe pe_1_1 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd1 && w_col_sel==2'd1),
        .w_in(w_data), .x_in(x_w_1_1), .psum_in(p_1_1),
        .x_out(x_w_2_1), .psum_out(p_1_2));
    pe pe_1_2 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd1 && w_col_sel==2'd2),
        .w_in(w_data), .x_in(x_w_1_2), .psum_in(p_1_2),
        .x_out(x_w_2_2), .psum_out(p_1_3));
    pe pe_1_3 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd1 && w_col_sel==2'd3),
        .w_in(w_data), .x_in(x_w_1_3), .psum_in(p_1_3),
        .x_out(x_w_2_3), .psum_out(p_1_4));

    // Row 2
    pe pe_2_0 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd2 && w_col_sel==2'd0),
        .w_in(w_data), .x_in(x_w_2_0), .psum_in(32'sd0),
        .x_out(x_w_3_0), .psum_out(p_2_1));
    pe pe_2_1 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd2 && w_col_sel==2'd1),
        .w_in(w_data), .x_in(x_w_2_1), .psum_in(p_2_1),
        .x_out(x_w_3_1), .psum_out(p_2_2));
    pe pe_2_2 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd2 && w_col_sel==2'd2),
        .w_in(w_data), .x_in(x_w_2_2), .psum_in(p_2_2),
        .x_out(x_w_3_2), .psum_out(p_2_3));
    pe pe_2_3 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd2 && w_col_sel==2'd3),
        .w_in(w_data), .x_in(x_w_2_3), .psum_in(p_2_3),
        .x_out(x_w_3_3), .psum_out(p_2_4));

    // Row 3
    pe pe_3_0 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd3 && w_col_sel==2'd0),
        .w_in(w_data), .x_in(x_w_3_0), .psum_in(32'sd0),
        .x_out(x_w_4_0), .psum_out(p_3_1));
    pe pe_3_1 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd3 && w_col_sel==2'd1),
        .w_in(w_data), .x_in(x_w_3_1), .psum_in(p_3_1),
        .x_out(x_w_4_1), .psum_out(p_3_2));
    pe pe_3_2 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd3 && w_col_sel==2'd2),
        .w_in(w_data), .x_in(x_w_3_2), .psum_in(p_3_2),
        .x_out(x_w_4_2), .psum_out(p_3_3));
    pe pe_3_3 (.clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight && w_row_sel==2'd3 && w_col_sel==2'd3),
        .w_in(w_data), .x_in(x_w_3_3), .psum_in(p_3_3),
        .x_out(x_w_4_3), .psum_out(p_3_4));

    // Pack results: {r3, r2, r1, r0}
    assign result = {p_3_4, p_2_4, p_1_4, p_0_4};

endmodule
