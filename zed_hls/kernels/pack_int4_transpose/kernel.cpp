#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <stdint.h>

// Configuration
#define BLOCK_SIZE 32
#define PACK_WIDTH 32 // 32 int4s make one 128-bit pack
#define ROWS 8
#define COLS 4

typedef ap_axiu<32, 0, 0, 0> float_pkt;
typedef ap_axiu<128, 0, 0, 0> packed_pkt;

void pack_int4_transpose(hls::stream<float_pkt>& in_stream, hls::stream<packed_pkt>& out_stream, int num_elements) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=num_elements bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int num_blocks = num_elements / BLOCK_SIZE;

    block_loop: for(int i = 0; i < num_blocks; i++) {
        // We pipeline the outer loop, but we need to manage the read/transpose/write sequence.
        // Reading 32 items takes 32 cycles. Writing takes 1 cycle.
        // Simple pipeline II=32 is acceptable for this streaming architecture.
        
        ap_int<4> buffer[BLOCK_SIZE];
        #pragma HLS ARRAY_PARTITION variable=buffer complete dim=1

        // 1. Read and Quantize (Row-Major Input)
        read_loop: for(int j = 0; j < BLOCK_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            float_pkt f_p = in_stream.read();
            uint32_t raw_bits = (uint32_t)f_p.data;
            float val = *(float*)&raw_bits;

            // Quantization: Float to Int8 then clamp/truncate to Int4
            // Range [-8, 7]
            int8_t i8 = (int8_t)val;
            if (i8 > 7) i8 = 7;
            if (i8 < -8) i8 = -8;
            
            buffer[j] = (ap_int<4>)i8;
        }

        // 2. Transpose and Pack (Column-Major Output)
        // We construct one 128-bit word.
        ap_int<128> out_val = 0;

        // Flattened transposition logic
        // Output index p (0..31) fills the 128-bit word from LSB to MSB
        // p corresponds to traversing the matrix in Column-Major order.
        // p=0 -> (R0, C0), p=1 -> (R1, C0), ... p=7 -> (R7, C0), p=8 -> (R0, C1)...
        
        pack_loop: for(int p = 0; p < BLOCK_SIZE; p++) {
            #pragma HLS UNROLL
            int r = p % ROWS;
            int c = p / ROWS;
            int buf_idx = r * COLS + c; // Input was Row-Major
            
            int start_bit = p * 4;
            int end_bit = start_bit + 3;
            
            out_val.range(end_bit, start_bit) = buffer[buf_idx];
        }

        packed_pkt p_out;
        p_out.data = out_val;
        p_out.keep = -1;
        p_out.last = (i == num_blocks - 1);
        out_stream.write(p_out);
    }
}
