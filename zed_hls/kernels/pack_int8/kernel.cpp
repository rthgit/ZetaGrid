#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

// Configuration
#define BATCH_SIZE 1024
#define PACK_WIDTH 16 // 128 bits / 8 bits = 16 elements

// Input: Standard Float Stream
// Output: 128-bit Packed Stream (16 ints per beat)
// We use AXI Stream interface for composability

typedef ap_axiu<32, 0, 0, 0> float_pkt;
typedef ap_axiu<128, 0, 0, 0> packed_pkt;

void pack_int8(hls::stream<float_pkt>& in_stream, hls::stream<packed_pkt>& out_stream, int num_elements) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=num_elements bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int num_packs = num_elements / PACK_WIDTH;

    pack_loop: for(int i = 0; i < num_packs; i++) {
        #pragma HLS PIPELINE II=1
        
        ap_int<128> pack_buffer = 0;

        // "Unroll" logic within the pipeline step? 
        // No, to achieve II=1 we need to consume 16 floats per clock? 
        // That requires Input Width to be 512 bits (16 * 32).
        // If input is 32-bit stream, we naturally have II=16. 
        // Let's assume input is ALSO wide (e.g. 512 bit from memory) or we accept II=16 relative to input.
        
        // MVP: Read 16 floats sequentially, pack, write 1.
        for(int j = 0; j < PACK_WIDTH; j++) {
            float_pkt f_p = in_stream.read();
            // Fix: Bitcast ap_uint<32> to float
            uint32_t raw_bits = (uint32_t)f_p.data;
            float val = *(float*)&raw_bits;
            
            // Simple Quantization: Cast to int8 (truncate)
            // Real Zed would scale here. 
            int8_t i8 = (int8_t)val;
            
            // Insert into pack (Little Endian)
            int start_bit = j * 8;
            int end_bit = start_bit + 7;
            pack_buffer.range(end_bit, start_bit) = i8;
        }

        packed_pkt p_out;
        p_out.data = pack_buffer;
        p_out.keep = -1;
        p_out.last = (i == num_packs - 1);
        out_stream.write(p_out);
    }
}
