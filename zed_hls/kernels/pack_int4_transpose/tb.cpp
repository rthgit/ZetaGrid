#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

// Typos matching kernel
typedef ap_axiu<32, 0, 0, 0> float_pkt;
typedef ap_axiu<128, 0, 0, 0> packed_pkt;

void pack_int4_transpose(hls::stream<float_pkt>& in_stream, hls::stream<packed_pkt>& out_stream, int num_elements);

#define BLOCK_SIZE 32
#define PACK_WIDTH 32
#define ROWS 8
#define COLS 4

// Software Reference
std::vector<ap_int<128>> ref_pack_transpose(const std::vector<float>& input) {
    std::vector<ap_int<128>> output;
    int num_blocks = input.size() / BLOCK_SIZE;

    for (int i = 0; i < num_blocks; i++) {
        std::vector<ap_int<4>> buffer(BLOCK_SIZE);
        
        // 1. Quantize
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float val = input[i * BLOCK_SIZE + j];
            int8_t i8 = (int8_t)val;
            if (i8 > 7) i8 = 7;
            if (i8 < -8) i8 = -8;
            buffer[j] = (ap_int<4>)i8;
        }

        // 2. Transpose & Pack
        ap_int<128> out_val = 0;
        for (int p = 0; p < BLOCK_SIZE; p++) {
            // Col-major traversal of buffer treated as 8x4 matrix
            int r = p % ROWS;
            int c = p / ROWS;
            int buf_idx = r * COLS + c;

            int start_bit = p * 4;
            int end_bit = start_bit + 3;
            out_val.range(end_bit, start_bit) = buffer[buf_idx];
        }
        output.push_back(out_val);
    }
    return output;
}

int main() {
    int num_elements = 1024; // 32 blocks
    std::vector<float> input_data(num_elements);
    
    // Gen Data
    // Use integer-ish floats to make debugging easier (e.g. 1.0, -3.0)
    for(int i=0; i<num_elements; i++) {
        input_data[i] = (float)((i % 16) - 8); 
    }

    // Run Reference
    std::vector<ap_int<128>> ref_out = ref_pack_transpose(input_data);

    // Prepare Streams
    hls::stream<float_pkt> in_stream;
    hls::stream<packed_pkt> out_stream;

    for(float f : input_data) {
        float_pkt pkt;
        // Float to raw bits
        uint32_t raw = *(uint32_t*)&f;
        pkt.data = raw;
        pkt.keep = -1;
        pkt.last = 0;
        in_stream.write(pkt);
    }

    // Run DUT
    pack_int4_transpose(in_stream, out_stream, num_elements);

    // Verify
    int errors = 0;
    for(int i=0; i<ref_out.size(); i++) {
        if(out_stream.empty()) {
            std::cerr << "Stream empty at index " << i << std::endl;
            errors++;
            break;
        }
        packed_pkt p = out_stream.read();
        if(p.data != ref_out[i]) {
            std::cout << "Mismatch at pack " << i << std::endl;
            std::cout << "Exp: " << std::hex << ref_out[i] << std::endl;
            std::cout << "Got: " << std::hex << p.data << std::dec << std::endl;
            errors++;
        }
    }

    if(errors == 0) {
        std::cout << "Test Passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed with " << errors << " errors." << std::endl;
        return 1;
    }
}
