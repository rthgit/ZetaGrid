#include <iostream>
#include <vector>
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <ap_axi_sdata.h>

// Typedefs copied from kernel.cpp (Quick Fix)
typedef ap_axiu<32, 0, 0, 0> float_pkt;
typedef ap_axiu<128, 0, 0, 0> packed_pkt;

// Prototype
void pack_int8(hls::stream<float_pkt>& in_stream, hls::stream<packed_pkt>& out_stream, int num_elements);

// Software Reference
void ref_pack(const std::vector<float>& in, std::vector<ap_int<128>>& out) {
    int count = 0;
    ap_int<128> buf = 0;
    for(float val : in) {
        int8_t i8 = (int8_t)val;
        int bit_idx = (count % 16) * 8;
        buf.range(bit_idx + 7, bit_idx) = i8;
        
        count++;
        if(count % 16 == 0) {
            out.push_back(buf);
            buf = 0;
        }
    }
}

int main() {
    std::cout << ">>> ZED-HLS TESTBENCH: K1 pack_int8 <<<" << std::endl;

    const int N = 1024; // Test size
    std::vector<float> input_data(N);
    std::vector<ap_int<128>> ref_data;
    
    // 1. Generate Data
    for(int i=0; i<N; i++) {
        input_data[i] = (float)(i % 127); // Simple ramp
        if (i%2==0) input_data[i] *= -1;  // Add negative
    }
    
    // 2. Compute Reference
    ref_pack(input_data, ref_data);
    
    // 3. Prepare HW Streams
    hls::stream<float_pkt> in_stream;
    hls::stream<packed_pkt> out_stream;
    
    for(float f : input_data) {
        float_pkt p;
        p.data = *(int*)&f; // Float treated as bits for transport, or just cast?
        // Note: In kernel.cpp we did `float val = f_p.data`. 
        // Ideally .data is a template type or we need union. 
        // For ap_axiu<32...>, data is ap_uint<32>.
        // We must memcpy/union float to int representation.
        union { float f; uint32_t i; } u;
        u.f = f;
        p.data = u.i;
        in_stream.write(p);
    }
    
    // 4. Run Kernel
    std::cout << "[TB] Running Kernel..." << std::endl;
    pack_int8(in_stream, out_stream, N);
    
    // 5. Verify
    std::cout << "[TB] Verifying Output..." << std::endl;
    bool pass = true;
    for(size_t i=0; i<ref_data.size(); i++) {
        if(out_stream.empty()) {
            std::cout << "[ERROR] Stream empty at index " << i << std::endl;
            pass = false; 
            break;
        }
        packed_pkt p = out_stream.read();
        if(p.data != ref_data[i]) {
            std::cout << "[FAIL] Mismatch at " << i << std::endl;
            std::cout << "  Exp: " << std::hex << ref_data[i] << std::endl;
            std::cout << "  Got: " << std::hex << p.data << std::endl;
            pass = false;
        }
    }
    
    if(!out_stream.empty()) {
         std::cout << "[WARN] Stream has leftover data!" << std::endl;
         pass = false;
    }

    if(pass) {
        std::cout << ">>> TEST PASSED <<<" << std::endl;
        return 0;
    } else {
        std::cout << ">>> TEST FAILED <<<" << std::endl;
        return 1;
    }
}
