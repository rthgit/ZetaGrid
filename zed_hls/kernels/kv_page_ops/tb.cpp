#include <iostream>
#include <vector>
#include <random>
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

// Types matching kernel
typedef ap_axiu<64, 0, 0, 0> cmd_pkt;
typedef ap_axiu<32, 0, 0, 0> resp_pkt;

void kv_page_ops(hls::stream<cmd_pkt>& cmd_stream, hls::stream<resp_pkt>& resp_stream, int num_cmds);

#define OP_READ 0
#define OP_WRITE 1

// Helper to bundle command
cmd_pkt make_cmd(uint8_t op, uint8_t vpage, uint16_t offset, uint32_t data) {
    cmd_pkt pkt;
    ap_uint<64> raw = 0;
    raw.range(7, 0) = op;
    raw.range(15, 8) = vpage;
    raw.range(31, 16) = offset;
    raw.range(63, 32) = data;
    pkt.data = raw;
    pkt.keep = -1;
    pkt.last = 0;
    return pkt;
}

int main() {
    int num_cmds = 200; // 100 writes, 100 reads
    hls::stream<cmd_pkt> cmd_stream;
    hls::stream<resp_pkt> resp_stream;
    
    std::cout << "Starting KV Page Ops Test..." << std::endl;

    // 1. Generate Data
    std::vector<uint32_t> ref_data(100);
    for(int i=0; i<100; i++) ref_data[i] = i * 100 + 0xABC;

    // 2. Stream Writes
    // Write to VPage 1, Offsets 0..99
    // Logic in kernel maps VPage 1 -> PPage (1 % MAX) = 1
    for(int i=0; i<100; i++) {
        cmd_stream.write(make_cmd(OP_WRITE, 1, i, ref_data[i]));
    }

    // 3. Stream Reads
    // Read back same locations
    for(int i=0; i<100; i++) {
        cmd_stream.write(make_cmd(OP_READ, 1, i, 0));
    }

    // 4. Run Kernel
    kv_page_ops(cmd_stream, resp_stream, num_cmds);

    // 5. Verify Reads
    int errors = 0;
    for(int i=0; i<100; i++) {
        if(resp_stream.empty()) {
            std::cerr << "Stream empty at read " << i << std::endl;
            errors++;
            break;
        }
        resp_pkt resp = resp_stream.read();
        
        if (resp.data != ref_data[i]) {
            std::cout << "Mismatch at " << i << " Exp: " << ref_data[i] << " Got: " << resp.data << std::endl;
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
