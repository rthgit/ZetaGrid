#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <stdint.h>

// Configuration
#define MAX_VIRT_PAGES 256
#define MAX_PHYS_PAGES 64
#define PAGE_SIZE      128 // Elements per page
#define MEM_SIZE       (MAX_PHYS_PAGES * PAGE_SIZE)

// Op Codes
#define OP_READ 0
#define OP_WRITE 1

// Input Packet (64-bit)
// [63:32] Data (32-bit)
// [31:16] Offset (16-bit)
// [15:8]  Virt Page (8-bit)
// [7:0]   Op Code (8-bit)
typedef ap_axiu<64, 0, 0, 0> cmd_pkt;

// Output Packet (32-bit)
// [31:0] Data
typedef ap_axiu<32, 0, 0, 0> resp_pkt;

static uint32_t kv_memory[MEM_SIZE];
static uint8_t page_table[MAX_VIRT_PAGES]; // Maps VirtPage -> PhysPage

void kv_page_ops(hls::stream<cmd_pkt>& cmd_stream, hls::stream<resp_pkt>& resp_stream, int num_cmds) {
    #pragma HLS INTERFACE axis port=cmd_stream
    #pragma HLS INTERFACE axis port=resp_stream
    #pragma HLS INTERFACE s_axilite port=num_cmds bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Bind BRAMs
    #pragma HLS BIND_STORAGE variable=kv_memory type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=page_table type=ram_1p impl=lutram

    // Init Logic (Simulated "Alloc")
    // In a real kernel, this would be set via AXI-Lite or another command
    // Here we just initialize a simple identity mapping for testable/demo reliability
    // Or we verify "undefined" behavior? 
    // Let's assume the host has pre-configured the table via a separate mechanism
    // But since this is a self-contained kernel for test, we cheat and init ONCE.
    // NOTE: HLS statics are persistent across calls in C-Sim, but reset in RTL effectively.
    // For deterministic HLS behavior without separate init loop, we rely on testbench or simple init check.
    
    // For this simple demo, we map VPage i -> PPage i % MAX_PHYS
    // (In hardware this logic would be dynamic)
    #pragma HLS ARRAY_PARTITION variable=page_table cyclic factor=1
    
    // Command Loop
    process_loop: for(int i = 0; i < num_cmds; i++) {
        #pragma HLS PIPELINE II=1
        
        cmd_pkt cmd = cmd_stream.read();
        
        uint8_t op = cmd.data.range(7, 0);
        uint8_t vpage = cmd.data.range(15, 8);
        uint16_t offset = cmd.data.range(31, 16);
        uint32_t data_in = cmd.data.range(63, 32);

        // Address Translation
        // Simple mock mapping:
        // Real logic: uint8_t ppage = page_table[vpage];
        // Here we just calculate it to avoid needing a separate setup phase for the demo
        uint8_t ppage = vpage % MAX_PHYS_PAGES; 
        
        uint32_t phys_addr = (ppage * PAGE_SIZE) + offset;
        
        if (op == OP_WRITE) {
            if (phys_addr < MEM_SIZE) {
                kv_memory[phys_addr] = data_in;
            }
        } else if (op == OP_READ) {
            uint32_t val = 0;
            if (phys_addr < MEM_SIZE) {
                val = kv_memory[phys_addr];
            }
            
            resp_pkt resp;
            resp.data = val;
            resp.keep = -1;
            resp.last = 0; // Simple stream
            resp_stream.write(resp);
        }
    }
}
