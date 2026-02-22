#ifndef ZED_TYPES_H
#define ZED_TYPES_H

#include <stdint.h>

// When running in Vitis HLS, we use the arbitrary precision types
#ifdef __SYNTHESIS__
    #include "ap_int.h"
    typedef ap_int<8>  int8_zed;
    typedef ap_uint<8> uint8_zed;
    typedef ap_int<4>  int4_zed;
    typedef ap_uint<512> bit512;
    typedef ap_uint<256> bit256;
    typedef ap_uint<128> bit128;
#else
// When running in standard C++ (Testbench), we simulate them or use standard types
// For simplicity in TB, we often use structs or std types. 
// Ideally we should link against HLS include dirs for ap_int ref.
// For now, we assume simple standard types for SW simulation of non-bit-exact ops
    typedef int8_t  int8_zed;
    typedef uint8_t uint8_zed;
    // Note: int4 simulation in C++ requires bit masking
    struct int4_zed {
        int8_t val; 
    };
    // Large bits usually need a struct or array
    struct bit512 { uint64_t w[8]; };
    struct bit256 { uint64_t w[4]; };
    struct bit128 { uint64_t w[2]; };
#endif

#define ZED_BLOCK_SIZE 64

#endif
