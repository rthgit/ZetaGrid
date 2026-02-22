set moduleName pack_int4_transpose_Pipeline_read_loop
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {pack_int4_transpose_Pipeline_read_loop}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ in_stream_V_data_V int 32 regular {axi_s 0 volatile  { in_stream Data } }  }
	{ in_stream_V_keep_V int 4 regular {axi_s 0 volatile  { in_stream Keep } }  }
	{ in_stream_V_strb_V int 4 regular {axi_s 0 volatile  { in_stream Strb } }  }
	{ in_stream_V_last_V int 1 regular {axi_s 0 volatile  { in_stream Last } }  }
	{ buffer_31_out int 4 regular {pointer 1}  }
	{ buffer_30_out int 4 regular {pointer 1}  }
	{ buffer_29_out int 4 regular {pointer 1}  }
	{ buffer_28_out int 4 regular {pointer 1}  }
	{ buffer_27_out int 4 regular {pointer 1}  }
	{ buffer_26_out int 4 regular {pointer 1}  }
	{ buffer_25_out int 4 regular {pointer 1}  }
	{ buffer_24_out int 4 regular {pointer 1}  }
	{ buffer_23_out int 4 regular {pointer 1}  }
	{ buffer_22_out int 4 regular {pointer 1}  }
	{ buffer_21_out int 4 regular {pointer 1}  }
	{ buffer_20_out int 4 regular {pointer 1}  }
	{ buffer_19_out int 4 regular {pointer 1}  }
	{ buffer_18_out int 4 regular {pointer 1}  }
	{ buffer_17_out int 4 regular {pointer 1}  }
	{ buffer_16_out int 4 regular {pointer 1}  }
	{ buffer_15_out int 4 regular {pointer 1}  }
	{ buffer_14_out int 4 regular {pointer 1}  }
	{ buffer_13_out int 4 regular {pointer 1}  }
	{ buffer_12_out int 4 regular {pointer 1}  }
	{ buffer_11_out int 4 regular {pointer 1}  }
	{ buffer_10_out int 4 regular {pointer 1}  }
	{ buffer_9_out int 4 regular {pointer 1}  }
	{ buffer_8_out int 4 regular {pointer 1}  }
	{ buffer_7_out int 4 regular {pointer 1}  }
	{ buffer_6_out int 4 regular {pointer 1}  }
	{ buffer_5_out int 4 regular {pointer 1}  }
	{ buffer_4_out int 4 regular {pointer 1}  }
	{ buffer_3_out int 4 regular {pointer 1}  }
	{ buffer_2_out int 4 regular {pointer 1}  }
	{ buffer_1_out int 4 regular {pointer 1}  }
	{ buffer_out int 4 regular {pointer 1}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "in_stream_V_data_V", "interface" : "axis", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "in_stream_V_keep_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "in_stream_V_strb_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "in_stream_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "buffer_31_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_30_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_29_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_28_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_27_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_26_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_25_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_24_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_23_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_22_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_21_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_20_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_19_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_18_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_17_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_16_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_15_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_14_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_13_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_12_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_11_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_10_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_9_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_8_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_7_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_6_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_5_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_4_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_3_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_2_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_1_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "buffer_out", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 76
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ in_stream_TVALID sc_in sc_logic 1 invld 0 } 
	{ in_stream_TDATA sc_in sc_lv 32 signal 0 } 
	{ in_stream_TREADY sc_out sc_logic 1 inacc 3 } 
	{ in_stream_TKEEP sc_in sc_lv 4 signal 1 } 
	{ in_stream_TSTRB sc_in sc_lv 4 signal 2 } 
	{ in_stream_TLAST sc_in sc_lv 1 signal 3 } 
	{ buffer_31_out sc_out sc_lv 4 signal 4 } 
	{ buffer_31_out_ap_vld sc_out sc_logic 1 outvld 4 } 
	{ buffer_30_out sc_out sc_lv 4 signal 5 } 
	{ buffer_30_out_ap_vld sc_out sc_logic 1 outvld 5 } 
	{ buffer_29_out sc_out sc_lv 4 signal 6 } 
	{ buffer_29_out_ap_vld sc_out sc_logic 1 outvld 6 } 
	{ buffer_28_out sc_out sc_lv 4 signal 7 } 
	{ buffer_28_out_ap_vld sc_out sc_logic 1 outvld 7 } 
	{ buffer_27_out sc_out sc_lv 4 signal 8 } 
	{ buffer_27_out_ap_vld sc_out sc_logic 1 outvld 8 } 
	{ buffer_26_out sc_out sc_lv 4 signal 9 } 
	{ buffer_26_out_ap_vld sc_out sc_logic 1 outvld 9 } 
	{ buffer_25_out sc_out sc_lv 4 signal 10 } 
	{ buffer_25_out_ap_vld sc_out sc_logic 1 outvld 10 } 
	{ buffer_24_out sc_out sc_lv 4 signal 11 } 
	{ buffer_24_out_ap_vld sc_out sc_logic 1 outvld 11 } 
	{ buffer_23_out sc_out sc_lv 4 signal 12 } 
	{ buffer_23_out_ap_vld sc_out sc_logic 1 outvld 12 } 
	{ buffer_22_out sc_out sc_lv 4 signal 13 } 
	{ buffer_22_out_ap_vld sc_out sc_logic 1 outvld 13 } 
	{ buffer_21_out sc_out sc_lv 4 signal 14 } 
	{ buffer_21_out_ap_vld sc_out sc_logic 1 outvld 14 } 
	{ buffer_20_out sc_out sc_lv 4 signal 15 } 
	{ buffer_20_out_ap_vld sc_out sc_logic 1 outvld 15 } 
	{ buffer_19_out sc_out sc_lv 4 signal 16 } 
	{ buffer_19_out_ap_vld sc_out sc_logic 1 outvld 16 } 
	{ buffer_18_out sc_out sc_lv 4 signal 17 } 
	{ buffer_18_out_ap_vld sc_out sc_logic 1 outvld 17 } 
	{ buffer_17_out sc_out sc_lv 4 signal 18 } 
	{ buffer_17_out_ap_vld sc_out sc_logic 1 outvld 18 } 
	{ buffer_16_out sc_out sc_lv 4 signal 19 } 
	{ buffer_16_out_ap_vld sc_out sc_logic 1 outvld 19 } 
	{ buffer_15_out sc_out sc_lv 4 signal 20 } 
	{ buffer_15_out_ap_vld sc_out sc_logic 1 outvld 20 } 
	{ buffer_14_out sc_out sc_lv 4 signal 21 } 
	{ buffer_14_out_ap_vld sc_out sc_logic 1 outvld 21 } 
	{ buffer_13_out sc_out sc_lv 4 signal 22 } 
	{ buffer_13_out_ap_vld sc_out sc_logic 1 outvld 22 } 
	{ buffer_12_out sc_out sc_lv 4 signal 23 } 
	{ buffer_12_out_ap_vld sc_out sc_logic 1 outvld 23 } 
	{ buffer_11_out sc_out sc_lv 4 signal 24 } 
	{ buffer_11_out_ap_vld sc_out sc_logic 1 outvld 24 } 
	{ buffer_10_out sc_out sc_lv 4 signal 25 } 
	{ buffer_10_out_ap_vld sc_out sc_logic 1 outvld 25 } 
	{ buffer_9_out sc_out sc_lv 4 signal 26 } 
	{ buffer_9_out_ap_vld sc_out sc_logic 1 outvld 26 } 
	{ buffer_8_out sc_out sc_lv 4 signal 27 } 
	{ buffer_8_out_ap_vld sc_out sc_logic 1 outvld 27 } 
	{ buffer_7_out sc_out sc_lv 4 signal 28 } 
	{ buffer_7_out_ap_vld sc_out sc_logic 1 outvld 28 } 
	{ buffer_6_out sc_out sc_lv 4 signal 29 } 
	{ buffer_6_out_ap_vld sc_out sc_logic 1 outvld 29 } 
	{ buffer_5_out sc_out sc_lv 4 signal 30 } 
	{ buffer_5_out_ap_vld sc_out sc_logic 1 outvld 30 } 
	{ buffer_4_out sc_out sc_lv 4 signal 31 } 
	{ buffer_4_out_ap_vld sc_out sc_logic 1 outvld 31 } 
	{ buffer_3_out sc_out sc_lv 4 signal 32 } 
	{ buffer_3_out_ap_vld sc_out sc_logic 1 outvld 32 } 
	{ buffer_2_out sc_out sc_lv 4 signal 33 } 
	{ buffer_2_out_ap_vld sc_out sc_logic 1 outvld 33 } 
	{ buffer_1_out sc_out sc_lv 4 signal 34 } 
	{ buffer_1_out_ap_vld sc_out sc_logic 1 outvld 34 } 
	{ buffer_out sc_out sc_lv 4 signal 35 } 
	{ buffer_out_ap_vld sc_out sc_logic 1 outvld 35 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "in_stream_TVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "in_stream_V_data_V", "role": "default" }} , 
 	{ "name": "in_stream_TDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "in_stream_V_data_V", "role": "default" }} , 
 	{ "name": "in_stream_TREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "inacc", "bundle":{"name": "in_stream_V_last_V", "role": "default" }} , 
 	{ "name": "in_stream_TKEEP", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "in_stream_V_keep_V", "role": "default" }} , 
 	{ "name": "in_stream_TSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "in_stream_V_strb_V", "role": "default" }} , 
 	{ "name": "in_stream_TLAST", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "in_stream_V_last_V", "role": "default" }} , 
 	{ "name": "buffer_31_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_31_out", "role": "default" }} , 
 	{ "name": "buffer_31_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_31_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_30_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_30_out", "role": "default" }} , 
 	{ "name": "buffer_30_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_30_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_29_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_29_out", "role": "default" }} , 
 	{ "name": "buffer_29_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_29_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_28_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_28_out", "role": "default" }} , 
 	{ "name": "buffer_28_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_28_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_27_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_27_out", "role": "default" }} , 
 	{ "name": "buffer_27_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_27_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_26_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_26_out", "role": "default" }} , 
 	{ "name": "buffer_26_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_26_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_25_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_25_out", "role": "default" }} , 
 	{ "name": "buffer_25_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_25_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_24_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_24_out", "role": "default" }} , 
 	{ "name": "buffer_24_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_24_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_23_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_23_out", "role": "default" }} , 
 	{ "name": "buffer_23_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_23_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_22_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_22_out", "role": "default" }} , 
 	{ "name": "buffer_22_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_22_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_21_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_21_out", "role": "default" }} , 
 	{ "name": "buffer_21_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_21_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_20_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_20_out", "role": "default" }} , 
 	{ "name": "buffer_20_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_20_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_19_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_19_out", "role": "default" }} , 
 	{ "name": "buffer_19_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_19_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_18_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_18_out", "role": "default" }} , 
 	{ "name": "buffer_18_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_18_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_17_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_17_out", "role": "default" }} , 
 	{ "name": "buffer_17_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_17_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_16_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_16_out", "role": "default" }} , 
 	{ "name": "buffer_16_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_16_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_15_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_15_out", "role": "default" }} , 
 	{ "name": "buffer_15_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_15_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_14_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_14_out", "role": "default" }} , 
 	{ "name": "buffer_14_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_14_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_13_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_13_out", "role": "default" }} , 
 	{ "name": "buffer_13_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_13_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_12_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_12_out", "role": "default" }} , 
 	{ "name": "buffer_12_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_12_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_11_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_11_out", "role": "default" }} , 
 	{ "name": "buffer_11_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_11_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_10_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_10_out", "role": "default" }} , 
 	{ "name": "buffer_10_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_10_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_9_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_9_out", "role": "default" }} , 
 	{ "name": "buffer_9_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_9_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_8_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_8_out", "role": "default" }} , 
 	{ "name": "buffer_8_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_8_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_7_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_7_out", "role": "default" }} , 
 	{ "name": "buffer_7_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_7_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_6_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_6_out", "role": "default" }} , 
 	{ "name": "buffer_6_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_6_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_5_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_5_out", "role": "default" }} , 
 	{ "name": "buffer_5_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_5_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_4_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_4_out", "role": "default" }} , 
 	{ "name": "buffer_4_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_4_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_3_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_3_out", "role": "default" }} , 
 	{ "name": "buffer_3_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_3_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_2_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_2_out", "role": "default" }} , 
 	{ "name": "buffer_2_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_2_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_1_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_1_out", "role": "default" }} , 
 	{ "name": "buffer_1_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_1_out", "role": "ap_vld" }} , 
 	{ "name": "buffer_out", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "buffer_out", "role": "default" }} , 
 	{ "name": "buffer_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "buffer_out", "role": "ap_vld" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
		"CDFG" : "pack_int4_transpose_Pipeline_read_loop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "35", "EstimateLatencyMax" : "35",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "in_stream_V_data_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "in_stream",
				"BlockSignal" : [
					{"Name" : "in_stream_TDATA_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "in_stream_V_keep_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "in_stream"},
			{"Name" : "in_stream_V_strb_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "in_stream"},
			{"Name" : "in_stream_V_last_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "in_stream"},
			{"Name" : "buffer_31_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_30_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_29_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_28_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_27_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_26_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_25_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_24_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_23_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_22_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_21_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_20_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_19_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_18_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_17_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_16_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_15_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_14_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_13_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_12_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_11_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_10_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_9_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_8_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_7_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_6_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_5_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_4_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_3_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_2_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_1_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "buffer_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "read_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter2", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter2", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	pack_int4_transpose_Pipeline_read_loop {
		in_stream_V_data_V {Type I LastRead 0 FirstWrite -1}
		in_stream_V_keep_V {Type I LastRead 0 FirstWrite -1}
		in_stream_V_strb_V {Type I LastRead 0 FirstWrite -1}
		in_stream_V_last_V {Type I LastRead 0 FirstWrite -1}
		buffer_31_out {Type O LastRead -1 FirstWrite 1}
		buffer_30_out {Type O LastRead -1 FirstWrite 1}
		buffer_29_out {Type O LastRead -1 FirstWrite 1}
		buffer_28_out {Type O LastRead -1 FirstWrite 1}
		buffer_27_out {Type O LastRead -1 FirstWrite 1}
		buffer_26_out {Type O LastRead -1 FirstWrite 1}
		buffer_25_out {Type O LastRead -1 FirstWrite 1}
		buffer_24_out {Type O LastRead -1 FirstWrite 1}
		buffer_23_out {Type O LastRead -1 FirstWrite 1}
		buffer_22_out {Type O LastRead -1 FirstWrite 1}
		buffer_21_out {Type O LastRead -1 FirstWrite 1}
		buffer_20_out {Type O LastRead -1 FirstWrite 1}
		buffer_19_out {Type O LastRead -1 FirstWrite 1}
		buffer_18_out {Type O LastRead -1 FirstWrite 1}
		buffer_17_out {Type O LastRead -1 FirstWrite 1}
		buffer_16_out {Type O LastRead -1 FirstWrite 1}
		buffer_15_out {Type O LastRead -1 FirstWrite 1}
		buffer_14_out {Type O LastRead -1 FirstWrite 1}
		buffer_13_out {Type O LastRead -1 FirstWrite 1}
		buffer_12_out {Type O LastRead -1 FirstWrite 1}
		buffer_11_out {Type O LastRead -1 FirstWrite 1}
		buffer_10_out {Type O LastRead -1 FirstWrite 1}
		buffer_9_out {Type O LastRead -1 FirstWrite 1}
		buffer_8_out {Type O LastRead -1 FirstWrite 1}
		buffer_7_out {Type O LastRead -1 FirstWrite 1}
		buffer_6_out {Type O LastRead -1 FirstWrite 1}
		buffer_5_out {Type O LastRead -1 FirstWrite 1}
		buffer_4_out {Type O LastRead -1 FirstWrite 1}
		buffer_3_out {Type O LastRead -1 FirstWrite 1}
		buffer_2_out {Type O LastRead -1 FirstWrite 1}
		buffer_1_out {Type O LastRead -1 FirstWrite 1}
		buffer_out {Type O LastRead -1 FirstWrite 1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "35", "Max" : "35"}
	, {"Name" : "Interval", "Min" : "35", "Max" : "35"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	in_stream_V_data_V { axis {  { in_stream_TVALID in_vld 0 1 }  { in_stream_TDATA in_data 0 32 } } }
	in_stream_V_keep_V { axis {  { in_stream_TKEEP in_data 0 4 } } }
	in_stream_V_strb_V { axis {  { in_stream_TSTRB in_data 0 4 } } }
	in_stream_V_last_V { axis {  { in_stream_TREADY in_acc 1 1 }  { in_stream_TLAST in_data 0 1 } } }
	buffer_31_out { ap_vld {  { buffer_31_out out_data 1 4 }  { buffer_31_out_ap_vld out_vld 1 1 } } }
	buffer_30_out { ap_vld {  { buffer_30_out out_data 1 4 }  { buffer_30_out_ap_vld out_vld 1 1 } } }
	buffer_29_out { ap_vld {  { buffer_29_out out_data 1 4 }  { buffer_29_out_ap_vld out_vld 1 1 } } }
	buffer_28_out { ap_vld {  { buffer_28_out out_data 1 4 }  { buffer_28_out_ap_vld out_vld 1 1 } } }
	buffer_27_out { ap_vld {  { buffer_27_out out_data 1 4 }  { buffer_27_out_ap_vld out_vld 1 1 } } }
	buffer_26_out { ap_vld {  { buffer_26_out out_data 1 4 }  { buffer_26_out_ap_vld out_vld 1 1 } } }
	buffer_25_out { ap_vld {  { buffer_25_out out_data 1 4 }  { buffer_25_out_ap_vld out_vld 1 1 } } }
	buffer_24_out { ap_vld {  { buffer_24_out out_data 1 4 }  { buffer_24_out_ap_vld out_vld 1 1 } } }
	buffer_23_out { ap_vld {  { buffer_23_out out_data 1 4 }  { buffer_23_out_ap_vld out_vld 1 1 } } }
	buffer_22_out { ap_vld {  { buffer_22_out out_data 1 4 }  { buffer_22_out_ap_vld out_vld 1 1 } } }
	buffer_21_out { ap_vld {  { buffer_21_out out_data 1 4 }  { buffer_21_out_ap_vld out_vld 1 1 } } }
	buffer_20_out { ap_vld {  { buffer_20_out out_data 1 4 }  { buffer_20_out_ap_vld out_vld 1 1 } } }
	buffer_19_out { ap_vld {  { buffer_19_out out_data 1 4 }  { buffer_19_out_ap_vld out_vld 1 1 } } }
	buffer_18_out { ap_vld {  { buffer_18_out out_data 1 4 }  { buffer_18_out_ap_vld out_vld 1 1 } } }
	buffer_17_out { ap_vld {  { buffer_17_out out_data 1 4 }  { buffer_17_out_ap_vld out_vld 1 1 } } }
	buffer_16_out { ap_vld {  { buffer_16_out out_data 1 4 }  { buffer_16_out_ap_vld out_vld 1 1 } } }
	buffer_15_out { ap_vld {  { buffer_15_out out_data 1 4 }  { buffer_15_out_ap_vld out_vld 1 1 } } }
	buffer_14_out { ap_vld {  { buffer_14_out out_data 1 4 }  { buffer_14_out_ap_vld out_vld 1 1 } } }
	buffer_13_out { ap_vld {  { buffer_13_out out_data 1 4 }  { buffer_13_out_ap_vld out_vld 1 1 } } }
	buffer_12_out { ap_vld {  { buffer_12_out out_data 1 4 }  { buffer_12_out_ap_vld out_vld 1 1 } } }
	buffer_11_out { ap_vld {  { buffer_11_out out_data 1 4 }  { buffer_11_out_ap_vld out_vld 1 1 } } }
	buffer_10_out { ap_vld {  { buffer_10_out out_data 1 4 }  { buffer_10_out_ap_vld out_vld 1 1 } } }
	buffer_9_out { ap_vld {  { buffer_9_out out_data 1 4 }  { buffer_9_out_ap_vld out_vld 1 1 } } }
	buffer_8_out { ap_vld {  { buffer_8_out out_data 1 4 }  { buffer_8_out_ap_vld out_vld 1 1 } } }
	buffer_7_out { ap_vld {  { buffer_7_out out_data 1 4 }  { buffer_7_out_ap_vld out_vld 1 1 } } }
	buffer_6_out { ap_vld {  { buffer_6_out out_data 1 4 }  { buffer_6_out_ap_vld out_vld 1 1 } } }
	buffer_5_out { ap_vld {  { buffer_5_out out_data 1 4 }  { buffer_5_out_ap_vld out_vld 1 1 } } }
	buffer_4_out { ap_vld {  { buffer_4_out out_data 1 4 }  { buffer_4_out_ap_vld out_vld 1 1 } } }
	buffer_3_out { ap_vld {  { buffer_3_out out_data 1 4 }  { buffer_3_out_ap_vld out_vld 1 1 } } }
	buffer_2_out { ap_vld {  { buffer_2_out out_data 1 4 }  { buffer_2_out_ap_vld out_vld 1 1 } } }
	buffer_1_out { ap_vld {  { buffer_1_out out_data 1 4 }  { buffer_1_out_ap_vld out_vld 1 1 } } }
	buffer_out { ap_vld {  { buffer_out out_data 1 4 }  { buffer_out_ap_vld out_vld 1 1 } } }
}
