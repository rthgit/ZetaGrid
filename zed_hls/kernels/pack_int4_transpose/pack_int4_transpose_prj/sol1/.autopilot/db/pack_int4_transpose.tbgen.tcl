set moduleName pack_int4_transpose
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {pack_int4_transpose}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ in_stream_V_data_V int 32 regular {axi_s 0 volatile  { in_stream Data } }  }
	{ in_stream_V_keep_V int 4 regular {axi_s 0 volatile  { in_stream Keep } }  }
	{ in_stream_V_strb_V int 4 regular {axi_s 0 volatile  { in_stream Strb } }  }
	{ in_stream_V_last_V int 1 regular {axi_s 0 volatile  { in_stream Last } }  }
	{ out_stream_V_data_V int 128 regular {axi_s 1 volatile  { out_stream Data } }  }
	{ out_stream_V_keep_V int 16 regular {axi_s 1 volatile  { out_stream Keep } }  }
	{ out_stream_V_strb_V int 16 regular {axi_s 1 volatile  { out_stream Strb } }  }
	{ out_stream_V_last_V int 1 regular {axi_s 1 volatile  { out_stream Last } }  }
	{ num_elements int 32 regular {axi_slave 0}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "in_stream_V_data_V", "interface" : "axis", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "in_stream_V_keep_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "in_stream_V_strb_V", "interface" : "axis", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "in_stream_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "out_stream_V_data_V", "interface" : "axis", "bitwidth" : 128, "direction" : "WRITEONLY"} , 
 	{ "Name" : "out_stream_V_keep_V", "interface" : "axis", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "out_stream_V_strb_V", "interface" : "axis", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "out_stream_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "num_elements", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} ]}
# RTL Port declarations: 
set portNum 32
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ in_stream_TDATA sc_in sc_lv 32 signal 0 } 
	{ in_stream_TVALID sc_in sc_logic 1 invld 3 } 
	{ in_stream_TREADY sc_out sc_logic 1 inacc 3 } 
	{ in_stream_TKEEP sc_in sc_lv 4 signal 1 } 
	{ in_stream_TSTRB sc_in sc_lv 4 signal 2 } 
	{ in_stream_TLAST sc_in sc_lv 1 signal 3 } 
	{ out_stream_TDATA sc_out sc_lv 128 signal 4 } 
	{ out_stream_TVALID sc_out sc_logic 1 outvld 7 } 
	{ out_stream_TREADY sc_in sc_logic 1 outacc 7 } 
	{ out_stream_TKEEP sc_out sc_lv 16 signal 5 } 
	{ out_stream_TSTRB sc_out sc_lv 16 signal 6 } 
	{ out_stream_TLAST sc_out sc_lv 1 signal 7 } 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 5 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 5 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"pack_int4_transpose","role":"start","value":"0","valid_bit":"0"},{"name":"pack_int4_transpose","role":"continue","value":"0","valid_bit":"4"},{"name":"pack_int4_transpose","role":"auto_start","value":"0","valid_bit":"7"},{"name":"num_elements","role":"data","value":"16"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"pack_int4_transpose","role":"start","value":"0","valid_bit":"0"},{"name":"pack_int4_transpose","role":"done","value":"0","valid_bit":"1"},{"name":"pack_int4_transpose","role":"idle","value":"0","valid_bit":"2"},{"name":"pack_int4_transpose","role":"ready","value":"0","valid_bit":"3"},{"name":"pack_int4_transpose","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "in_stream_TDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "in_stream_V_data_V", "role": "default" }} , 
 	{ "name": "in_stream_TVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "in_stream_V_last_V", "role": "default" }} , 
 	{ "name": "in_stream_TREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "inacc", "bundle":{"name": "in_stream_V_last_V", "role": "default" }} , 
 	{ "name": "in_stream_TKEEP", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "in_stream_V_keep_V", "role": "default" }} , 
 	{ "name": "in_stream_TSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "in_stream_V_strb_V", "role": "default" }} , 
 	{ "name": "in_stream_TLAST", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "in_stream_V_last_V", "role": "default" }} , 
 	{ "name": "out_stream_TDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "out_stream_V_data_V", "role": "default" }} , 
 	{ "name": "out_stream_TVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "out_stream_V_last_V", "role": "default" }} , 
 	{ "name": "out_stream_TREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "outacc", "bundle":{"name": "out_stream_V_last_V", "role": "default" }} , 
 	{ "name": "out_stream_TKEEP", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "out_stream_V_keep_V", "role": "default" }} , 
 	{ "name": "out_stream_TSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "out_stream_V_strb_V", "role": "default" }} , 
 	{ "name": "out_stream_TLAST", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "out_stream_V_last_V", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
		"CDFG" : "pack_int4_transpose",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2", "EstimateLatencyMax" : "2617245659",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "in_stream_V_data_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "in_stream",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_pack_int4_transpose_Pipeline_read_loop_fu_254", "Port" : "in_stream_V_data_V", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "in_stream_V_keep_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "in_stream",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_pack_int4_transpose_Pipeline_read_loop_fu_254", "Port" : "in_stream_V_keep_V", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "in_stream_V_strb_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "in_stream",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_pack_int4_transpose_Pipeline_read_loop_fu_254", "Port" : "in_stream_V_strb_V", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "in_stream_V_last_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "in_stream",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_pack_int4_transpose_Pipeline_read_loop_fu_254", "Port" : "in_stream_V_last_V", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "out_stream_V_data_V", "Type" : "Axis", "Direction" : "O", "BaseName" : "out_stream",
				"BlockSignal" : [
					{"Name" : "out_stream_TDATA_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "out_stream_V_keep_V", "Type" : "Axis", "Direction" : "O", "BaseName" : "out_stream"},
			{"Name" : "out_stream_V_strb_V", "Type" : "Axis", "Direction" : "O", "BaseName" : "out_stream"},
			{"Name" : "out_stream_V_last_V", "Type" : "Axis", "Direction" : "O", "BaseName" : "out_stream"},
			{"Name" : "num_elements", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "block_loop", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "6", "FirstState" : "ap_ST_fsm_state3", "LastState" : ["ap_ST_fsm_state6"], "QuitState" : ["ap_ST_fsm_state3"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_pack_int4_transpose_Pipeline_read_loop_fu_254", "Parent" : "0", "Child" : ["2"],
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
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_pack_int4_transpose_Pipeline_read_loop_fu_254.flow_control_loop_pipe_sequential_init_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.control_s_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_in_stream_V_data_V_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_in_stream_V_keep_V_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_in_stream_V_strb_V_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_in_stream_V_last_V_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_out_stream_V_data_V_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_out_stream_V_keep_V_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_out_stream_V_strb_V_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_out_stream_V_last_V_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	pack_int4_transpose {
		in_stream_V_data_V {Type I LastRead 0 FirstWrite -1}
		in_stream_V_keep_V {Type I LastRead 0 FirstWrite -1}
		in_stream_V_strb_V {Type I LastRead 0 FirstWrite -1}
		in_stream_V_last_V {Type I LastRead 0 FirstWrite -1}
		out_stream_V_data_V {Type O LastRead -1 FirstWrite 4}
		out_stream_V_keep_V {Type O LastRead -1 FirstWrite 4}
		out_stream_V_strb_V {Type O LastRead -1 FirstWrite 4}
		out_stream_V_last_V {Type O LastRead -1 FirstWrite 4}
		num_elements {Type I LastRead 0 FirstWrite -1}}
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
	{"Name" : "Latency", "Min" : "2", "Max" : "2617245659"}
	, {"Name" : "Interval", "Min" : "3", "Max" : "-1677721636"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	in_stream_V_data_V { axis {  { in_stream_TDATA in_data 0 32 } } }
	in_stream_V_keep_V { axis {  { in_stream_TKEEP in_data 0 4 } } }
	in_stream_V_strb_V { axis {  { in_stream_TSTRB in_data 0 4 } } }
	in_stream_V_last_V { axis {  { in_stream_TVALID in_vld 0 1 }  { in_stream_TREADY in_acc 1 1 }  { in_stream_TLAST in_data 0 1 } } }
	out_stream_V_data_V { axis {  { out_stream_TDATA out_data 1 128 } } }
	out_stream_V_keep_V { axis {  { out_stream_TKEEP out_data 1 16 } } }
	out_stream_V_strb_V { axis {  { out_stream_TSTRB out_data 1 16 } } }
	out_stream_V_last_V { axis {  { out_stream_TVALID out_vld 1 1 }  { out_stream_TREADY out_acc 0 1 }  { out_stream_TLAST out_data 1 1 } } }
}

set maxi_interface_dict [dict create]

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
