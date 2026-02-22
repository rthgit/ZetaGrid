set moduleName kv_page_ops
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
set C_modelName {kv_page_ops}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ cmd_stream_V_data_V int 64 regular {axi_s 0 volatile  { cmd_stream Data } }  }
	{ cmd_stream_V_keep_V int 8 regular {axi_s 0 volatile  { cmd_stream Keep } }  }
	{ cmd_stream_V_strb_V int 8 regular {axi_s 0 volatile  { cmd_stream Strb } }  }
	{ cmd_stream_V_last_V int 1 regular {axi_s 0 volatile  { cmd_stream Last } }  }
	{ resp_stream_V_data_V int 32 regular {axi_s 1 volatile  { resp_stream Data } }  }
	{ resp_stream_V_keep_V int 4 regular {axi_s 1 volatile  { resp_stream Keep } }  }
	{ resp_stream_V_strb_V int 4 regular {axi_s 1 volatile  { resp_stream Strb } }  }
	{ resp_stream_V_last_V int 1 regular {axi_s 1 volatile  { resp_stream Last } }  }
	{ num_cmds int 32 regular {axi_slave 0}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "cmd_stream_V_data_V", "interface" : "axis", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "cmd_stream_V_keep_V", "interface" : "axis", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "cmd_stream_V_strb_V", "interface" : "axis", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "cmd_stream_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "resp_stream_V_data_V", "interface" : "axis", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "resp_stream_V_keep_V", "interface" : "axis", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "resp_stream_V_strb_V", "interface" : "axis", "bitwidth" : 4, "direction" : "WRITEONLY"} , 
 	{ "Name" : "resp_stream_V_last_V", "interface" : "axis", "bitwidth" : 1, "direction" : "WRITEONLY"} , 
 	{ "Name" : "num_cmds", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} ]}
# RTL Port declarations: 
set portNum 32
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ cmd_stream_TVALID sc_in sc_logic 1 invld 3 } 
	{ resp_stream_TREADY sc_in sc_logic 1 outacc 7 } 
	{ cmd_stream_TDATA sc_in sc_lv 64 signal 0 } 
	{ cmd_stream_TREADY sc_out sc_logic 1 inacc 3 } 
	{ cmd_stream_TKEEP sc_in sc_lv 8 signal 1 } 
	{ cmd_stream_TSTRB sc_in sc_lv 8 signal 2 } 
	{ cmd_stream_TLAST sc_in sc_lv 1 signal 3 } 
	{ resp_stream_TDATA sc_out sc_lv 32 signal 4 } 
	{ resp_stream_TVALID sc_out sc_logic 1 outvld 7 } 
	{ resp_stream_TKEEP sc_out sc_lv 4 signal 5 } 
	{ resp_stream_TSTRB sc_out sc_lv 4 signal 6 } 
	{ resp_stream_TLAST sc_out sc_lv 1 signal 7 } 
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
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"kv_page_ops","role":"start","value":"0","valid_bit":"0"},{"name":"kv_page_ops","role":"continue","value":"0","valid_bit":"4"},{"name":"kv_page_ops","role":"auto_start","value":"0","valid_bit":"7"},{"name":"num_cmds","role":"data","value":"16"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"kv_page_ops","role":"start","value":"0","valid_bit":"0"},{"name":"kv_page_ops","role":"done","value":"0","valid_bit":"1"},{"name":"kv_page_ops","role":"idle","value":"0","valid_bit":"2"},{"name":"kv_page_ops","role":"ready","value":"0","valid_bit":"3"},{"name":"kv_page_ops","role":"auto_start","value":"0","valid_bit":"7"}] },
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
 	{ "name": "cmd_stream_TVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "cmd_stream_V_last_V", "role": "default" }} , 
 	{ "name": "resp_stream_TREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "outacc", "bundle":{"name": "resp_stream_V_last_V", "role": "default" }} , 
 	{ "name": "cmd_stream_TDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "cmd_stream_V_data_V", "role": "default" }} , 
 	{ "name": "cmd_stream_TREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "inacc", "bundle":{"name": "cmd_stream_V_last_V", "role": "default" }} , 
 	{ "name": "cmd_stream_TKEEP", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "cmd_stream_V_keep_V", "role": "default" }} , 
 	{ "name": "cmd_stream_TSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "cmd_stream_V_strb_V", "role": "default" }} , 
 	{ "name": "cmd_stream_TLAST", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmd_stream_V_last_V", "role": "default" }} , 
 	{ "name": "resp_stream_TDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "resp_stream_V_data_V", "role": "default" }} , 
 	{ "name": "resp_stream_TVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "resp_stream_V_last_V", "role": "default" }} , 
 	{ "name": "resp_stream_TKEEP", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "resp_stream_V_keep_V", "role": "default" }} , 
 	{ "name": "resp_stream_TSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "resp_stream_V_strb_V", "role": "default" }} , 
 	{ "name": "resp_stream_TLAST", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "resp_stream_V_last_V", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
		"CDFG" : "kv_page_ops",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "cmd_stream_V_data_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "cmd_stream",
				"BlockSignal" : [
					{"Name" : "cmd_stream_TDATA_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "cmd_stream_V_keep_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "cmd_stream"},
			{"Name" : "cmd_stream_V_strb_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "cmd_stream"},
			{"Name" : "cmd_stream_V_last_V", "Type" : "Axis", "Direction" : "I", "BaseName" : "cmd_stream"},
			{"Name" : "resp_stream_V_data_V", "Type" : "Axis", "Direction" : "O", "BaseName" : "resp_stream",
				"BlockSignal" : [
					{"Name" : "resp_stream_TDATA_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "resp_stream_V_keep_V", "Type" : "Axis", "Direction" : "O", "BaseName" : "resp_stream"},
			{"Name" : "resp_stream_V_strb_V", "Type" : "Axis", "Direction" : "O", "BaseName" : "resp_stream"},
			{"Name" : "resp_stream_V_last_V", "Type" : "Axis", "Direction" : "O", "BaseName" : "resp_stream"},
			{"Name" : "num_cmds", "Type" : "None", "Direction" : "I"},
			{"Name" : "kv_memory", "Type" : "Memory", "Direction" : "IO"}],
		"Loop" : [
			{"Name" : "process_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter3", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter3", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kv_memory_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.control_s_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_cmd_stream_V_data_V_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_cmd_stream_V_keep_V_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_cmd_stream_V_strb_V_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_cmd_stream_V_last_V_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_resp_stream_V_data_V_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_resp_stream_V_keep_V_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_resp_stream_V_strb_V_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.regslice_both_resp_stream_V_last_V_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	kv_page_ops {
		cmd_stream_V_data_V {Type I LastRead 0 FirstWrite -1}
		cmd_stream_V_keep_V {Type I LastRead 0 FirstWrite -1}
		cmd_stream_V_strb_V {Type I LastRead 0 FirstWrite -1}
		cmd_stream_V_last_V {Type I LastRead 0 FirstWrite -1}
		resp_stream_V_data_V {Type O LastRead -1 FirstWrite 2}
		resp_stream_V_keep_V {Type O LastRead -1 FirstWrite 2}
		resp_stream_V_strb_V {Type O LastRead -1 FirstWrite 2}
		resp_stream_V_last_V {Type O LastRead -1 FirstWrite 2}
		num_cmds {Type I LastRead 0 FirstWrite -1}
		kv_memory {Type IO LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	cmd_stream_V_data_V { axis {  { cmd_stream_TDATA in_data 0 64 } } }
	cmd_stream_V_keep_V { axis {  { cmd_stream_TKEEP in_data 0 8 } } }
	cmd_stream_V_strb_V { axis {  { cmd_stream_TSTRB in_data 0 8 } } }
	cmd_stream_V_last_V { axis {  { cmd_stream_TVALID in_vld 0 1 }  { cmd_stream_TREADY in_acc 1 1 }  { cmd_stream_TLAST in_data 0 1 } } }
	resp_stream_V_data_V { axis {  { resp_stream_TREADY out_acc 0 1 }  { resp_stream_TDATA out_data 1 32 } } }
	resp_stream_V_keep_V { axis {  { resp_stream_TKEEP out_data 1 4 } } }
	resp_stream_V_strb_V { axis {  { resp_stream_TSTRB out_data 1 4 } } }
	resp_stream_V_last_V { axis {  { resp_stream_TVALID out_vld 1 1 }  { resp_stream_TLAST out_data 1 1 } } }
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
