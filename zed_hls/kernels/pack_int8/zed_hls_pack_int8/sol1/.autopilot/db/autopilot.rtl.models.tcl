set SynModuleInfo {
  {SRCNAME __hls_fptosi_float_i8 MODELNAME p_hls_fptosi_float_i8 RTLNAME pack_int8_p_hls_fptosi_float_i8}
  {SRCNAME pack_int8_Pipeline_pack_loop MODELNAME pack_int8_Pipeline_pack_loop RTLNAME pack_int8_pack_int8_Pipeline_pack_loop
    SUBMODULES {
      {MODELNAME pack_int8_flow_control_loop_pipe_sequential_init RTLNAME pack_int8_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME pack_int8_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME pack_int8 MODELNAME pack_int8 RTLNAME pack_int8 IS_TOP 1
    SUBMODULES {
      {MODELNAME pack_int8_control_s_axi RTLNAME pack_int8_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
      {MODELNAME pack_int8_regslice_both RTLNAME pack_int8_regslice_both BINDTYPE interface TYPE adapter IMPL reg_slice}
    }
  }
}
