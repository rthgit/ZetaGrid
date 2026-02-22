set SynModuleInfo {
  {SRCNAME pack_int4_transpose_Pipeline_read_loop MODELNAME pack_int4_transpose_Pipeline_read_loop RTLNAME pack_int4_transpose_pack_int4_transpose_Pipeline_read_loop
    SUBMODULES {
      {MODELNAME pack_int4_transpose_flow_control_loop_pipe_sequential_init RTLNAME pack_int4_transpose_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME pack_int4_transpose_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME pack_int4_transpose MODELNAME pack_int4_transpose RTLNAME pack_int4_transpose IS_TOP 1
    SUBMODULES {
      {MODELNAME pack_int4_transpose_control_s_axi RTLNAME pack_int4_transpose_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
      {MODELNAME pack_int4_transpose_regslice_both RTLNAME pack_int4_transpose_regslice_both BINDTYPE interface TYPE adapter IMPL reg_slice}
    }
  }
}
