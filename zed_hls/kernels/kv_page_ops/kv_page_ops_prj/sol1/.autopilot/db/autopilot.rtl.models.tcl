set SynModuleInfo {
  {SRCNAME kv_page_ops MODELNAME kv_page_ops RTLNAME kv_page_ops IS_TOP 1
    SUBMODULES {
      {MODELNAME kv_page_ops_kv_memory_RAM_2P_BRAM_1R1W RTLNAME kv_page_ops_kv_memory_RAM_2P_BRAM_1R1W BINDTYPE storage TYPE ram_2p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME kv_page_ops_control_s_axi RTLNAME kv_page_ops_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
      {MODELNAME kv_page_ops_regslice_both RTLNAME kv_page_ops_regslice_both BINDTYPE interface TYPE adapter IMPL reg_slice}
      {MODELNAME kv_page_ops_flow_control_loop_pipe RTLNAME kv_page_ops_flow_control_loop_pipe BINDTYPE interface TYPE internal_upc_flow_control INSTNAME kv_page_ops_flow_control_loop_pipe_U}
    }
  }
}
