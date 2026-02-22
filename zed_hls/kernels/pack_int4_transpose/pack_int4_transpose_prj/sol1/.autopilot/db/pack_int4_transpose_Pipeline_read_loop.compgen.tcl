# This script segment is generated automatically by AutoPilot

# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 1 \
    name in_stream_V_data_V \
    reset_level 1 \
    sync_rst true \
    corename {in_stream} \
    metadata {  } \
    op interface \
    ports { in_stream_TVALID { I 1 bit } in_stream_TDATA { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'in_stream_V_data_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 2 \
    name in_stream_V_keep_V \
    reset_level 1 \
    sync_rst true \
    corename {in_stream} \
    metadata {  } \
    op interface \
    ports { in_stream_TKEEP { I 4 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'in_stream_V_keep_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 3 \
    name in_stream_V_strb_V \
    reset_level 1 \
    sync_rst true \
    corename {in_stream} \
    metadata {  } \
    op interface \
    ports { in_stream_TSTRB { I 4 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'in_stream_V_strb_V'"
}
}


# Native AXIS:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::native_axis_add] == "::AESL_LIB_XILADAPTER::native_axis_add"} {
eval "::AESL_LIB_XILADAPTER::native_axis_add { \
    id 4 \
    name in_stream_V_last_V \
    reset_level 1 \
    sync_rst true \
    corename {in_stream} \
    metadata {  } \
    op interface \
    ports { in_stream_TREADY { O 1 bit } in_stream_TLAST { I 1 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'in_stream_V_last_V'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 5 \
    name buffer_31_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_31_out \
    op interface \
    ports { buffer_31_out { O 4 vector } buffer_31_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 6 \
    name buffer_30_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_30_out \
    op interface \
    ports { buffer_30_out { O 4 vector } buffer_30_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 7 \
    name buffer_29_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_29_out \
    op interface \
    ports { buffer_29_out { O 4 vector } buffer_29_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 8 \
    name buffer_28_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_28_out \
    op interface \
    ports { buffer_28_out { O 4 vector } buffer_28_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 9 \
    name buffer_27_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_27_out \
    op interface \
    ports { buffer_27_out { O 4 vector } buffer_27_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 10 \
    name buffer_26_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_26_out \
    op interface \
    ports { buffer_26_out { O 4 vector } buffer_26_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11 \
    name buffer_25_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_25_out \
    op interface \
    ports { buffer_25_out { O 4 vector } buffer_25_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 12 \
    name buffer_24_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_24_out \
    op interface \
    ports { buffer_24_out { O 4 vector } buffer_24_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 13 \
    name buffer_23_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_23_out \
    op interface \
    ports { buffer_23_out { O 4 vector } buffer_23_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 14 \
    name buffer_22_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_22_out \
    op interface \
    ports { buffer_22_out { O 4 vector } buffer_22_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 15 \
    name buffer_21_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_21_out \
    op interface \
    ports { buffer_21_out { O 4 vector } buffer_21_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 16 \
    name buffer_20_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_20_out \
    op interface \
    ports { buffer_20_out { O 4 vector } buffer_20_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 17 \
    name buffer_19_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_19_out \
    op interface \
    ports { buffer_19_out { O 4 vector } buffer_19_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 18 \
    name buffer_18_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_18_out \
    op interface \
    ports { buffer_18_out { O 4 vector } buffer_18_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 19 \
    name buffer_17_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_17_out \
    op interface \
    ports { buffer_17_out { O 4 vector } buffer_17_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 20 \
    name buffer_16_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_16_out \
    op interface \
    ports { buffer_16_out { O 4 vector } buffer_16_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 21 \
    name buffer_15_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_15_out \
    op interface \
    ports { buffer_15_out { O 4 vector } buffer_15_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 22 \
    name buffer_14_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_14_out \
    op interface \
    ports { buffer_14_out { O 4 vector } buffer_14_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 23 \
    name buffer_13_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_13_out \
    op interface \
    ports { buffer_13_out { O 4 vector } buffer_13_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 24 \
    name buffer_12_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_12_out \
    op interface \
    ports { buffer_12_out { O 4 vector } buffer_12_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 25 \
    name buffer_11_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_11_out \
    op interface \
    ports { buffer_11_out { O 4 vector } buffer_11_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 26 \
    name buffer_10_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_10_out \
    op interface \
    ports { buffer_10_out { O 4 vector } buffer_10_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 27 \
    name buffer_9_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_9_out \
    op interface \
    ports { buffer_9_out { O 4 vector } buffer_9_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 28 \
    name buffer_8_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_8_out \
    op interface \
    ports { buffer_8_out { O 4 vector } buffer_8_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 29 \
    name buffer_7_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_7_out \
    op interface \
    ports { buffer_7_out { O 4 vector } buffer_7_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 30 \
    name buffer_6_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_6_out \
    op interface \
    ports { buffer_6_out { O 4 vector } buffer_6_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 31 \
    name buffer_5_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_5_out \
    op interface \
    ports { buffer_5_out { O 4 vector } buffer_5_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 32 \
    name buffer_4_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_4_out \
    op interface \
    ports { buffer_4_out { O 4 vector } buffer_4_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 33 \
    name buffer_3_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_3_out \
    op interface \
    ports { buffer_3_out { O 4 vector } buffer_3_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 34 \
    name buffer_2_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_2_out \
    op interface \
    ports { buffer_2_out { O 4 vector } buffer_2_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 35 \
    name buffer_1_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_1_out \
    op interface \
    ports { buffer_1_out { O 4 vector } buffer_1_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 36 \
    name buffer_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_buffer_out \
    op interface \
    ports { buffer_out { O 4 vector } buffer_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


# flow_control definition:
set InstName pack_int4_transpose_flow_control_loop_pipe_sequential_init_U
set CompName pack_int4_transpose_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix pack_int4_transpose_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


