open_project -reset zed_hls_pack_int8
set_top pack_int8

add_files kernel.cpp
add_files -tb tb.cpp

open_solution -reset sol1
# Target Part: Zynq UltraScale+ MPSoC (Generic)
# Fallback to Virtex UltraScale+ if Zynq is missing
set_part {xcvu9p-flga2104-2-e}
create_clock -period 4.0 -name default

# Optimization
config_compile -pipeline_loops 64

# Execution Steps
csim_design -clean
csynth_design
# eps = export_design (optional, creates IP)
# export_design -format ip_catalog

exit
