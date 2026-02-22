open_project kv_page_ops_prj

add_files kernel.cpp
add_files -tb tb.cpp

set_top kv_page_ops

open_solution "sol1" -flow_target vivado

# Target Part: VCU9P
set_part {xcvu9p-flga2104-2-i}

# Clock: 300MHz
create_clock -period 3.33 -name default

config_export -format ip_catalog -rtl verilog

# 1. C Simulation
csim_design

# 2. Synthesis
csynth_design

close_project
exit
